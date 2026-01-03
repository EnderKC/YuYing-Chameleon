"""JSON Schema 转换器：把 MCP tool.inputSchema 转成 OpenAI tools 的 parameters schema。

核心策略：尽力转换 + 安全降级。

为什么需要"尽力 + 降级"：
- MCP tool.inputSchema 通常是完整 JSON Schema（可能包含 $ref/oneOf/anyOf/递归/复杂关键字）。
- OpenAI function calling 虽使用 JSON Schema 形态，但实际支持子集；过于复杂的 schema 可能导致：
  1) SDK/服务端校验失败
  2) 模型理解困难，生成 arguments 更容易偏离
  3) tokens 成本增加

因此这里做：
- 允许关键字白名单（type/properties/items/required/enum/...）
- 对 oneOf/anyOf/allOf：尽力"展平"为更宽松的 object schema，并返回 note 提示约束被放宽
- 对 $ref 等无法处理：剔除并 note
- 永不抛异常：最坏返回 {"type":"object","properties":{}}（保证工具可用）
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


_ALLOWED_KEYS: set[str] = {
    # 结构
    "type",
    "properties",
    "required",
    "items",
    "additionalProperties",
    # 约束
    "description",
    "enum",
    "default",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "pattern",
}


def _as_dict(obj: Any) -> Dict[str, Any]:
    return dict(obj) if isinstance(obj, dict) else {}


def _is_object_schema(schema: Dict[str, Any]) -> bool:
    t = schema.get("type")
    if t == "object":
        return True
    # JSON Schema 允许 type 省略，此时通常也能视为 object（更宽松）
    return t is None


def _merge_object_variants(variants: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """把 oneOf/anyOf 的多个 object 变体合并成一个更宽松的 object schema。

合并策略（保守但实用）：
- properties：并集
- required：不取交集也不取并集（统一清空），避免模型被"必填"卡死
- additionalProperties：若任一 variant 明确 True，则 True；否则默认 True（更宽松）
"""
    merged_props: Dict[str, Any] = {}
    notes: List[str] = []
    for idx, v in enumerate(variants):
        props = v.get("properties")
        if isinstance(props, dict):
            for k, sub in props.items():
                if k not in merged_props and isinstance(k, str) and k.strip():
                    merged_props[k] = sub
        if v.get("required"):
            notes.append(f"variant#{idx} had required fields")

    merged: Dict[str, Any] = {
        "type": "object",
        "properties": merged_props,
        "required": [],
        "additionalProperties": True,
    }
    note = "Flattened oneOf/anyOf to a looser object schema"
    if notes:
        note = f"{note}; required constraints were relaxed"
    return merged, note


def _strip_unsupported(schema: Dict[str, Any], *, note_out: List[str]) -> Dict[str, Any]:
    """移除不在白名单中的 key，并把重要的被移除项写到 note。"""
    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in _ALLOWED_KEYS:
            out[k] = v
        else:
            # 常见"风险关键字"：尽量提示
            if k in {"$schema", "$id", "title"}:
                continue
            if k in {"$ref", "definitions", "$defs"}:
                note_out.append(f"Removed unsupported keyword: {k}")
            elif k in {"oneOf", "anyOf", "allOf", "not"}:
                # 组合关键字在上层会处理，这里不重复记
                continue
            else:
                # 其他未知关键字不逐个写，避免 note 过长
                continue
    return out


def _normalize_schema(
    schema: Dict[str, Any],
    *,
    depth: int,
    note_out: List[str],
    max_depth: int = 6,
    max_properties: int = 80,
) -> Dict[str, Any]:
    """递归规范化 schema，保证输出尽量落在 OpenAI tools 可接受的子集里。"""
    if depth > max_depth:
        note_out.append(f"Schema depth>{max_depth}, truncated")
        return {"type": "object", "properties": {}, "additionalProperties": True}

    # 组合关键字：尽力展平
    for comb in ("oneOf", "anyOf"):
        variants = schema.get(comb)
        if isinstance(variants, list) and variants:
            norm_variants: List[Dict[str, Any]] = []
            for v in variants:
                dv = _as_dict(v)
                if not dv:
                    continue
                dv2 = _normalize_schema(dv, depth=depth + 1, note_out=note_out)
                norm_variants.append(dv2)
            if norm_variants and all(_is_object_schema(v) for v in norm_variants):
                merged, note = _merge_object_variants(norm_variants)
                note_out.append(note)
                schema = merged
            else:
                note_out.append(f"{comb} present but not mergeable; downgraded to object")
                schema = {"type": "object", "properties": {}, "additionalProperties": True}
            break

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        # allOf：尽力当作多个对象约束的叠加；这里同样放宽成 properties 并集
        variants: List[Dict[str, Any]] = []
        for v in all_of:
            dv = _as_dict(v)
            if not dv:
                continue
            variants.append(_normalize_schema(dv, depth=depth + 1, note_out=note_out))
        if variants and all(_is_object_schema(v) for v in variants):
            merged, _ = _merge_object_variants(variants)
            note_out.append("Flattened allOf to a looser object schema")
            schema = merged
        else:
            note_out.append("allOf present but not mergeable; downgraded to object")
            schema = {"type": "object", "properties": {}, "additionalProperties": True}

    # 去除不支持 key
    schema = _strip_unsupported(schema, note_out=note_out)

    # 补默认 type
    t = schema.get("type")
    if t is None:
        schema["type"] = "object"
        t = "object"

    # 处理 type 为列表的情况（如 ["string", "null"]）
    if isinstance(t, list):
        original_types = [x for x in t if isinstance(x, str)]
        if not original_types:
            schema["type"] = "object"
            t = "object"
            note_out.append("type array had no usable string types; downgraded to object")
        else:
            # 过滤掉 "null"，取第一个非 null 类型
            non_null_types = [x for x in original_types if x != "null"]
            if non_null_types:
                schema["type"] = non_null_types[0]
                t = non_null_types[0]
                if len(non_null_types) > 1:
                    note_out.append(f"type array {original_types} reduced to {schema['type']}")
            else:
                # 全是 null，降级为 object
                schema["type"] = "object"
                t = "object"
                note_out.append("type array was all null; downgraded to object")

    # 只处理最常见类型；其他类型（如 integer/number/string/boolean）按白名单直接通过
    if t == "object":
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = {}
        # 限制属性数，避免把巨大 schema 注入 tools（token/模型理解成本）
        keys = list(props.keys())
        if len(keys) > max_properties:
            note_out.append(f"properties>{max_properties}, truncated")
            keys = keys[:max_properties]
        norm_props: Dict[str, Any] = {}
        for k in keys:
            if not isinstance(k, str) or not k.strip():
                continue
            sub = props.get(k)
            dsub = _as_dict(sub)
            if not dsub:
                # 若子 schema 无效，降级为任意 object
                norm_props[k] = {"type": "object", "additionalProperties": True}
                continue
            norm_props[k] = _normalize_schema(dsub, depth=depth + 1, note_out=note_out)
        schema["properties"] = norm_props

        req = schema.get("required")
        if isinstance(req, list):
            # required 只保留存在于 properties 的字段
            schema["required"] = [x for x in req if isinstance(x, str) and x in norm_props]
        else:
            schema["required"] = []

        # additionalProperties：OpenAI 通常可接受 bool 或 schema；这里尽量简化为 bool
        ap = schema.get("additionalProperties")
        if isinstance(ap, dict):
            schema["additionalProperties"] = True
            note_out.append("additionalProperties schema downgraded to boolean True")
        elif isinstance(ap, bool):
            pass
        else:
            schema["additionalProperties"] = True

    if t == "array":
        items = schema.get("items")
        ditems = _as_dict(items)
        if ditems:
            schema["items"] = _normalize_schema(ditems, depth=depth + 1, note_out=note_out)
        else:
            schema["items"] = {"type": "object", "additionalProperties": True}

    return schema


def convert_mcp_input_schema_to_openai_parameters(input_schema: Any) -> Tuple[Dict[str, Any], str]:
    """把 MCP inputSchema 转换为 OpenAI tools 的 parameters（JSON Schema object）。

返回：
  (parameters_schema, note)

note 用途：
- 当发生降级/裁剪/移除关键字时，返回简短说明，便于把约束补到 tool description 里。
"""
    schema = _as_dict(input_schema)
    notes: List[str] = []

    # MCP 可能直接给空 schema：按"无参数 object"处理
    if not schema:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": True}, ""

    normalized = _normalize_schema(schema, depth=0, note_out=notes)

    # OpenAI function calling 的 parameters 期望顶层是 object
    if normalized.get("type") != "object":
        notes.append("Top-level schema was not object; wrapped into object as `input`")
        normalized = {
            "type": "object",
            "properties": {"input": normalized},
            "required": ["input"],
            "additionalProperties": False,
        }

    note = "; ".join(dict.fromkeys([x for x in notes if x.strip()]))[:400]
    return normalized, note
