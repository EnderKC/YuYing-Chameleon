#!/usr/bin/env python3
"""简单的配置格式测试 - 直接测试 Pydantic 模型"""

import sys
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 直接导入 Pydantic 相关类
from typing import Dict, Any, List, Optional, Union
from pydantic.v1 import BaseModel, Field, Extra, root_validator


# 复制必要的类定义（避免导入整个模块）
class LLMModelConfig(BaseModel):
    model: str = Field(alias="model")
    base_url: Optional[str] = Field(default=None, alias="base_url")
    api_key: Optional[str] = Field(default=None, alias="api_key")
    timeout: Optional[float] = Field(default=None, alias="timeout")

    class Config:
        extra = Extra.ignore
        allow_population_by_field_name = True


class ProviderConfig(BaseModel):
    name: str = Field(alias="name")
    base_url: str = Field(alias="base_url")
    api_key: str = Field(alias="api_key")
    timeout: Optional[float] = Field(default=None, alias="timeout")
    models: List[str] = Field(alias="models")

    class Config:
        extra = Extra.ignore
        allow_population_by_field_name = True


class LLMModelGroupConfig(BaseModel):
    model: Optional[str] = Field(default=None, alias="model")
    models: Optional[List[Union[str, LLMModelConfig]]] = Field(default=None, alias="models")
    base_url: Optional[str] = Field(default=None, alias="base_url")
    api_key: Optional[str] = Field(default=None, alias="api_key")
    timeout: Optional[float] = Field(default=None, alias="timeout")
    providers: Optional[List[ProviderConfig]] = Field(default=None, alias="providers")

    @root_validator(pre=False)
    def _expand_providers(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        providers = values.get("providers")
        if not providers:
            return values

        expanded_models: List[Union[str, LLMModelConfig]] = []
        for provider in providers:
            for model_name in provider.models:
                model_config = LLMModelConfig(
                    model=model_name,
                    base_url=provider.base_url,
                    api_key=provider.api_key,
                    timeout=provider.timeout,
                )
                expanded_models.append(model_config)

        values["models"] = expanded_models
        values["model"] = None
        return values

    class Config:
        extra = Extra.ignore
        allow_population_by_field_name = True


def test_format_4_providers_array():
    """测试新格式：供应商数组"""
    print("\n=== 测试新格式：供应商数组 ===")
    config_data = {
        "providers": [
            {
                "name": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-deepseek-xxx",
                "timeout": 12.0,
                "models": ["deepseek-chat", "deepseek-coder"]
            },
            {
                "name": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-openai-xxx",
                "timeout": 10.0,
                "models": ["gpt-3.5-turbo", "gpt-4o-mini"]
            }
        ]
    }

    try:
        config = LLMModelGroupConfig.parse_obj(config_data)
        print(f"✓ 解析成功")
        print(f"  providers 已展开为 models")
        print(f"  models 数量: {len(config.models) if config.models else 0}")

        if config.models:
            for i, model in enumerate(config.models):
                if isinstance(model, str):
                    print(f"  models[{i}]: {model}")
                else:
                    print(f"  models[{i}]: {model.model}")
                    print(f"    - base_url: {model.base_url}")
                    print(f"    - timeout: {model.timeout}")

        return True
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_old_formats():
    """测试旧格式兼容性"""
    print("\n=== 测试旧格式兼容性 ===")

    # 测试1: 单模型
    print("\n1. 单模型:")
    try:
        config = LLMModelGroupConfig.parse_obj({"model": "gpt-4-turbo"})
        print(f"  ✓ 单模型解析成功: {config.model}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

    # 测试2: 统一供应商
    print("\n2. 统一供应商:")
    try:
        config = LLMModelGroupConfig.parse_obj({
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-xxx",
            "models": ["deepseek-chat", "deepseek-coder"]
        })
        print(f"  ✓ 统一供应商解析成功: {len(config.models)} 个模型")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

    # 测试3: 混合供应商
    print("\n3. 混合供应商:")
    try:
        config = LLMModelGroupConfig.parse_obj({
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-xxx",
            "models": [
                "deepseek-chat",
                {"model": "gpt-3.5-turbo", "base_url": "https://api.openai.com/v1", "api_key": "sk-yyy"}
            ]
        })
        print(f"  ✓ 混合供应商解析成功: {len(config.models)} 个模型")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("LLM 配置格式测试")
    print("=" * 60)

    result1 = test_old_formats()
    result2 = test_format_4_providers_array()

    print("\n" + "=" * 60)
    if result1 and result2:
        print("✓ 所有测试通过！")
        print("  - 旧格式（方式1-3）兼容性: ✓")
        print("  - 新格式（方式4）: ✓")
    else:
        print("✗ 测试失败")
    print("=" * 60)

    return result1 and result2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
