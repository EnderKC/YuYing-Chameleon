#!/usr/bin/env python3
"""测试 LLM 配置格式的兼容性

测试目标：
1. 旧格式（方式1、2、3）是否仍然可以正常解析
2. 新格式（方式4：providers数组）是否可以正常解析
3. providers 是否能正确展开为 models 格式
"""

from typing import Dict, Any
from src.plugins.yuying_chameleon.config import LLMModelGroupConfig, ProviderConfig


def test_format_1_single_model():
    """测试方式1：单模型配置"""
    print("\n=== 测试方式1：单模型 ===")
    config_data = {
        "model": "gpt-4-turbo"
    }

    try:
        config = LLMModelGroupConfig.parse_obj(config_data)
        print(f"✓ 解析成功")
        print(f"  model: {config.model}")
        print(f"  models: {config.models}")
        return True
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        return False


def test_format_2_unified_provider():
    """测试方式2：统一供应商配置"""
    print("\n=== 测试方式2：统一供应商 ===")
    config_data = {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-deepseek-xxx",
        "timeout": 12.0,
        "models": ["deepseek-chat", "deepseek-coder"]
    }

    try:
        config = LLMModelGroupConfig.parse_obj(config_data)
        print(f"✓ 解析成功")
        print(f"  base_url: {config.base_url}")
        print(f"  models: {config.models}")
        return True
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        return False


def test_format_3_mixed_providers():
    """测试方式3：混合多供应商（旧格式）"""
    print("\n=== 测试方式3：混合多供应商（旧格式） ===")
    config_data = {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-deepseek-xxx",
        "timeout": 12.0,
        "models": [
            "deepseek-chat",
            {
                "model": "gpt-3.5-turbo",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-openai-xxx",
                "timeout": 10.0
            }
        ]
    }

    try:
        config = LLMModelGroupConfig.parse_obj(config_data)
        print(f"✓ 解析成功")
        print(f"  models 数量: {len(config.models)}")
        for i, model in enumerate(config.models):
            if isinstance(model, str):
                print(f"  models[{i}]: {model} (字符串)")
            else:
                print(f"  models[{i}]: {model.model} (对象)")
        return True
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        return False


def test_format_4_providers_array():
    """测试方式4：供应商数组（新格式）"""
    print("\n=== 测试方式4：供应商数组（新格式） ===")
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
        print(f"  models 数量: {len(config.models)}")

        # 验证展开后的 models
        for i, model in enumerate(config.models):
            if isinstance(model, str):
                print(f"  models[{i}]: {model} (字符串)")
            else:
                print(f"  models[{i}]: {model.model}")
                print(f"    - base_url: {model.base_url}")
                print(f"    - api_key: {model.api_key[:10]}...")
                print(f"    - timeout: {model.timeout}")

        return True
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("LLM 配置格式兼容性测试")
    print("=" * 60)

    results = []
    results.append(("方式1：单模型", test_format_1_single_model()))
    results.append(("方式2：统一供应商", test_format_2_unified_provider()))
    results.append(("方式3：混合多供应商（旧格式）", test_format_3_mixed_providers()))
    results.append(("方式4：供应商数组（新格式）", test_format_4_providers_array()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")

    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
