#!/usr/bin/env python3
"""测试新配置格式的加载和向下兼容性"""

import sys
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 直接测试配置加载
def test_config_loading():
    print("=" * 60)
    print("测试新配置格式的加载和向下兼容性")
    print("=" * 60)

    try:
        # 导入配置模块（绕过 nonebot 依赖）
        import importlib.util

        # 直接加载 config.py 模块
        config_path = project_root / "src" / "plugins" / "yuying_chameleon" / "config.py"
        spec = importlib.util.spec_from_file_location("config_test", config_path)
        config_module = importlib.util.module_from_spec(spec)

        # Mock nonebot logger
        import logging

        class MockLogger:
            def __init__(self):
                self.logger = logging.getLogger("test")
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

            def info(self, msg, *args):
                self.logger.info(msg.format(*args) if args else msg)

            def warning(self, msg, *args):
                self.logger.warning(msg.format(*args) if args else msg)

            def error(self, msg, *args):
                self.logger.error(msg.format(*args) if args else msg)

        # Mock nonebot 模块
        sys.modules['nonebot'] = type(sys)('nonebot')
        sys.modules['nonebot'].logger = MockLogger()
        sys.modules['nonebot.log'] = type(sys)('nonebot.log')

        # 加载配置模块
        spec.loader.exec_module(config_module)

        # 获取配置对象
        cfg = config_module.plugin_config

        print("\n=== 旧字段（向下兼容） ===")
        print(f"main 模型: {cfg.yuying_openai_model}")
        print(f"main base_url: {cfg.yuying_openai_base_url}")
        print(f"main API key: {'已配置 (' + cfg.yuying_openai_api_key[:20] + '...)' if cfg.yuying_openai_api_key else '未配置'}")
        print(f"main timeout: {cfg.yuying_openai_timeout}")

        print(f"\ncheap 模型: {cfg.yuying_cheap_llm_model}")
        print(f"cheap base_url: {cfg.yuying_cheap_llm_base_url}")
        print(f"cheap API key: {'已配置 (' + cfg.yuying_cheap_llm_api_key[:20] + '...)' if cfg.yuying_cheap_llm_api_key else '未配置'}")
        print(f"cheap timeout: {cfg.yuying_cheap_llm_timeout}")

        print(f"\nnano 模型: {cfg.yuying_nano_llm_model}")
        print(f"nano base_url: {cfg.yuying_nano_llm_base_url}")
        print(f"nano API key: {'已配置 (' + cfg.yuying_nano_llm_api_key[:20] + '...)' if cfg.yuying_nano_llm_api_key else '未配置'}")
        print(f"nano timeout: {cfg.yuying_nano_llm_timeout}")

        print("\n=== 新格式（yuying_llm） ===")
        if cfg.yuying_llm:
            print(f"新格式已配置")

            if cfg.yuying_llm.main:
                main_group = cfg.yuying_llm.main
                if main_group.models:
                    print(f"\nmain 模型组:")
                    print(f"  模型数量: {len(main_group.models)}")
                    for i, model in enumerate(main_group.models[:3]):  # 只显示前3个
                        if hasattr(model, 'model'):
                            print(f"  [{i+1}] {model.model} @ {model.base_url}")
                        else:
                            print(f"  [{i+1}] {model}")

            if cfg.yuying_llm.cheap:
                cheap_group = cfg.yuying_llm.cheap
                if cheap_group.models:
                    print(f"\ncheap 模型组:")
                    print(f"  模型数量: {len(cheap_group.models)}")
                    for i, model in enumerate(cheap_group.models):
                        if hasattr(model, 'model'):
                            print(f"  [{i+1}] {model.model} @ {model.base_url}")
                        else:
                            print(f"  [{i+1}] {model}")

            if cfg.yuying_llm.nano:
                nano_group = cfg.yuying_llm.nano
                if nano_group.models:
                    print(f"\nnano 模型组:")
                    print(f"  模型数量: {len(nano_group.models)}")
                    for i, model in enumerate(nano_group.models):
                        if hasattr(model, 'model'):
                            print(f"  [{i+1}] {model.model} @ {model.base_url}")
                        else:
                            print(f"  [{i+1}] {model}")
        else:
            print("未配置新格式")

        print("\n" + "=" * 60)
        print("✓ 配置加载成功！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)
