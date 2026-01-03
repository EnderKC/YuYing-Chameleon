"""内部工具模块：为 LLM 提供机器人内部功能工具。

说明：
- 内部工具与 MCP 工具并存，内部工具总是可用
- 工具通过 OpenAI function calling 机制调用
- 工具名使用前缀避免冲突：internal__<tool_name>
"""

from .internal_tools_manager import internal_tools_manager

__all__ = ["internal_tools_manager"]
