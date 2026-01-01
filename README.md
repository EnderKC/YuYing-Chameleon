<div style="text-align: center;">
    <img src="https://img.qwwq.top/i/2025/02/22/67b9d79d5a5c9.png" alt="语影 Logo">
</div>

<div style="text-align: center;">
    <img src="https://img.shields.io/github/issues/EnderKC/YuYing-Chameleon.svg" alt="GitHub issues" onclick="window.location.href='https://github.com/EnderKC/YuYing-Chameleon/issues'" style="cursor:pointer;">
    <img src="https://img.shields.io/github/stars/EnderKC/YuYing-Chameleon.svg" alt="GitHub stars" onclick="window.location.href='https://github.com/EnderKC/YuYing-Chameleon/stargazers'" style="cursor:pointer;">
</div>

# 语影 (YuYing-Chameleon)

## 📖 项目简介

语影是一个基于 NoneBot2 的智能 QQ 机器人，通过先进的 AI 技术实现自然、智能的群聊互动。它能够理解上下文、记住用户信息、智能决策回复时机，为群聊带来更加真实的对话体验。

## ✨ 核心功能

### 🧠 智能对话系统
- **上下文理解**：基于向量检索（RAG）技术，准确理解对话上下文和历史信息
- **长期记忆**：自动提取并保存用户的个人信息、偏好和习惯，支持个性化对话
- **记忆浓缩**：自动合并和归档记忆，保持记忆库精简高效
- **对话摘要**：自动生成对话窗口摘要，优化 LLM 上下文使用

### 🎯 心流模式（Flow Mode）
- **智能决策**：使用轻量级 nano 模型判断最佳回复时机
- **多模态支持**：支持视觉理解，能够"看懂"图片内容并做出合适反应
- **自然参与**：避免无意义回复，让机器人参与更加自然流畅

### 🎨 表情包系统
- **自动收集**：从群聊中智能识别并收集表情包
- **智能匹配**：根据对话意图和情绪自动选择合适的表情包
- **使用统计**：记录使用情况，优化表情包推荐

### 🛡️ 频率控制
- **防刷屏保护**：智能限流机制，避免过度回复
- **真人节奏**：模拟真人打字延迟，提升自然度
- **场景冷却**：针对不同场景（群聊/私聊）的独立冷却策略

## 🏗️ 技术架构

- **框架**：NoneBot2（异步 Python 机器人框架）
- **协议适配**：OneBot V11（QQ 协议）
- **数据库**：SQLite + SQLAlchemy（ORM）
- **向量存储**：Qdrant（向量检索）
- **LLM**：OpenAI 兼容 API（支持多种大语言模型）
- **向量化**：Embedding 模型（文本向量化）
- **包管理**：uv（现代 Python 包管理器）

## 📦 快速开始

### 环境要求
- Python 3.10+
- Qdrant 向量数据库
- LLM API（OpenAI 或兼容接口）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/EnderKC/YuYing-Chameleon.git
   cd YuYing-Chameleon
   ```

2. **安装依赖**
   ```bash
   # 使用 uv（推荐）
   pip install uv
   uv sync

   # 或使用 pip
   pip install -e .
   ```

3. **配置机器人**
   ```bash
   # 复制配置文件模板
   cp configs/config.example.toml configs/config.toml

   # 编辑配置文件，填入你的 API 密钥和机器人信息
   vim configs/config.toml
   ```

4. **启动 Qdrant**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

5. **运行机器人**
   ```bash
   nb run
   ```

## ⚙️ 核心配置

在 `configs/config.toml` 中配置以下关键项：

```toml
[yuying_chameleon]
# LLM 配置
openai_api_key = "your-api-key"
openai_model = "gpt-4-turbo"

# 向量化模型
embedder_model = "text-embedding-3-small"

# Qdrant 向量库
qdrant_host = "localhost"
qdrant_port = 6333

# 心流模式（可选）
flow_mode_enabled = true
nano_llm_model = "THUDM/GLM-4.1V-9B-Thinking"
nano_llm_api_key = "your-nano-api-key"
```

详细配置说明请参考 [运行逻辑说明文档.md](./运行逻辑说明文档.md)

## 📸 展示

<div style="text-align: center;">
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8f55a62c.jpg" alt="语影 1" width="500"/>
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8fa17300.jpg" alt="语影 2" width="500"/>
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8fc5992d.jpg" alt="语影 3" width="500"/>
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8ffb12d9.jpg" alt="语影 4" width="500"/>
</div>

## 📚 文档

- [运行逻辑说明文档](./运行逻辑说明文档.md) - 详细的架构说明和运行原理

## 🤝 贡献

欢迎任何形式的贡献！请提交 Issue 或 Pull Request。

## 🙏 致谢

- [1254qwer](https://github.com/1254qwer) - 提供丰富的表情包支持
- [nonebot](https://github.com/nonebot/nonebot) - 提供强大的机器人框架
- [EmojiPackage](https://github.com/getActivity/EmojiPackage) - 提供表情包资源

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](./LICENSE) 文件。
