<div style="text-align: center;">
    <img src="https://img.qwwq.top/i/2025/02/22/67b9d79d5a5c9.png" alt="语影 Logo">
</div>

<div style="text-align: center;">
    <img src="https://img.shields.io/github/issues/EnderKC/YuYing-Chameleon.svg" alt="GitHub issues" onclick="window.location.href='https://github.com/EnderKC/YuYing-Chameleon/issues'" style="cursor:pointer;">
    <img src="https://img.shields.io/github/stars/EnderKC/YuYing-Chameleon.svg" alt="GitHub stars" onclick="window.location.href='https://github.com/EnderKC/YuYing-Chameleon/stargazers'" style="cursor:pointer;">
</div>

# 语影 (YuYing-Chameleon)

## 📖 1. 项目简介

语影是一个基于 NoneBot2 的智能 QQ 机器人，通过先进的 AI 技术实现自然、智能的群聊互动。它能够理解上下文、记住用户信息、智能决策回复时机，为群聊带来更加真实的对话体验。

## ✨ 2. 核心功能

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

## 📦 3. 快速开始

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

## ⚙️ 4. 核心配置

在 `configs/config.toml` 中配置以下关键项：

```toml
[yuying_chameleon]
# LLM 配置
openai_api_key = "your-api-key"
openai_model = "gpt-4-turbo"

# 多模态：传给 LLM 的历史图片上限（从当前消息开始向前数，含当前；自动跳过 gif）
llm_history_max_images = 2

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

详细配置说明请参考 [运行逻辑说明文档.md](./docs/运行逻辑说明文档.md)

## 📸 5. 展示

<div style="text-align: center;">
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8f55a62c.jpg" alt="语影 1" width="500"/>
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8fa17300.jpg" alt="语影 2" width="500"/>
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8fc5992d.jpg" alt="语影 3" width="500"/>
    <img src="https://img.qwwq.top/i/2025/02/26/67bef8ffb12d9.jpg" alt="语影 4" width="500"/>
</div>

## 📚 6. 文档

- [运行逻辑说明文档](./docs/运行逻辑说明文档.md) - 详细的架构说明和运行原理

## 🤝 7. 贡献

欢迎任何形式的贡献！请提交 Issue 或 Pull Request。

---

## 8. 控制逻辑&公式

### 8.1 用户消息防抖公式

$$WaitTime = w_1 \cdot Length + w_2 \cdot Length^2 + w_3 \cdot IsEndPunctuation + b$$

- 令 $w_1 > 0$（正系数）：字数少时，时间随字数增加。
- 令 $w_2 < 0$（负系数）：字数多到一定程度，平方项的负值会把等待时间拉下来（形成抛物线开口向下）。
- 参数示例（凭经验构造）：
- $$WaitTime = 0.5 \cdot L - 0.02 \cdot L^2 - 2.0 \cdot P + 1.5$$
> 2个字 (无标点): $0.5(2) - 0.02(4) + 1.5 = 2.42s$ (等待中等)
> 
> 12个字 (无标点): $0.5(12) - 0.02(144) + 1.5 = 6 - 2.88 + 1.5 = 4.62s$ (最长等待)
> 
> 30个字 (无标点): $0.5(30) - 0.02(900) + 1.5 = 15 - 18 + 1.5 = -1.5s$ (立即发送)
> 
> 任何字数 (有标点 P=1): 减去 2.0s，大幅加速。

### 8.2 图片等待加成（自适应防抖）

在上述基础等待时间上，追加图片相关的额外等待（与拼接段数上限无关）：

- 若**防抖窗口首段为纯图片**（`msg_type == "image"`）：额外 `+ adaptive_debounce_first_image_extra_wait_seconds`（默认 10s）
- 若首段不是纯图片：按当前窗口累计的图片数 `N`（`image_ref_map` 的 unique 数量）额外 `+ N * adaptive_debounce_image_extra_wait_seconds`（默认每张 5s）
- 最终等待会被 `adaptive_debounce_max_hold_seconds` 的剩余时间限制，避免超过硬截止

## 9 参考文献
### 9.1 用户消息防抖
[1] [Sacks 的话轮转换理论 ](https://www.jstor.org/stable/412243)
> Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). A simplest systematics for the organization of turn-taking for conversation.
> 
> 这是对话分析的圣经。虽然是讲语音的，但它定义了“话轮构建单元 (TCU)”，指出句子在语法完整点（Transition Relevance Place, TRP）最容易结束。长句子（20字+）往往到达了 TRP，而中等句子往往处于 TCU 中间。

[2] [Koester 的击键动力学](https://ieeexplore.ieee.org/document/331567)
> Koester, H. H., & Levine, S. P. (1994). Modeling the speed of text entry with a word prediction interface.
> 
> 研究表明认知负荷高的时候（比如想句子中间的词），输入速度会变慢，暂停会变长。

[3] [话轮转换的线索](https://www.sciencedirect.com/science/article/abs/pii/S0885230810000690)
> 论文标题: Turn-taking cues in task-oriented dialogue
> 
> 作者: Gravano, A., & Hirschberg, J.
> 
> 发表于: Computer Speech & Language (2011)
> 
> 贡献: 这篇论文详细列出了影响话轮结束的 7 个核心特征，其中包括 Text features (文本特征) 和 Prosodic features (韵律特征)。
>
> 他们证明了 Intonation (语调/标点) 和 Duration (时长/字数) 是预测“对方是否说完”的最重要变量。
>
> [公式](#81-用户消息防抖公式)其实就是这篇论文中 "Feature Selection" 环节的简化版。

[4] [基于决策树或回归的预测](https://www.researchgate.net/publication/221480953_Learning_decision_tree_to_determine_turn-taking_by_spoken_dialogue_systems)
> 论文标题: Learning Decision Trees for Turn-Taking Prediction in Spoken Dialogue Systems
> 
> 作者: Sato, R., et al.
> 
> 发表于: Interspeech (2002)
> 
> 贡献: 他们使用决策树和线性分类器来判断 EOT (End of Turn)。论文中的模型输入就是向量 $X = [Length, Silence, Syntax]$。
> 
> 公式 $w \cdot x + b$ 正是这类 **线性分类器（Linear Classifier）** 的决策边界公式。
### 9.2 机器人记忆行为
[1] [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
> 核心概念：这是“AI 记忆”领域的开山之作（著名的“西部世界”实验）。
>
>解决什么问题？它提出了一个完整的记忆架构，不仅仅是“储存”，而是分为三步：
> 
> Memory Stream（记忆流）：记录所有的原始对话（不做总结，保留原样）。
> 
> Retrieval（检索）：当用户说话时，不把所有记忆都给模型，而是根据相关性（Relevance）、 **新近度（Recency）和重要性（Importance）** 去数据库里捞出最相关的几条记忆。
> 
> Reflection（反思）：定期生成高层次的推论（例如：用户多次提到由于加班很累 -> 生成反思“用户最近工作压力很大”）。

[2] [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
> 核心概念：将 LLM 的上下文窗口视为“内存（RAM）”，将数据库视为“硬盘（Disk）”。
> 
> 解决你的什么问题：它教你怎么在有限的提示词窗口里管理无限的记忆。
> 
> 关键机制：它把记忆分为两类：
>
> Core Memory（核心显存）：一直放在 Prompt 里的关键信息（如：用户叫什么，性格是啥，当前任务）。模型可以自行决定何时更新这里的内容。
> 
> Archival Memory（档案存储）：历史对话的大库。模型需要学会调用“函数”去里面翻找以前的信息。

[3] [RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities](https://arxiv.org/abs/2310.00746)
>核心概念：这是专门研究角色扮演的大型研究。
>
> 解决什么问题：它指出主要问题在于 context-based instruction（基于上下文的指令）容易失效。
> 
> 解决方案：
>
> Context-Aware extraction：不仅仅给设定，还要给大量的Few-Shot（少样本示例）。
>
> 如果条件允许，使用特定风格的对话数据进行微调（SFT）。
> 
> 工程建议：你的 Prompt 里可能只有“人设描述”。必须加入 3-5 对“完美的对话示例”。
>
> Bad: "你是一个高冷的杀手。"
>
> Good: "你是一个高冷的杀手。示例：[用户：你好。机器人：有事快说，我赶时间。]" —— 让模型模仿语气，而不是理解描述。

[4] [Better Zero-Shot Reasoning with Role-Play Prompting](https://arxiv.org/abs/2308.07702)
> 核心概念：通过特定的 Prompt 结构（比如让模型先进入角色的内心独白），可以大幅降低 AI 味。
>
> 工程建议：尝试 CoT (Chain of Thought) for Roleplay。
>
> 让模型在输出回复前，先在 `<thought>` 标签里思考：“我现在是这个角色，面对这句话，我应该用什么情绪？我之前的记忆里有没有相关的事？我应该避免什么样的 AI 常用语？”

## 🙏 10. 致谢

- [1254qwer](https://github.com/1254qwer) - 提供丰富的表情包支持
- [nonebot](https://github.com/nonebot/nonebot) - 提供强大的机器人框架
- [EmojiPackage](https://github.com/getActivity/EmojiPackage) - 提供表情包资源

## 📄 11. 许可证

本项目采用 MIT 许可证，详见 [LICENSE](./LICENSE) 文件。
