# 🏠 HomeNav-Agent

<div align="center">

# HomeNav-Agent

### A Vision-Language Driven Embodied Navigation Agent for Household Environments

**基于视觉语言模型与大语言模型的家庭环境具身导航智能体**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Developing-orange)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-success)]()

*An Agent-Tool based embodied AI framework integrating Vision-Language Models, Large Language Models, semantic navigation, visual perception and long-term memory.*

</div>

---

# 📖 Introduction

HomeNav-Agent 是一个面向家庭服务场景的具身智能导航系统，采用 **Agent-Tool** 架构，以 **Large Language Model (LLM)** 作为中央决策核心，结合 **Vision-Language Model (VLM)**、目标检测模型以及长期记忆模块，实现自然语言理解、自主任务规划、语义导航、视觉感知和环境记忆等能力。

系统遵循 **ReAct（Reasoning + Acting）** 推理范式，使机器人能够根据环境反馈动态调整策略，而不是传统 Pipeline 的固定执行流程。

项目目标并非构建单一导航算法，而是探索 **LLM Agent 在家庭服务机器人中的自主决策框架**。

---

# ✨ Highlights

- 🤖 Agent-Tool Architecture
- 🧠 ReAct-based Dynamic Reasoning
- 👁️ Vision-Language Semantic Navigation
- 📷 YOLOv8 Visual Perception
- 🏠 Household Knowledge Base
- 💾 Long-term Environment Memory
- 🔌 Modular Tool Plugin System
- ⚡ Lightweight Deployment

---

# 🏛️ System Architecture

```text
                    User Instruction
                           │
                           ▼
                 ┌────────────────────┐
                 │   Central Agent     │
                 │  (LLM + ReAct)      │
                 └────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
 ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
 │ NavigationTool │ │ PerceptionTool │ │  MemoryTool    │
 │     VLFM       │ │    YOLOv8      │ │ SQLite + KB    │
 └────────────────┘ └────────────────┘ └────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                 Household Environment
```

---

# 🔄 Workflow

```text
User Task
    │
    ▼
Task Understanding
    │
    ▼
LLM Reasoning
    │
    ▼
Tool Selection
    │
    ▼
Navigation / Perception / Memory
    │
    ▼
Environment Feedback
    │
    ▼
ReAct Reasoning Loop
    │
    ▼
Task Completed
    │
    ▼
Memory Update
```

---

# 🚀 Features

| Module | Description |
|---------|-------------|
| **Central Agent** | 基于 ReAct 的任务规划与动态决策 |
| **Navigation** | 基于 VLFM 的自然语言语义导航 |
| **Perception** | YOLOv8 目标检测、场景感知、目标定位 |
| **Memory** | 家庭常识库 + 长期环境记忆 + 置信度管理 |
| **Tool System** | 插件式工具架构，支持快速扩展 |
| **Engineering** | 配置管理、日志、异常恢复、REST API |

---

# 📦 Installation

Clone this repository.

```bash
git clone https://github.com/yourname/HomeNav-Agent.git

cd HomeNav-Agent
```

Install dependencies.

```bash
pip install -r requirements.txt
```

If the network connection is unstable, use the Tsinghua mirror.

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

# ⚙️ Configuration

Copy the configuration template.

```bash
cp config.yaml.example config.yaml
```

Edit **config.yaml**.

### LLM

```yaml
llm:
  provider: OpenAI
  model: GPT-4o
  api_key: YOUR_API_KEY
```

---

### Vision Model

```yaml
vision:
  detector: models/yolov8.pt
  navigator: models/vlfm
```

---

### Environment

```yaml
environment:
  simulator: AI2-THOR
  scene: FloorPlan1
  mode: debug
```

> Lightweight mode supports replacing AI2-THOR with local images or custom datasets.

---

### Memory

```yaml
memory:
  database: data/memory.db
```

SQLite is used as the default memory backend.

---

# ▶️ Quick Start

Interactive Mode

```bash
python main.py --interactive
```

Render Mode

```bash
python main.py --interactive --render
```

RESTful API

```bash
uvicorn interfaces.api:app --host 0.0.0.0 --port 8000
```

API Example

```http
POST /api/v1/task
```

---

# 📁 Project Structure

```text
HomeNav-Agent
│
├── agent
│   ├── central_agent.py
│   ├── parser.py
│   ├── prompt.py
│   └── task_state.py
│
├── tools
│   ├── base.py
│   ├── manager.py
│   ├── navigation.py
│   ├── perception.py
│   └── memory.py
│
├── memory
│   ├── long_term.py
│   ├── knowledge_base.py
│   └── schema.py
│
├── models
│   ├── llm_client.py
│   ├── yolo_model.py
│   └── vlfm_model.py
│
├── interfaces
│   ├── cli.py
│   └── api.py
│
├── config
│   ├── settings.py
│   └── config.yaml
│
├── utils
│   ├── logger.py
│   ├── exceptions.py
│   └── formatter.py
│
├── assets
│   ├── architecture.png
│   ├── workflow.png
│   └── demo.gif
│
├── data
│   ├── knowledge
│   └── memory.db
│
├── docs
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🛣️ Roadmap

| Version | Progress |
|----------|----------|
| ✅ v1.0 | Agent Framework |
| ✅ v1.1 | Tool System |
| ✅ v1.2 | Memory Module |
| 🚧 v1.3 | Vision-Language Navigation |
| 🚧 v1.4 | Gradio Demo |
| 🚧 v2.0 | Real Robot Deployment |

---

# 📚 Technology Stack

- Python
- PyTorch
- OpenCV
- YOLOv8
- Vision-Language Model (VLFM)
- OpenAI API / Qwen / InternVL
- SQLite
- FastAPI
- Gradio
- AI2-THOR (Optional)

---

# 📌 Future Work

- Multi-Agent Collaboration
- Long-Horizon Task Planning
- Retrieval-Augmented Memory
- Real Robot Deployment
- Multimodal World Model
- Visual Chain-of-Thought Reasoning

---

# 📖 Citation

```bibtex
@misc{HomeNavAgent2026,
  title={HomeNav-Agent: A Vision-Language Driven Embodied Navigation Agent for Household Environments},
  author={Your Name},
  year={2026},
  note={Under Development}
}
```

---

# 🙏 Acknowledgements

This project is inspired by and built upon the following open-source projects:

- YOLOv8
- LangChain
- OpenAI API
- FastAPI
- Gradio
- AI2-THOR
- InternVL
- Qwen2.5-VL

---

<div align="center">

⭐ If you find this project useful, please consider giving it a Star.

</div>
