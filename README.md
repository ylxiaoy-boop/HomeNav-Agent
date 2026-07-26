# 🏠 HomeNav‑Agent

> *面向家庭服务任务的 Agent‑Tool 架构具身导航智能体*

[![GitHub license](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-🚧%20开发中-yellow)]()

---

## 📖 项目简介

**HomeNav‑Agent** 是一套基于 **Agent‑Tool** 架构的家庭服务具身导航智能体系统。它将大语言模型（LLM）作为中央决策核心，通过 **ReAct（思考‑行动‑观察）** 循环动态调度导航、感知和记忆三大工具，使机器人能在开放的家庭环境中自主完成寻找物品、隐含需求响应、复合任务等复杂服务。

与传统 Pipeline 架构相比，HomeNav‑Agent 具备：
- ✅ **动态决策** – 根据环境反馈实时调整策略，而非“一次性执行”
- ✅ **语义交互** – 直接理解自然语言指令，包括隐含需求
- ✅ **可扩展架构** – 新能力以插件式工具加入，即插即用
- ✅ **持续成长** – 长期记忆机制使重复任务效率提升 **40%+**

---

## 🏛️ 系统架构
决策层 (CentralAgent)
│
▼
工具层 (Tool System)
├── NavigationTool (VLFM 语义导航)
├── PerceptionTool (YOLOv8 目标检测)
└── MemoryTool (常识库 + 长期记忆)
│
▼
基础层 (Models & Infrastructure)
├── LLM API / YOLOv8 / VLFM
├── SQLite 记忆数据库
├── AI2‑THOR 仿真环境
└── 配置 / 日志 / 异常处理


**工作流程**：用户指令 → CentralAgent 思考 → 调用工具执行 → 观察结果 → 循环直至任务完成 → 更新记忆 → 返回结果。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **动态决策** | 采用 ReAct 循环，Agent 可在每步自主推理、选择工具并调整计划 |
| **语义导航** | 基于 VLFM 模型，支持自然语言目标（“去厨房”“找红色杯子”）导航 |
| **视觉感知** | 集成 YOLOv8，提供全景检测、目标搜索、环视扫描 |
| **双层记忆** | 预构建家庭常识库 + 长期环境记忆（置信度/验证/衰减机制） |
| **工具化扩展** | 统一 Tool 接口，新增能力只需添加新工具，核心系统无需修改 |
| **工程完备** | 配置管理、结构化日志、异常重试、RESTful API、CLI 交互 |

---

## 📊 实验结果（论文摘要）

在 AI2‑THOR 仿真环境中，HomeNav‑Agent 表现显著优于传统 Pipeline：

| 任务类型 | Pipeline | HomeNav‑Agent |
|----------|----------|---------------|
| 单物体寻找 | 71%      | **82%**       |
| 隐含需求   | 58%      | **72%**       |
| 复合任务   | 42%      | **64%**       |

- **记忆效率**：重复任务平均路径长度缩短 **47%**，耗时减少 **42%**
- **消融实验**：ReAct 循环贡献最大（+11% 成功率），常识与感知协同亦有显著提升

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- （可选）NVIDIA GPU（用于加速 YOLO / VLFM 推理）

---

### 📦 安装

克隆仓库并安装依赖：

```bash
git clone https://github.com/你的用户名/HomeNav-Agent.git
cd HomeNav-Agent
pip install -r requirements.txt
如果遇到网络问题，可使用国内镜像源，如 pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

---

## ⚙️ 配置
复制示例配置文件：

bash
cp config.yaml.example config.yaml
编辑 config.yaml，填入必要信息：

LLM：API 密钥、模型名称（如 gpt-3.5-turbo）

模型路径：YOLOv8 权重文件路径、VLFM 预训练模型路径

环境设置：AI2‑THOR 场景路径、运行模式（调试/发布）

记忆数据库：SQLite 存储路径（默认 data/memory.db）

你也可以通过环境变量覆盖敏感配置（如 API_KEY），生产环境推荐这样做。

---

## ▶️ 运行
命令行交互模式
bash
python main.py --interactive
输入自然语言指令（如 “帮我找一个杯子”），系统会实时显示 Agent 的思考与行动轨迹。

RESTful API 模式
启动服务：

bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
然后通过 POST /api/v1/task 提交任务（详见 API 文档）。

仿真环境可视化（可选）
如果希望观察机器人在 AI2‑THOR 中的第一视角画面，可启用 --render 参数：

---

bash
python main.py --interactive --render
📁 项目结构
text
HomeNav-Agent/
├── agent/                     # CentralAgent 实现
│   ├── central_agent.py       # 主类，ReAct 循环
│   ├── prompt.py              # 系统 Prompt 模板
│   ├── parser.py              # 输出解析器
│   └── task_state.py          # 任务状态管理
├── tools/                     # 工具系统
│   ├── base.py                # BaseTool 抽象类
│   ├── manager.py             # ToolManager
│   ├── navigation.py          # NavigationTool
│   ├── perception.py          # PerceptionTool
│   └── memory.py              # MemoryTool
├── memory/                    # 记忆底层
│   ├── long_term.py           # 长期记忆（SQLite）
│   ├── knowledge_base.py      # 常识库加载与查询
│   └── schema.py              # 数据结构定义
├── models/                    # AI 模型封装
│   ├── llm_client.py          # LLM API 客户端
│   ├── vlfm_model.py          # VLFM 导航模型
│   └── yolo_model.py          # YOLOv8 检测模型
├── config/                    # 配置管理
│   ├── settings.py            # 配置加载器
│   └── config.yaml            # 主配置文件（不提交，由 .gitignore 忽略）
├── utils/                     # 通用工具
│   ├── logger.py              # 结构化日志
│   ├── exceptions.py          # 自定义异常
│   └── formatters.py          # 格式转换
├── interfaces/                # 对外接口
│   ├── cli.py                 # 命令行入口
│   └── api.py                 # FastAPI 服务
├── data/                      # 数据目录
│   ├── knowledge/             # 常识库 JSON 文件
│   └── memory.db              # 长期记忆数据库（自动生成）
├── config.yaml.example        # 配置示例（提交）
├── requirements.txt           # Python 依赖
├── README.md
└── LICENSE
