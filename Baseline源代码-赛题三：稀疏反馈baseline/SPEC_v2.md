# SPEC v2 — 自主科研Agent框架规格说明

## 1. 系统定位

端到端自主科研Agent，面向金融场景图学习任务的自动化实验控制竞赛。
系统在零人工干预下完成"文献解析 → 瓶颈诊断 → 代码设计 → 实验验证 → 提交打包"的完整科研闭环。

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchOrchestrator                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Literature│→ │Diagnosis │→ │  Design  │→ │Experiment│   │
│  │  Phase   │  │  Phase   │  │  Phase   │  │  Phase   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         ↑_________________________________________↓         │
│                      (迭代闭环)                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  _finalize(): 收集最佳结果 → 生成提交产物 → 打包 submission  │
└─────────────────────────────────────────────────────────────┘
```

## 3. 核心组件

### 3.1 Agent 系统 (`agent/`)

| 文件 | 职责 |
|------|------|
| `main.py` | CLI入口，解析--task/--config/--resume |
| `config.py` | 配置管理：LLMConfig / ResearchConfig / ModelConfig，支持YAML |
| `orchestrator.py` | 主编排器：四阶段调度、停止条件、_finalize打包 |
| `phases.py` | 四阶段实现：LiteraturePhase / DiagnosisPhase / DesignPhase / ExperimentPhase |
| `memory.py` | ResearchMemory + ExperimentRecord：状态持久化 |
| `llm_client.py` | OpenAI兼容LLM客户端，自动记录合规日志 |
| `tools.py` | 工具注册表：文件读写、Shell执行、代码验证、模型smoke test、数据探查、日志分析 |
| `code_generator.py` | 代码生成器：自主编写/修改PyTorch代码 |

### 3.2 自主生成的代码 (`code/`)

| 文件 | 职责 |
|------|------|
| `models.py` | 模型定义：GNNClassifier + SeqRecommender（Agent可自主修改） |
| `datasets.py` | 数据集定义：GraphDataset + RecDataset |
| `train.py` | 训练入口：支持分类和推荐两种任务的训练 |
| `infer.py` | 推理入口：加载checkpoint → 生成预测 → 输出提交格式 |
| `utils.py` | 工具函数：邻接矩阵处理、序列处理、评估指标 |

## 4. 四阶段科研闭环

### Phase 1: Literature（文献/上下文解析）
- 阅读竞赛说明、README文档
- 探查数据结构（HDF5/npz/csv）
- 审查现有代码
- 输出：task{N}/literature_summary.md

### Phase 2: Diagnosis（瓶颈诊断）
- 分析训练日志
- 识别瓶颈：准确率、NDCG、训练稳定性、泛化能力
- 提出1-3个可验证优化假设
- 输出：task{N}/diagnosis_report.md

### Phase 3: Design（代码设计）
- 根据假设自主编写/修改PyTorch代码
- 语法检查
- 模型smoke test
- 输出：task{N}/design_notes.md

### Phase 4: Experiment（实验验证）
- 调用训练脚本
- LLM分析结果，决策：CONTINUE / PIVOT / STOP
- 输出：task{N}/experiment_M_report.md

## 5. 配置系统

```yaml
# config.yaml 示例
llm:
  model: "qwen-turbo"
  api_key: ""  # 从环境变量LLM_API_KEY读取
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  temperature: 0.7

research:
  budget: 10              # 总实验预算
  time_limit: 3600        # 时间限制（秒）
  early_stop_patience: 3  # 早停耐心值
  phases:
    literature: true
    diagnosis: true
    design: true
    experiment: true

tasks:
  task1:                  # 产品分类
    data_path: ""
    sample_path: ""
    model_type: "sage"
    hidden_dim: 128
    
  task2:                  # 产品推荐
    data_dir: ""
    model_type: "gru4rec"
    embedding_dim: 64
```

## 6. 记忆系统

```json
{
  "iterations": [
    {
      "iteration": 1,
      "phase": "literature",
      "config": {...},
      "metrics": {"acc": 0.85, "ndcg": 0.42},
      "decision": "CONTINUE",
      "timestamp": "2026-06-04T12:00:00"
    }
  ],
  "best_result": {
    "iteration": 3,
    "metrics": {"acc": 0.92, "ndcg": 0.48}
  }
}
```

## 7. 工具注册表

| 工具名 | 功能 |
|--------|------|
| `inspect_data` | 探查数据结构（npz/csv） |
| `summarize_code` | 代码摘要分析 |
| `validate_code` | Python语法检查 |
| `smoke_test_model` | 模型前向传播测试 |
| `run_training` | 执行训练脚本 |
| `analyze_log` | 训练日志分析 |
| `generate_code` | LLM代码生成 |
| `file_read` | 文件读取 |
| `file_write` | 文件写入 |
| `shell_exec` | Shell命令执行 |
