# 稀疏反馈下的自动化实验控制 — Baseline 方案详解

> 面向金融场景图学习竞赛的自主科研 Agent 系统

---

## 一、Agent 简介

本 Baseline 实现了一个**四阶段自主科研 Agent 系统**，能够在有限预算、稀疏反馈、禁止并行约束下，自动完成"读取历史实验结果 → 分析反馈信号 → 决定下一轮实验配置 → 持续迭代优化 → 输出最终预测结果"的完整闭环。

Agent 的核心设计理念是将人类科研人员的实验优化过程抽象为可自动化的工作流：

- **文献解析（Literature）**：读取赛题说明、探查数据结构、分析现有代码
- **瓶颈诊断（Diagnosis）**：基于历史实验记录，识别当前方案的性能瓶颈
- **代码设计（Design）**：调用 LLM 生成代码修改方案，验证语法并执行 Smoke Test
- **实验验证（Experiment）**：运行训练脚本，读取验证指标，做出 CONTINUE/PIVOT/STOP 决策

系统支持两个子任务：
- **Task 1（产品分类）**：基于 GNN（GCN/SAGE/GAT）的图节点分类
- **Task 2（产品推荐）**：基于序列模型（GRU4Rec/SASRec）的 Top-K 推荐

---

## 二、分析赛题、对问题进行建模

### 2.1 赛题核心分析

#### 赛事背景

在真实的金融模型研发过程中，实验优化往往不是一次性完成的。研究者面临以下约束：

- **有限预算**：只能运行有限次数的实验（如 10~20 轮）
- **稀疏反馈**：只能看到验证集指标、训练耗时、资源占用等间接信号
- **禁止并行**：每次只能运行一个实验，必须串行决策
- **目标未知**：不知道全局最优配置，只能通过局部反馈逐步逼近

本赛题将上述挑战具象为两个图学习任务，要求参赛者构建一个能在多轮交互中持续优化实验决策的 Agent。

#### 问题本质

从控制论视角，本赛题是一个**部分可观察马尔可夫决策过程（POMDP）**：

- **状态空间**：当前实验配置（模型结构、超参数、训练策略）
- **观察空间**：稀疏反馈（val_acc、val_ndcg、训练耗时、错误日志）
- **动作空间**：下一轮实验的调整方向（增大 hidden_dim、降低 lr、更换模型等）
- **奖励函数**：最终测试集上的 Accuracy / NDCG@10
- **约束条件**：预算上限、时间限制、禁止并行

Agent 的核心能力在于：**如何在信息不完整的条件下，通过序列决策逐步逼近最优配置**。

#### 技术建模框架

```
┌─────────────────────────────────────────────────────────────┐
│                    自主科研 Agent 系统                        │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│ Literature  │  Diagnosis  │   Design    │   Experiment    │
│  文献解析    │  瓶颈诊断    │  代码设计    │   实验验证       │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ 读取README   │ 分析历史记录 │ LLM生成代码  │ 运行train.py    │
│ 探查数据分布 │ 识别瓶颈     │ 语法验证     │ 读取metrics     │
│ 分析现有代码 │ 提出假设     │ Smoke Test  │ 决策CONTINUE/   │
│             │             │ 保存修改     │      PIVOT/STOP │
└─────────────┴─────────────┴─────────────┴─────────────────┘
                              ↑_________________________________|
                                    迭代闭环（预算内循环）
```

#### 提交要求

| 文件 | 格式要求 | 说明 |
|------|---------|------|
| `A1.csv` | `test_idx,label` | 产品分类预测结果，label 为 0~N 的整数 |
| `A2.csv` | `uid,prediction` | 产品推荐预测结果，prediction 为逗号分隔的 10 个 iid |
| `prediction.zip` | ZIP 压缩包 | 包含 A1.csv 和 A2.csv |
| `trajectory_B1.json` | JSON | B 榜过程日志（实验轮次、配置、反馈、策略） |
| `trajectory_B2.json` | JSON | B 榜过程日志（实验轮次、配置、反馈、策略） |

#### 评分维度

- **Task 1**：Accuracy（测试节点上的平均预测准确率）
- **Task 2**：NDCG@10（检查目标 item 是否在预测列表中及排位情况）
- **最终得分**：`Score_final = 0.5 × Score_cls + 0.5 × Score_ndcg`
- **复审维度**：Agent 的实验控制能力、反馈利用能力、多轮优化能力（通过 trajectory 日志评估）

---

## 三、任务分层拆解

### ① 数据获取层

数据获取层负责读取竞赛数据并进行初步探查：

- **Task 1 数据**：`.npz` 文件，包含 CSR 格式的邻接矩阵、特征矩阵、标签、训练/测试索引
- **Task 2 数据**：`train.csv` / `test.csv` / `user.csv` / `item.csv`，包含用户交互序列和匿名特征
- **数据探查工具**：`ToolRegistry._inspect_data()` 自动分析数据 shape、分布、缺失值等
- **LLM 辅助分析**：将数据洞察（节点数、边数、类别分布、序列长度等）输入 LLM，生成初步建模建议

### ② 分析计算层

分析计算层是 Agent 的"大脑"，负责从稀疏反馈中提炼规律：

- **实验记忆系统（ResearchMemory）**：持久化存储所有实验记录（配置、指标、耗时、决策）
- **瓶颈诊断（DiagnosisPhase）**：基于历史实验记录，调用 LLM 分析当前瓶颈（如"学习率过高导致震荡"、"模型容量不足导致欠拟合"）
- **策略生成**：LLM 根据诊断结果输出下一轮的优化方向（如"增大 hidden_dim 到 256"、"降低 lr 到 0.0005"）
- **指标解析**：从 `metrics.json` 中提取关键指标（val_acc、val_ndcg、train_loss 等），过滤非数值类型

### ③ 内容生成层

内容生成层负责将 LLM 的策略建议转化为可执行的代码修改：

- **代码生成器（CodeGenerator）**：根据策略生成完整的代码修改方案
- **代码验证**：使用 `ast.parse()` 进行语法验证，确保生成的代码可编译
- **Smoke Test**：导入模型并执行前向传播，验证模型可运行
- **配置更新提取**：从 LLM 的文本输出中解析超参数变更（如 `epochs=100`、`hidden_dim=256`），保存为 `config_update.json`

### ④ 质量控制层

质量控制层确保整个闭环的可靠性和稳定性：

- **参数名映射**：`_get_task_config_params()` 将 TaskConfig 属性名映射到 `train.py` 实际接受的参数名（如 `max_seq_len` → `max_len`、`early_stop` → `patience`）
- **任务目录隔离**：每个任务使用独立的输出目录（`./output/task1/`、`./output/task2/`），避免模型文件互相覆盖
- **训练后自动推理**：ExperimentPhase 在训练成功后自动调用 `infer.py` 生成提交文件
- **早停机制**：`EarlyStopping` 监控验证指标，连续多轮无提升时自动停止训练
- **梯度裁剪**：防止梯度爆炸，提升训练稳定性

---

## 四、开发建议

### 如何获得实验反馈数据？

Agent 的决策质量取决于反馈数据的丰富程度。建议从以下维度收集反馈：

1. **验证指标**：`val_acc`（分类）、`val_ndcg` / `val_mrr`（推荐）——核心优化目标
2. **训练指标**：`train_loss`、`train_acc`——判断过拟合/欠拟合
3. **资源反馈**：训练耗时、内存占用——在预算约束下权衡精度与效率
4. **错误日志**：训练失败时的异常信息——用于诊断和修复
5. **模型复杂度**：参数量、层数、维度——评估模型容量是否匹配数据规模

### 解题思考过程

1. **初始探索阶段（前 2~3 轮）**：快速尝试不同的模型架构（GCN vs SAGE vs GRU4Rec），确定 baseline 性能
2. **定向优化阶段（中间轮次）**：根据诊断反馈，聚焦优化关键超参数（lr、hidden_dim、dropout、epochs）
3. **收敛判断阶段（后段）**：当连续多轮无显著提升时，及时 STOP 并将预算留给其他任务
4. **跨任务迁移**：从 Task 1 的优化经验（如哪种归一化方式更有效）迁移到 Task 2

---

## 五、Baseline 方案详解

### 5.1 Baseline 概况

本 Baseline 是一个端到端的自动化实验 Agent，具备以下特性：

| 特性 | 说明 |
|------|------|
| **四阶段闭环** | Literature → Diagnosis → Design → Experiment |
| **LLM 驱动** | 基于 Qwen/GPT 等 OpenAI 兼容 API 进行策略决策 |
| **代码自进化** | Agent 可自主修改模型代码和超参数配置 |
| **记忆持久化** | 实验记录保存为 JSON，支持断点续跑 |
| **双任务支持** | 同时支持图分类（Task 1）和序列推荐（Task 2） |
| **预算管理** | 支持每任务预算控制、早停、时间限制 |

### 5.2 Baseline 核心代码及解释

#### 5.2.1 代码执行器（训练与推理）

**`code/train.py`** —— 统一训练入口，支持 Task 1 和 Task 2：

```python
def train_task1(args):
    """Task 1: GNN节点分类训练"""
    data = GraphDataset.load(args.data_path)
    # 划分训练/验证集
    train_idx, val_idx = split_train_val(data['train_idx'], args.val_ratio, args.seed)
    # 数据预处理
    features = torch.FloatTensor(data['features']).to(device)
    adj = normalize_adj(data['adj']).to(device)
    labels = torch.LongTensor(data['labels']).to(device)
    # 创建模型
    model = GNNClassifier(
        in_dim=num_features, hidden_dim=args.hidden_dim,
        num_classes=num_classes, num_layers=args.num_layers,
        dropout=args.dropout, model_type=args.model_type
    ).to(device)
    # 训练循环（全图训练 + 验证）
    for epoch in range(1, args.epochs + 1):
        logits = model(features, adj)
        loss = criterion(logits[train_idx_t], labels[train_idx_t])
        loss.backward()
        optimizer.step()
        # 验证 & 早停
        val_acc = compute_accuracy(logits[val_idx_t], labels[val_idx_t])
        if val_acc > best_val_acc:
            torch.save({'model_state_dict': model.state_dict(), ...}, best_path)
    tracker.save(os.path.join(args.output_dir, 'metrics.json'))
```

**`code/infer.py`** —— 统一推理入口，生成提交格式的 CSV：

```python
def infer_task1(args):
    """Task 1 推理"""
    data = GraphDataset.load(args.data_path)
    model, _ = load_model_from_checkpoint(args.checkpoint, device)
    with torch.no_grad():
        logits = model(features, adj)
        predictions = torch.argmax(logits[test_idx_t], dim=1).cpu().numpy()
    pd.DataFrame({'test_idx': test_idx, 'label': predictions}).to_csv(args.output_path, index=False)

def infer_task2_v2(args):
    """Task 2 推理（Top-10 推荐）"""
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    idx2iid = checkpoint.get('idx2iid', {})  # 整数索引 → 原始字符串 iid
    # 逐用户推理
    for user_id in sorted(user_recommendations.keys()):
        recs = user_recommendations[user_id]
        rec_strs = [idx2iid.get(r, str(r)) for r in recs]
        result_rows.append({'uid': user_id, 'prediction': ','.join(rec_strs)})
```

#### 5.2.2 代码提取器（数据探查与模型验证）

**`agent/tools.py`** —— 工具注册表，封装文件读写、代码验证、训练脚本调用等：

```python
class ToolRegistry:
    def _run_training(self, task_type, data_path, code_dir, output_dir, iteration, config):
        # 根据任务类型设置默认模型
        if task_type == 'task2' and 'model_type' not in config:
            config['model_type'] = 'gru4rec'
        cmd = ["python", os.path.join(code_dir, "train.py"),
               "--task", task_type, "--data_path", data_path, "--output_dir", output_dir]
        for k, v in config.items():
            cmd.extend([f"--{k}", str(v)])
        return self._shell_exec(cmd, timeout=1800)

    def _smoke_test_model(self, code_path, task_type):
        """导入模型并执行前向传播验证"""
        spec = importlib.util.spec_from_file_location("test_models", code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if task_type == "classification":
            model = module.GNNClassifier(16, 32, 5, num_layers=2)
            out = model(torch.randn(10, 16), torch.eye(10))
        # ...
```

#### 5.2.3 Agent 部分涉及到的提示词

**`agent/phases.py`** 中各阶段的核心 Prompt 设计：

**LiteraturePhase（文献解析）Prompt 模板：**

```
系统提示：你是一位图机器学习领域的专家研究员。

用户提示：
请基于以下赛题说明和数据分析报告，总结关键挑战和建议方法：

[赛题说明]
{readme_content}

[数据洞察]
{data_insights}

[代码分析]
{code_analysis}

请输出：
1. 关键挑战列表（3~5条）
2. 建议方法列表（3~5条，包含预期效果）
3. 数据洞察摘要
```

**DiagnosisPhase（瓶颈诊断）Prompt 模板：**

```
系统提示：你是一位机器学习实验优化专家。

用户提示：
以下是当前任务的实验历史记录：

{experiment_history}

当前最佳指标：{best_metrics}

请诊断：
1. 当前方案的主要瓶颈是什么？
2. 提出 3 个可验证的优化假设（含预期改进幅度和验证方法）
3. 每个假设的优先级排序
```

**DesignPhase（代码设计）Prompt 模板：**

```
系统提示：你是一位 PyTorch 深度学习工程师。

用户提示：
基于以下诊断结论，请生成具体的代码修改方案：

{diagnosis_report}

当前代码摘要：
{code_summary}

请输出：
1. 需要修改的文件列表
2. 每个文件的具体修改内容（用代码块包裹）
3. 修改后的预期效果
```

#### 5.2.4 实现数据分析功能

**`agent/phases.py` — LiteraturePhase._inspect_data()**：

```python
def _inspect_data(self, data_path, data_type):
    """根据任务类型探查 npz 或 csv 数据"""
    if data_type == 'npz':
        data = np.load(data_path)
        info = {
            "num_nodes": int(data['adj_shape'][0]),
            "num_features": int(data['attr_shape'][1]),
            "num_classes": int(data['labels'][data['labels'] >= 0].max()) + 1,
            "train_size": len(data['train_idx']),
            "test_size": len(data['test_idx'])
        }
    else:  # csv
        train_df = pd.read_csv(f'{data_path}/train.csv')
        test_df = pd.read_csv(f'{data_path}/test.csv')
        info = {
            "train_users": len(train_df),
            "test_users": len(test_df),
            "num_items": len(pd.read_csv(f'{data_path}/item.csv'))
        }
    return info
```

#### 5.2.5 实验分析报告生成以及资料收集部分

**`agent/memory.py`** —— 研究记忆系统：

```python
@dataclass
class ExperimentRecord:
    iteration: int = 0
    phase: str = ""           # experiment / diagnosis / design
    task_id: int = 1
    config: Dict[str, Any] = field(default_factory=dict)   # 实验配置
    metrics: Dict[str, float] = field(default_factory=dict) # 验证指标
    decision: str = "CONTINUE"  # CONTINUE / PIVOT / STOP
    reason: str = ""            # 决策理由
    duration: float = 0.0       # 实验耗时
    timestamp: str = ""         # 时间戳

class ResearchMemory:
    def add_iteration(self, record: ExperimentRecord):
        """添加实验记录并自动更新最佳结果"""
        self.iterations.append(record)
        if self._is_better(record, self.best_result):
            self.best_result = record

    def _is_better(self, record, current_best):
        """比较实验记录，判断是否为更优结果"""
        # Task 1: 优先比较 val_acc
        # Task 2: 优先比较 val_ndcg / mrr
        numeric_values = [v for v in record.metrics.values()
                          if isinstance(v, (int, float))]
        return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
```

**轨迹文件生成（用于 B 榜提交）**：

```python
# orchestrator.py 中 _save_trajectory 方法
def _save_trajectory(self, task_id, submission_dir):
    """生成符合 B 榜要求的 trajectory.json"""
    trajectory = {
        "task_id": task_id,
        "iterations": [],
        "best_iteration": {},
        "total_duration": 0.0
    }
    for record in self.memory.get_iterations(task_id):
        trajectory["iterations"].append({
            "round": record.iteration,
            "config": record.config,
            "metrics": record.metrics,
            "feedback": record.reason,
            "strategy": record.decision,
            "duration": record.duration
        })
    # 保存为 trajectory_task{N}.json
```

#### 5.2.6 集成：分析报告生成、资料收集、实验控制

**`agent/orchestrator.py`** —— 主编排器，集成所有组件：

```python
class ResearchOrchestrator:
    def run_task(self, task_id: int) -> Dict[str, Any]:
        # 1. 设置任务专属输出目录（隔离多任务）
        task_output_dir = os.path.join(self.output_dir, f"task{task_id}")
        task_config.output_dir = task_output_dir

        # 2. Phase 1: Literature（仅执行一次）
        lit_phase = LiteraturePhase(self.llm, self.memory, self.tools, task_config)
        lit_result = lit_phase.run()

        # 3. 迭代闭环: Diagnosis -> Design -> Experiment
        while decision in (CONTINUE, PIVOT):
            diag_phase = DiagnosisPhase(...)
            diagnosis = diag_phase.run()

            design_phase = DesignPhase(...)
            design = design_phase.run()

            exp_phase = ExperimentPhase(...)
            experiment = exp_phase.run(iteration)

            decision = experiment["decision"]

        # 4. Finalize: 打包最佳结果
        return self._finalize(task_id)
```

### 5.3 Baseline 运行说明

#### 环境准备

```bash
# 安装依赖
pip install torch numpy scipy pandas pyyaml requests

# 配置 LLM API（可选，支持 Qwen/GPT 等 OpenAI 兼容接口）
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

#### 运行方式

```bash
# 完整运行（Task 1 + Task 2，每任务 2 轮预算）
cd framework
python main.py --task 1 --task 2 --budget 2 --output_dir ./output --device cpu

# 仅运行 Task 1
python main.py --task 1 --budget 5 --device cuda

# 从配置文件运行
python main.py --config config.yaml

# 恢复之前运行
python main.py --resume ./output/task1/research_memory.json
```

#### 输出结构

```
output/
├── task1/
│   ├── best_model.pt          # 最佳模型检查点
│   ├── final_model.pt         # 最终模型
│   ├── metrics.json           # 训练指标历史
│   ├── A1.csv                 # 分类预测结果
│   ├── literature_summary.json
│   ├── diagnosis_report.json
│   ├── design_notes.json
│   └── experiment_N_report.json
├── task2/
│   ├── best_model.pt
│   ├── metrics.json
│   ├── A2.csv                 # 推荐预测结果
│   └── ...（同上）
└── submission/
    ├── A1.csv                 # 最终提交文件
    ├── A2.csv
    ├── trajectory.json        # 完整实验轨迹
    ├── trajectory_task1.json
    └── trajectory_task2.json
```

---

## 六、Baseline 方案思路

### 6.1 核心设计思路

1. **人机协作式自动化**：LLM 负责高层次的策略决策（诊断瓶颈、提出假设），代码负责低层次的高效执行（训练、推理、指标计算）
2. **渐进式优化**：每轮实验只做一个关键改动，便于归因分析（A/B 测试思想）
3. **记忆驱动决策**：完整的实验记忆让 Agent 避免重复犯错，并从历史中提取可迁移经验
4. **防御式工程**：语法验证、Smoke Test、梯度裁剪、早停等机制确保系统稳定运行

### 6.2 模型选择策略

- **Task 1（图分类）**：默认使用 GraphSAGE
  - GCN：简单高效，但对稀疏图效果一般
  - SAGE：均值聚合，对稀疏图更鲁棒，适合金融场景
  - GAT：注意力机制，表达能力更强但计算开销大
- **Task 2（序列推荐）**：默认使用 GRU4Rec
  - GRU4Rec：参数量小、训练快，适合预算有限场景
  - SASRec：Transformer 架构，表达能力更强但需要更多数据

---

## 七、Baseline 核心逻辑

### 7.1 四阶段闭环详解

```
第 1 轮（冷启动）：
  Literature  →  读取数据 + 分析代码 → 生成初始建模建议
      ↓
  Diagnosis   →  无历史记录 → 提出 3 个初始假设
      ↓
  Design      →  LLM 修改 hidden_dim / lr → 保存 config_update.json
      ↓
  Experiment  →  运行 train.py → 读取 val_acc → 决策 CONTINUE
      ↓
  （记录到 Memory，更新 best_result）

第 2~N 轮（迭代优化）：
  Diagnosis   →  读取历史记录 → 识别最佳/最差实验 → 聚焦优化方向
      ↓
  Design      →  根据诊断结果调整模型/参数
      ↓
  Experiment  →  训练 + 推理 → 生成 A{N}.csv → 决策 CONTINUE/PIVOT/STOP
```

### 7.2 关键决策逻辑

**CONTINUE**：当前指标有提升或仍有优化空间，继续同方向迭代  
**PIVOT**：当前方向遇到瓶颈，更换优化方向（如从调 lr 转向换模型架构）  
**STOP**：指标已收敛或预算耗尽，停止迭代并打包结果

### 7.3 参数映射与传递

```
TaskConfig（YAML/代码中定义）
    ├── epochs=100
    ├── max_seq_len=50      ──→ 映射为 ──→  --max_len 50
    ├── early_stop=20       ──→ 映射为 ──→  --patience 20
    └── lr=0.01

        ↓ ExperimentPhase._get_task_config_params()

命令行参数列表
    ["--epochs", "100", "--max_len", "50", "--patience", "20", "--lr", "0.01"]

        ↓ subprocess.run(train.py, ...)

train.py 解析参数 → 训练模型 → 保存 best_model.pt + metrics.json
        ↓ ExperimentPhase 自动调用 infer.py
生成 A{N}.csv 提交文件
```

---

## 八、思考题

### 8.1 如何优化输入数据的质量？

1. **特征工程**：对稀疏特征矩阵进行 PCA 降维、特征选择，或对匿名用户/物品特征进行 Embedding 编码
2. **数据增强**：对图数据进行随机边丢弃（DropEdge）、特征掩码（Feature Masking）等增强操作
3. **序列预处理**：对推荐任务的交互序列进行去重、时间衰减加权、滑动窗口截断等
4. **归一化策略**：比较对称归一化（symmetric）和随机游走归一化（random_walk）的效果

### 8.2 如何优化处理流程？

1. **并行探索 vs 串行优化**：虽然赛题禁止并行训练，但可以在 Diagnosis 阶段同时生成多个假设，按优先级串行验证
2. **贝叶斯优化替代 LLM**：用 Optuna/Hyperopt 等工具替代 LLM 进行超参数搜索，效率更高
3. **元学习（Meta-Learning）**：从 Task 1 的优化轨迹中提取先验知识（如"SAGE 优于 GCN"），加速 Task 2 的初始探索
4. **模型集成**：保存多轮实验的模型，推理时进行集成投票或加权平均，提升鲁棒性
5. **自适应预算分配**：根据当前任务的提升潜力动态调整预算分配（如 Task 1 已收敛则将预算转移给 Task 2）
6. **失败快速恢复**：训练失败时自动回滚到上一个稳定配置，而非从头开始

---

## 九、附录：补充知识

### A. CSR 稀疏矩阵格式

竞赛数据中的邻接矩阵和特征矩阵均采用 CSR（Compressed Sparse Row）格式存储，优势包括：
- 节省内存：只存储非零元素
- 高效矩阵乘法：`adj @ x` 的时间复杂度与边数成正比

```python
import scipy.sparse as sp
# 从 .npz 还原 CSR 矩阵
adj = sp.csr_matrix(
    (data['adj_data'], data['adj_indices'], data['adj_indptr']),
    shape=tuple(data['adj_shape'])
)
```

### B. GraphSAGE 原理

GraphSAGE 通过**采样 + 聚合**邻居特征生成节点表示：

```
h_v^(k) = σ(W^k · CONCAT(h_v^(k-1), AGG({h_u^(k-1), ∀u ∈ N(v)})))
```

本 Baseline 采用**均值聚合**（mean aggregator），对稀疏图更鲁棒。

### C. GRU4Rec 原理

GRU4Rec 将用户交互序列建模为序列预测问题：

```
1. 将 item ID 映射为 Embedding
2. 通过多层 GRU 建模序列时序依赖
3. 输出层投影到 item 空间，计算每个候选 item 的得分
4. 取 Top-K 作为推荐结果
```

### D. NDCG@10 计算

```
DCG@10 = Σ_{i=1}^{10} (2^{rel_i} - 1) / log2(i + 1)
NDCG@10 = DCG@10 / IDCG@10
```

其中 `rel_i = 1` 表示第 i 位是目标 item，否则为 0。IDCG 是理想排序下的 DCG。

---

## 十、参考资料

1. **竞赛官方说明**：`竞赛说明.txt`
2. **Hamilton, W. L., et al.** "Inductive representation learning on large graphs." *NeurIPS 2017.*（GraphSAGE）
3. **Kipf, T. N., & Welling, M.** "Semi-supervised classification with graph convolutional networks." *ICLR 2017.*（GCN）
4. **Hidasi, B., et al.** "Session-based recommendations with recurrent neural networks." *ICLR 2016.*（GRU4Rec）
5. **Kang, W. C., & McAuley, J.** "Self-attentive sequential recommendation." *ICDM 2018.*（SASRec）
6. **PyTorch 官方文档**：https://pytorch.org/docs/
7. **SciPy 稀疏矩阵文档**：https://docs.scipy.org/doc/scipy/reference/sparse.html

---

> 本文档对应 Baseline 代码版本：`framework/` 目录下的完整 Agent 系统。
> 如需运行，请确保已安装 PyTorch、NumPy、SciPy、Pandas、PyYAML。
