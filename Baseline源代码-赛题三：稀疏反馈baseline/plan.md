# Agent长程推理与自动化实验控制 - 竞赛代码构建计划

## 项目概述
构建一个完整的Agent系统，用于在有限预算、稀疏反馈、禁止并行约束下自动化实验控制。
包含两个子任务：
1. **产品分类任务(A1)**: 图节点分类 (GNN-based)
2. **产品推荐任务(A2)**: 序列推荐 (Sequence-based)

## 技能选择
- **Capability Skill**: `vibecoding-general-swarm` (Mode A - multi-agent, 3+ distinct modules)
- **Mode**: Mode A - multi-agent (infrastructure + application logic + multiple components)

## 执行阶段

### Stage 1: 规格设计 (Main Agent)
- 编写 SPEC.md - 系统架构、模块划分、接口定义
- 初始化项目结构和git仓库

### Stage 2: 并行模块开发 (Sub-agents)
- **Agent A**: 数据加载与预处理模块 (Task 1 & Task 2)
- **Agent B**: GNN分类器 + 序列推荐模型
- **Agent C**: Agent核心系统 (LLM API, 记忆系统, 调度策略)
- **Agent D**: 训练/推理pipeline + 主入口

### Stage 3: 集成与测试 (Main Agent)
- 合并所有分支
- 集成测试
- 最终验证

## 模块划分
1. `data_loaders/` - 图数据加载器 + 推荐数据加载器
2. `models/` - GNN分类器 + 序列推荐模型
3. `agent/` - Agent核心、记忆系统、调度器、LLM接口
4. `trainers/` - 训练循环、验证、早停
5. `predictors/` - 推理和结果生成
6. `main.py` - 主入口点
7. `config.py` - 配置管理
