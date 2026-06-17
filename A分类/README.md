# 数据文件说明

该文件夹包含节点分类任务A榜数据集 `.npz` 文件，另外包含对应测试集提交模板 CSV 文件。所有节点编号均为从 `0` 开始的整数编号。

## 数据集

| 原始文件   | 节点数 | 特征维度 | 类别数 | 训练节点数 | 测试节点数 |
| ---------- | -----: | -------: | -----: | ---------: | ---------: |
| `A1.npz` | 13,752 |      767 |     10 |     11,001 |      2,751 |

## `.npz` 变量含义

| 变量名           | 含义                                                                                                                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adj_data`     | 图邻接矩阵的 CSR `data` 数组，表示非零边的权重。当前文件中边权主要为 `1.0`。                                                                                                             |
| `adj_indices`  | 图邻接矩阵的 CSR `indices` 数组，表示每个非零边所在的列编号，也就是目标节点编号。                                                                                                          |
| `adj_indptr`   | 图邻接矩阵的 CSR `indptr` 数组，长度为 `节点数 + 1`。节点 `i` 的邻接非零项位于 `adj_data[adj_indptr[i]:adj_indptr[i+1]]` 和 `adj_indices[adj_indptr[i]:adj_indptr[i+1]]`。         |
| `adj_shape`    | 图邻接矩阵形状，格式为 `(节点数, 节点数)`。请按文件中保存的邻接关系原样使用，不额外假设一定无向、对称或无自环。                                                                            |
| `attr_data`    | 节点特征矩阵的 CSR `data` 数组，表示非零特征值。                                                                                                                                           |
| `attr_indices` | 节点特征矩阵的 CSR `indices` 数组，表示每个非零特征所在的特征列编号。                                                                                                                      |
| `attr_indptr`  | 节点特征矩阵的 CSR `indptr` 数组，长度为 `节点数 + 1`。节点 `i` 的特征非零项位于 `attr_data[attr_indptr[i]:attr_indptr[i+1]]` 和 `attr_indices[attr_indptr[i]:attr_indptr[i+1]]`。 |
| `attr_shape`   | 节点特征矩阵形状，格式为 `(节点数, 特征维度)`。                                                                                                                                            |
| `labels`       | 节点标签数组，长度为 `节点数`。`train_idx` 对应位置为公开标签，取值范围是 `0` 到 `类别数 - 1`；`test_idx` 对应位置在公开文件中统一置为 `-1`，表示测试标签隐藏。                  |
| `train_idx`    | 可用于训练/验证的节点编号数组。训练和验证节点已合并提供。                                                                                                                                    |
| `test_idx`     | 测试节点编号数组。选手需要对这些节点预测标签，并按提交模板中的 `test_idx` 提交对应 `label`。                                                                                             |

可以使用如下方式还原邻接矩阵和特征矩阵：

```python
import numpy as np
from scipy.sparse import csr_matrix

data = np.load("A1.npz")

adj = csr_matrix(
    (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
    shape=tuple(data["adj_shape"]),
)

features = csr_matrix(
    (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
    shape=tuple(data["attr_shape"]),
)

labels = data["labels"]
train_idx = data["train_idx"]
test_idx = data["test_idx"]
```

## 提交文件格式

数据集对应一个 CSV 提交模板（ `sample_submission.csv`）。CSV 包含表头，并且只有两列，形如：

```csv
test_idx,label
18,-1
19,-1
20,-1
```

其中：

| 列名         | 含义                                                                               |
| ------------ | ---------------------------------------------------------------------------------- |
| `test_idx` | 测试节点编号，必须与模板保持一致。                                                 |
| `label`    | 选手对该测试节点预测的类别编号。模板中的 `-1` 是占位符，提交时应替换为预测标签。 |

提交要求：

1. 保留 CSV 表头 `test_idx,label`。
2. 不要删除、增加或重排 `test_idx` 行。
3. `label` 必须是整数类别编号，合法类别为 `0` 到 `9`。
4. 公开 `.npz` 中 `test_idx` 对应的 `labels` 均为 `-1`，不是真实测试标签。
