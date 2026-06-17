# 金融场景序列推荐 A 榜数据集

## 1. 数据集目的

本目录提供一份面向“稀疏反馈下的自动化实验挑战”命题方向的金融场景序列推荐数据。该版本基于 5 万用户样本生成，定位为 A 榜数据，用于验证数据格式、字段规范、稀疏反馈设置、提交格式和评测流程。

基本任务形式为：

```text
给定用户近期 item 交互序列、匿名用户侧特征和匿名 item 侧特征，
为每个测试用户预测一个按置信度排序的 item 列表。
```

默认候选 item 集合为 `item.csv` 中全部 `iid`。

## 2. 文件列表

```text
./
  train.csv
  test.csv
  sample_submission.csv
  user.csv
  item.csv
  metadata.json
  README.md
```

## 3. 文件结构

### 3.1 train.csv

训练集，包含目标 item。

```text
uid,target_iid,item_seq_raw,item_seq_dedup,item_seq_counts
```

### 3.2 test.csv

测试集，隐藏目标 item。

```text
uid,item_seq_raw,item_seq_dedup,item_seq_counts
```

### 3.3 sample_submission.csv

```text
uid,prediction
```

`prediction` 为按置信度排序的 item id 列表，英文逗号分隔。

### 3.4 user.csv

```text
uid,u_cat_01,u_cat_02,u_cat_03,u_cat_04,u_cat_05,u_cat_06,u_cat_07,u_cat_08
```

### 3.5 item.csv

```text
iid,i_cat_01,i_cat_02,i_cat_03,i_bucket_01
```

## 4. 数据规模

```text
train.csv: 40000 行
test.csv: 10000 行
user.csv: 50000 行
item.csv: 2156 行
```

## 5. 匿名化说明

本数据集采用匿名 id、匿名特征名和匿名特征值，不包含内部映射表。

## 6. 提交说明

提交csv文件（命名为A2.csv）需包含两列：

```csv
uid,prediction
u000001,"i000001,i000002,i000003,i000004,i000005"
```

其中 `prediction` 为按置信度从高到低排序的 item id 列表，使用英文逗号分隔。默认候选 item 集合为 `item.csv` 中全部 `iid`。
