# 文本分类实验 - Project 1

## 项目简介
基于 TF-IDF + 多种机器学习模型的中文新闻文本分类（10分类）。

## 环境依赖
- Python 3.10+
- numpy, pandas, scikit-learn, jieba, matplotlib, seaborn

## 文件结构
├── train_data.csv              # 训练集
├── test_data_unlabeled.csv     # 测试集
├── draft_main.py               # 主代码
├── predictions.csv             # 预测结果
├── confusion_matrix.png        # 混淆矩阵图
├── loss_curve.png              # MLP损失曲线图
└── README.md                   # 本文件

## 运行方式
python draft_main.py
### 首次运行需安装依赖：
pip install numpy pandas scikit-learn jieba matplotlib seaborn


## 实验流程
1. 数据加载与探索性分析
2. 文本预处理（清洗、分词、去停用词）
3. 数据增强（因类别均衡未触发）
4. TF-IDF 特征提取（max_features=20000, ngram=(1,2)）
5. 模型训练与调参（NB, LR, SVM, MLP）
6. 结果评估（准确率、F1、混淆矩阵、损失曲线）
7. 全量重训 + 测试集预测

### 各模型家族最优结果
| 模型 | 最优参数 | 准确率 | F1-macro | 耗时 |
|------|----------|--------|----------|------|
| NB   | alpha=0.01 | 0.8976 | 0.8988 | 0.0s |
| LR   | C=5.0 | 0.9274 | 0.9279 | 2.7s |
| SVM  | C=2.0 | 0.9281 | 0.9286 | 65.2s |
| MLP  | (50,50)| 0.9037 | 0.9051 | 79.9s |

**最优模型**: SVM (C=2.0)，验证集准确率 92.81%

## 可复现性
- 固定随机种子: `random_state=42`
- 分层抽样划分: `train_test_split(stratify=y)`