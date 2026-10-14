# my_experiment.py

# 1. 导入库
import pandas as pd
import jieba
import jieba.analyse
import re
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ================= 函数定义区 =================

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled


def load_stopwords():
    """加载停用词，如果没有文件则使用基础停用词列表"""
    try:
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            return set([line.strip() for line in f])
    except FileNotFoundError:
        base_stopwords = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
            '它', '们', '那', '些', '为', '所', '以', '之', '与', '及',
            '但', '而', '或', '被', '把', '对', '从', '向', '让', '将',
            '等', '能', '可', '还', '又', '再', '才', '只', '已', '经',
            '吗', '啊', '呢', '吧', '哦', '哈', '呀', '嗯', '嘛', '哎'
        ]
        return set(base_stopwords)


def clean_text(text):
    """文本清洗：去除HTML、URL、特殊符号"""
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_filter(text, stopwords):
    """分词并过滤停用词和单字词"""
    words = jieba.lcut(text)
    filtered_words = [
        word for word in words
        if word.strip() and len(word) > 1 and word not in stopwords
    ]
    return " ".join(filtered_words)


# ================= 主流程 =================

# 2. 调用函数来获取数据
X_train, y_train, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "=" * 50)

# 新增：从原始X_train,y_train切分验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=RANDOM_SEED
)

# --- 在这里开始你的实验！ ---

# 4. 文本预处理（清洗 + 分词）
print("\n========== 文本预处理 ==========")
stopwords = load_stopwords()
print(f"停用词数量: {len(stopwords)}")

X_train_processed = [tokenize_and_filter(clean_text(text), stopwords) for text in X_train]
X_train_split_processed = [tokenize_and_filter(clean_text(text), stopwords) for text in X_train_split]
X_val_processed = [tokenize_and_filter(clean_text(text), stopwords) for text in X_val]
X_test_processed = [tokenize_and_filter(clean_text(text), stopwords) for text in X_test_unlabeled]

print(f"处理完成。训练集: {len(X_train_split_processed)} 条, 验证集: {len(X_val_processed)} 条")

# 5. TF-IDF特征提取
print("\n========== TF-IDF特征提取 ==========")
feature_options = [10000, 15000, 20000]
best_vec = None
best_feat_score = 0

for max_feat in feature_options:
    vec_temp = TfidfVectorizer(
        max_features=max_feat,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    X_tr_temp = vec_temp.fit_transform(X_train_split_processed)
    lr_temp = LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
    lr_temp.fit(X_tr_temp, y_train_split)
    score = accuracy_score(y_val, lr_temp.predict(vec_temp.transform(X_val_processed)))
    print(f"  max_features={max_feat}: 验证准确率 = {score:.4f}")
    if score > best_feat_score:
        best_feat_score = score
        best_vec = vec_temp

vec = best_vec
X_train_tfidf = vec.fit_transform(X_train_split_processed)
X_val_tfidf = vec.transform(X_val_processed)
X_test_tfidf = vec.transform(X_test_processed)
print(f"最终特征维度: {X_train_tfidf.shape[1]}")

# 6. 模型训练与调参
print("\n" + "=" * 50)
print("模型训练与调参")
print("=" * 50)

results = []

# 6.1 朴素贝叶斯
print("\n========== 朴素贝叶斯 ==========")
for alpha in [0.01, 0.1, 0.5, 1.0]:
    t0 = time.time()
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_tfidf, y_train_split)
    y_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    train_time = time.time() - t0
    results.append({
        'model': model, 'name': f"NB (alpha={alpha})",
        'accuracy': acc, 'f1_macro': f1_macro, 'time': train_time, 'y_pred': y_pred
    })
    print(f"  NB (alpha={alpha}): Acc={acc:.4f}, F1-macro={f1_macro:.4f}, Time={train_time:.1f}s")

# 6.2 逻辑回归
print("\n========== 逻辑回归 ==========")
for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
    t0 = time.time()
    model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_tfidf, y_train_split)
    y_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    train_time = time.time() - t0
    results.append({
        'model': model, 'name': f"LR (C={C})",
        'accuracy': acc, 'f1_macro': f1_macro, 'time': train_time, 'y_pred': y_pred
    })
    print(f"  LR (C={C}): Acc={acc:.4f}, F1-macro={f1_macro:.4f}, Time={train_time:.1f}s")

# 6.3 SVM
print("\n========== SVM ==========")
for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
    t0 = time.time()
    model = SVC(kernel='linear', C=C, random_state=RANDOM_SEED)
    model.fit(X_train_tfidf, y_train_split)
    y_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    train_time = time.time() - t0
    results.append({
        'model': model, 'name': f"SVM (C={C})",
        'accuracy': acc, 'f1_macro': f1_macro, 'time': train_time, 'y_pred': y_pred
    })
    print(f"  SVM (C={C}): Acc={acc:.4f}, F1-macro={f1_macro:.4f}, Time={train_time:.1f}s")

# 6.4 MLP（多层感知机）
print("\n========== MLP ==========")
best_loss_history = None
for hidden_layer in [(50,), (100,), (50, 50)]:
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer,
        max_iter=1,
        random_state=RANDOM_SEED,
        warm_start=True,
        solver='sgd',
        learning_rate_init=0.01,
        verbose=False
    )
    loss_history = []
    for epoch in range(100):
        mlp.partial_fit(X_train_tfidf, y_train_split, classes=np.unique(y_train_split))
        loss_history.append(mlp.loss_)
    train_time = time.time() - t0
    y_pred = mlp.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    results.append({
        'model': mlp, 'name': f"MLP ({hidden_layer})",
        'accuracy': acc, 'f1_macro': f1_macro, 'time': train_time, 'y_pred': y_pred
    })
    print(f"  MLP {hidden_layer}: Acc={acc:.4f}, F1-macro={f1_macro:.4f}, Time={train_time:.1f}s")
    if hidden_layer == (100,):
        best_loss_history = loss_history

# 7. 结果汇总
print("\n" + "=" * 50)
print("结果汇总")
print("=" * 50)

print(f"\n{'模型':<22} {'准确率':<10} {'F1-macro':<12} {'耗时':<8}")
print("-" * 55)
for r in results:
    print(f"{r['name']:<22} {r['accuracy']:<10.4f} {r['f1_macro']:<12.4f} {r['time']:<8.1f}s")

best_result = max(results, key=lambda x: x['f1_macro'])
print(f"\n✅ 最优模型: {best_result['name']}")
print(f"   准确率: {best_result['accuracy']:.4f}")
print(f"   F1-macro: {best_result['f1_macro']:.4f}")

# 8. 混淆矩阵
print("\n生成混淆矩阵...")
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_result["name"]}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ 混淆矩阵已保存为 confusion_matrix.png")

# 9. 损失曲线（MLP）
if best_loss_history is not None:
    print("\n绘制MLP训练损失曲线...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best_loss_history) + 1), best_loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch（训练轮数）', fontsize=12)
    plt.ylabel('Loss（损失值）', fontsize=12)
    plt.title('MLP (hidden=(100,)) 训练损失曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    print("✅ 损失曲线已保存为 loss_curve.png")

# 10. 分类报告
print(f"\n--- 最优模型 {best_result['name']} 的分类报告 ---")
print(classification_report(y_val, best_result['y_pred'], digits=4))

# 11. 全量数据重训（使用原始完整X_train、y_train）
print("\n" + "=" * 50)
print("全量数据重训最优模型")
print("=" * 50)

vec_full = TfidfVectorizer(
    max_features=vec.max_features,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)
X_full_tfidf = vec_full.fit_transform(X_train_processed)
print(f"全量特征维度: {X_full_tfidf.shape[1]}")

best_model = best_result['model'].__class__(**{
    k: v for k, v in best_result['model'].get_params().items()
    if k not in ['classes_', 'n_features_in_', 'feature_names_in_']
})
if isinstance(best_model, MLPClassifier):
    best_model.set_params(warm_start=False, max_iter=300)
best_model.fit(X_full_tfidf, y_train)
print(f"最优模型: {type(best_model).__name__}, 参数: {best_result['name']}")

# 12. 预测测试集
print("\n" + "=" * 50)
print("预测测试集")
print("=" * 50)

X_test_final_tfidf = vec_full.transform(X_test_processed)
predictions = best_model.predict(X_test_final_tfidf)

pred_dist = pd.Series(predictions).value_counts().sort_index()
print(f"\n预测类别分布:")
for cls, count in pred_dist.items():
    print(f"  类别 {cls}: {count} 条 ({count/len(predictions)*100:.1f}%)")

# 13. 保存预测结果
pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)
print(f"\n✅ predictions.csv 已生成")
print(f"   测试集预测完成: {len(predictions)} 条")