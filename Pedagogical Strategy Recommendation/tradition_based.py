import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB  # 传统方法：朴素贝叶斯
from collections import Counter
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 下载停用词数据包
nltk.download("stopwords")

# 初始化停用词表和词干提取器
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# 1. 加载数据
file_path = "exp_data_prob.csv"
data = pd.read_csv(file_path)

# 2. 对 "cleaned_history" 进行前置处理
def preprocess_with_sep_tokens(text):
    text = text.replace("||| teacher: ", "[SEP] ").replace("||| student: ", "[SEP] ")
    tokens = text.split()
    # 移除停用词并进行词干提取
    tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    text = "[CLS] " + " ".join(tokens) + " [SEP]"  # 加上 [CLS] 和 [SEP]
    return text.strip()

data["cleaned_history"] = data["History"].apply(preprocess_with_sep_tokens)

# 3. 标签编码
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["Peda_Strategies"])

# 4. 划分训练集和测试集
X_train = data[data["Split"] != "test"]["cleaned_history"]
y_train = data[data["Split"] != "test"]["label"]
X_test = data[data["Split"] == "test"]["cleaned_history"]
y_test = data[data["Split"] == "test"]["label"]

# 5. 将文本特征转换为TF-IDF向量
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. 训练 SVM 模型
svm = SVC(probability=True, kernel="linear", C=1.0, random_state=42)
svm.fit(X_train_tfidf, y_train)

# 7. 训练 Boosting 模型
boosting = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
boosting.fit(X_train_tfidf, y_train)

# 8. 训练传统模型（朴素贝叶斯）
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# 9. 获取每种方法的预测结果
svm_predictions = svm.predict(X_test_tfidf)
boosting_predictions = boosting.predict(X_test_tfidf)
naive_bayes_predictions = naive_bayes.predict(X_test_tfidf)

# 10. 投票机制
def majority_vote(row):
    # 收集三种模型的预测结果
    predictions = [row["SVM"], row["Boosting"], row["Naive_Bayes"]]
    # 如果三种方法预测一致，则直接返回
    if predictions[0] == predictions[1] == predictions[2]:
        return predictions[0]
    # 如果不一致，随机选择一个
    return random.choice(predictions)

# 构建投票数据
test_data = pd.DataFrame({
    "SVM": svm_predictions,
    "Boosting": boosting_predictions,
    "Naive_Bayes": naive_bayes_predictions,
    "True_Labels": y_test
})

# 对每一行进行投票
test_data["Voting_Result"] = test_data.apply(majority_vote, axis=1)

# 11. 评估性能
actual_labels = test_data["True_Labels"]
predicted_labels = test_data["Voting_Result"]

accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, average="weighted")
recall = recall_score(actual_labels, predicted_labels, average="weighted")
f1 = f1_score(actual_labels, predicted_labels, average="weighted")

# 输出评估结果
print(f"投票算法的准确率：{accuracy:.4f}")
print(f"投票算法的Precision：{precision:.4f}")
print(f"投票算法的Recall：{recall:.4f}")
print(f"投票算法的F1 Score：{f1:.4f}")