import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# 加载包含所有预测结果的主数据文件
exp_data_path = "exp_data.csv"
exp_data = pd.read_csv(exp_data_path)

# 筛选出测试集数据
test_data = exp_data[exp_data["Split"] == "test"].copy()  # 使用 .copy() 防止警告

# 确保需要的算法列存在
algorithm_columns = ["BERT", "LPD", "BES"]
missing_columns = [col for col in algorithm_columns if col not in test_data.columns]

if missing_columns:
    raise ValueError(f"以下列缺失，请检查文件：{', '.join(missing_columns)}")

# 定义投票函数
def majority_vote(row, algorithm_columns):
    # 获取当前行的所有算法预测
    predictions = [row[col] for col in algorithm_columns if pd.notna(row[col])]
    if not predictions:
        return None  # 如果没有预测结果，返回空值
    # 投票统计
    most_common = Counter(predictions).most_common()
    top_vote = [label for label, count in most_common if count == most_common[0][1]]
    # 返回最高票的预测，若有并列则返回第一个
    return top_vote[0]

# 对测试集数据应用投票机制
test_data.loc[:, "Voting_Result"] = test_data.apply(majority_vote, axis=1, algorithm_columns=algorithm_columns)

# 评估投票结果
actual_labels = test_data["Peda_Strategies"]  # 实际标签
predicted_labels = test_data["Voting_Result"]  # 投票预测结果

# 去除空值
valid_indices = actual_labels.notna() & predicted_labels.notna()
actual_labels = actual_labels[valid_indices]
predicted_labels = predicted_labels[valid_indices]

# 计算性能指标
accuracy = accuracy_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels, average="weighted", zero_division=1)
recall = recall_score(actual_labels, predicted_labels, average="weighted", zero_division=1)
precision = precision_score(actual_labels, predicted_labels, average="weighted", zero_division=1)

# 输出性能指标
print(f"投票算法的准确率：{accuracy:.4f}")
print(f"投票算法的F1 Score：{f1:.4f}")
print(f"投票算法的Recall：{recall:.4f}")
print(f"投票算法的Precision：{precision:.4f}")
