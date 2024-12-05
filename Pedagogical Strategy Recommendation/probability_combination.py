import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# 加载含概率数据的文件
exp_dat_path = 'exp_data_prob.csv'
exp_dat = pd.read_csv(exp_dat_path)


# 解析存储的概率字典
def parse_probabilities(prob_string):
    """
    将字符串形式的概率字典解析为实际的 Python 字典。
    """
    import ast
    return ast.literal_eval(prob_string)


# 仅处理测试集数据
test_data = exp_dat[exp_dat['Split'] == 'test'].copy()

# 提取概率列并解析为字典
test_data['BERT_Probabilities'] = test_data['BERT_Probabilities'].apply(parse_probabilities)
test_data['LPD_Probabilities'] = test_data['LPD_Probabilities'].apply(parse_probabilities)
test_data['BES_Probabilities'] = test_data['BES_Probabilities'].apply(parse_probabilities)

# 设置权重
weights = {
    'BERT_Probabilities': 0.5,
    'LPD_Probabilities': 0.2,
    'BES_Probabilities': 0.3
}

# 标签列表（确保顺序一致）
labels = list(test_data['BERT_Probabilities'].iloc[0].keys())


# 计算加权组合概率
def combine_probabilities(row):
    combined_probs = {label: 0 for label in labels}

    for prob_column, weight in weights.items():
        for label in labels:
            combined_probs[label] += row[prob_column][label] * weight

    return combined_probs


test_data['Combined_Probabilities'] = test_data.apply(combine_probabilities, axis=1)


# 选择最终标签
def select_final_label(prob_dict):
    return max(prob_dict, key=prob_dict.get)


test_data['Combined_Label'] = test_data['Combined_Probabilities'].apply(select_final_label)

# 获取真实标签和预测标签
true_labels = test_data['Peda_Strategies'].tolist()
predicted_labels = test_data['Combined_Label'].tolist()

# 计算评估指标
f1 = f1_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
precision = precision_score(true_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(true_labels, predicted_labels)

# 打印评估结果
print(f"基于概率组合的 F1 Score: {f1:.4f}")
print(f"基于概率组合的 Recall: {recall:.4f}")
print(f"基于概率组合的 Precision: {precision:.4f}")
print(f"基于概率组合的 Accuracy: {accuracy:.4f}")