import pickle
import random
import pandas as pd
import json
import numpy as np

txt_file_path = "D:/Bai/Spc2/SPC2/SPC2/partition_strategy/Name.txt"
with open(txt_file_path, 'r') as file:
    patientID_list = [line.strip() for line in file]

# 定义客户端数量
num_clients = 3

# 划分比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# 计算各部分数据集大小
train_size = int(len(patientID_list) * train_ratio)
val_size = int(len(patientID_list) * val_ratio)
test_size = len(patientID_list) - train_size - val_size

# 随机打乱病人列表
random.shuffle(patientID_list)

# 将数据集划分为训练集、验证集和测试集
train_patient_list = patientID_list[:train_size]
val_patient_list = patientID_list[train_size:train_size + val_size]
test_patient_list = patientID_list[train_size + val_size:]

# 计算有标记客户端和无标记客户端的数据大小
labeled_data_size = int(train_size * 0.2)
unlabeled_data_size = train_size - labeled_data_size

# 将前20%的训练数据分配给有标记客户端
labeled_client_data = train_patient_list[:labeled_data_size]

# 将剩余的80%按照迪利克雷分布分配给两个无标记客户端
dirichlet_distribution = np.random.dirichlet(np.ones(num_clients - 1), size=1)[0]
client_train_sizes = np.round(dirichlet_distribution * unlabeled_data_size).astype(int)

# 分配数据给两个无标记客户端
unlabeled_client_data = []
start_idx = 0
for size in client_train_sizes:
    end_idx = start_idx + size
    unlabeled_client_data.append(train_patient_list[labeled_data_size + start_idx:labeled_data_size + end_idx])
    start_idx = end_idx

# 创建字典以保存划分结果
split_results = {
    'labeled': labeled_client_data,
    'unlabeled': unlabeled_client_data,
    'val': val_patient_list,
    'test': test_patient_list
}

# 将划分结果保存为JSON文件
save_split_path = r"D:/Bai/Spc2/SPC2/SPC2/partition_strategy/CBCT/split_1.pkl"
with open(save_split_path, 'w', encoding='utf-8') as f:
    json.dump(split_results, f)

# 读取划分结果
with open(save_split_path, 'r', encoding='utf-8') as f:
    out = json.load(f)

print(out)
