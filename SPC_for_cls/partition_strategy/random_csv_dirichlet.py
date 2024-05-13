import pandas as pd
import os
import numpy as np
import random


# choose dataset from [chest, skin, ich]
config = 'chest'
data = {}

if config == 'chest':
    chest = {
        "root_dir": '/datasets/ChestXRay2017/chestxray',
        "file_name": 'all.csv',
        "seed": 1339,
        "beta": 0.8,
        "index": 'image',
        "client_num": 10,
        "label_ratio": 0.1,
        "class_num": 3,
        "class_name": ['normal', 'bacteria', 'virus'],
        "minimum": 10,
    }
    data = chest

root_dir = data["root_dir"]
file_name = data["file_name"]
seed = data["seed"]
beta = data["beta"]
index = data["index"]
client_num = data["client_num"]
label_ratio = data["label_ratio"]
class_num = data["class_num"]
class_name = data["class_name"]
minimum = data["minimum"]

# fix state
random.seed(seed)
np.random.seed(seed)


# divide into 7:1:2
csv_path = root_dir + '/' + file_name
csv_data = pd.read_csv(csv_path)
csv_data.set_index(index, inplace=True)

val_num = np.floor(csv_data.shape[0] * 0.1).astype(int)
test_num = np.floor(csv_data.shape[0] * 0.2).astype(int)
train_num = csv_data.shape[0] - val_num - test_num
# print('train:', train_num, 'test:', test_num, 'val:', val_num)

random_data = csv_data.sample(frac=1, random_state=seed)  # random sample
train_data = random_data[0:train_num]
test_data = random_data[train_num:train_num+test_num]
val_data = random_data[train_num+test_num:]

train_divide = [train_data[class_name[c]].sum().astype(int) for c in range(class_num)]
print('training:', train_divide, 'total:', train_data.shape[0])
test_divide = [test_data[class_name[c]].sum().astype(int) for c in range(class_num)]
print('testing:', test_divide, 'total:', test_data.shape[0])
val_divide = [val_data[class_name[c]].sum().astype(int) for c in range(class_num)]
print('validation:', val_divide, 'total:', val_data.shape[0])

# save csv path
file_path = root_dir + '/' + str(seed)
if not os.path.exists(file_path):
    os.makedirs(file_path)
# print(file_path)
train_data.to_csv(file_path + '/training.csv')
test_data.to_csv(file_path + '/testing.csv')
val_data.to_csv(file_path + '/validation.csv')


# labeled and unlabeled
labeled_num = np.floor(train_data.shape[0] * label_ratio).astype(int)
labeled_data = train_data[0:labeled_num]
unlabeled_data = train_data[labeled_num:]
labeled_divide = [labeled_data[class_name[c]].sum().astype(int) for c in range(class_num)]
print('client_0:', labeled_divide, 'total:', labeled_data.shape[0])
unlabeled_divide = [unlabeled_data[class_name[c]].sum() for c in range(class_num)]

# save csv path
client_path = file_path + '/client_'
for i in range(client_num):
    if not os.path.exists(client_path + str(i)):
        os.makedirs(client_path + str(i))
# print(client_path)
labeled_data.to_csv(client_path + '0' + '/training.csv')


# dirichlet
client_unlabeled = client_num - 1
class_proportions = []
flag = True
while flag:
    class_proportions = []
    flag = False
    for c in range(class_num):
        dirichlet_proportions = np.random.dirichlet(np.repeat(beta, client_unlabeled))
        #dirichlet_proportions = dirichlet_proportions[1:]
        #dirichlet_proportions = dirichlet_proportions / np.sum(dirichlet_proportions)
        dirichlet_divide = (dirichlet_proportions * unlabeled_divide[c]).astype(int)
        for i in range(client_unlabeled):
            if dirichlet_divide[i] < minimum:
                flag = True
                break
        total = np.sum(dirichlet_divide[1:])
        dirichlet_divide[0:1] = unlabeled_divide[c] - total
        class_proportions.append(dirichlet_divide.tolist())

client_proportions = [list(i) for i in zip(*class_proportions)]
for i in range(client_unlabeled):
    print('client_' + str(i+1) + ':', client_proportions[i], 'total:', np.sum(client_proportions[i]))


# read unlabeled data
unlabeled_num = unlabeled_data.shape[0]
class_df = [pd.DataFrame() for i in range(class_num)]
for i in range(unlabeled_num):
    sample = unlabeled_data[i:i+1]
    label = np.array(sample).squeeze(0)
    label = np.argmax(label)
    for c in range(class_num):
        if label == c:
            class_df[c] = pd.concat([class_df[c], sample])

# for each client
client_df = [pd.DataFrame() for i in range(client_unlabeled)]
for i in range(client_unlabeled):
    for c in range(class_num):
        num = client_proportions[i][c]
        temp_df = class_df[c][0:num]
        class_df[c] = class_df[c][num:]
        # print(class_df[c].shape[0])
        client_df[i] = pd.concat([client_df[i], temp_df])
    # print(client_df[i].shape[0])
    client_data = client_df[i].sample(frac=1, random_state=seed)  # random sample
    client_data.to_csv(client_path + str(i+1) + '/training.csv')

print('finish!')
