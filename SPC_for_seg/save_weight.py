import torch
import pandas as pd
import torch.nn.functional as F

column_name = ['文件名', '权重', '所属类别']


# 文件名-权重-所属类别
def save_weight(epoch, item_list, mask_list, label_list):
    # 处理label转换为单个数字
    true_label = torch.argmax(label_list, dim=1)
    true_label = true_label.unsqueeze(1)

    # 拼接张量
    weight = torch.concat([mask_list, true_label], dim=1)
    weight = weight.cpu().numpy()
    # print(weight)

    # 保存权重文件
    df = pd.DataFrame()
    for i in range(label_list.size(0)):
        tp1 = pd.DataFrame([item_list[i]])
        tp2 = pd.DataFrame([weight[i]])
        df = df.append(pd.concat([tp1, tp2], axis=1), ignore_index=True)
    df.columns = column_name
    df.set_index(column_name[0], inplace=True)
    df.to_csv('weight_list_' + str(epoch) + '.csv')


epoch = 10
item_list = ('111.jpg', '222.jpg', '333.jpg', '444.jpg')
item_list = list(item_list)
mask_list = [[0.92],
             [0.55],
             [0.69],
             [0.88]]
mask_list = torch.tensor(mask_list)
label_list = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 1, 0]]
label_list = torch.tensor(label_list)
save_weight(epoch, item_list, mask_list, label_list)
