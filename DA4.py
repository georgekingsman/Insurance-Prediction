# !/user/bin/env python3
# -*- coding: utf-8 -*-
# L1 正则化（L1 Regularization）：适用于分类或回归问题，通过加入 L1 正则项来约束模型参数，使得部分特征的权重变为0，从而实现特征选择。
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 读取表格数据
path = r'./test.xlsx'
data = pd.read_excel(path)  # 替换为你的表格文件路径和名称

# 将特征列和目标列分开
X = data.drop('86 CARAVAN Number of mobile home policies 0 - 1', axis=1)  # 替换'interested_in_insurance'为你的目标列名称
y = data['86 CARAVAN Number of mobile home policies 0 - 1']  # 替换'interested_in_insurance'为你的目标列名称

# 输入选择的特征数量
k = int(input("请输入需要选择的前几个特征值："))

# 使用 L1 正则化进行特征选择
model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(X, y)

# 获取选择的特征索引
selected_features = model.coef_[0].argsort()[:k]

# 获取选择的特征名称
selected_feature_names = X.columns[selected_features]

print(selected_feature_names)