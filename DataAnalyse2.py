# !/user/bin/env python3
# -*- coding: utf-8 -*-
# 卡方检验（Chi-square test）：适用于分类问题，用于衡量特征与目标变量之间的相关性。
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# 读取表格数据
path = r'test.xlsx'
data = pd.read_excel(path)  # 替换为你的表格文件路径和名称

# 将特征列和目标列分开
X = data.drop('86 CARAVAN Number of mobile home policies 0 - 1', axis=1)  # 替换'interested_in_insurance'为你的目标列名称
y = data['86 CARAVAN Number of mobile home policies 0 - 1']  # 替换'interested_in_insurance'为你的目标列名称

# 输入选择的特征数量
k = int(input("请输入需要选择的前几个特征值："))

# 使用卡方检验进行特征选择
selector = SelectKBest(chi2, k=k)  # 选择前k个特征
X_new = selector.fit_transform(X, y)

# 获取选择的特征索引
selected_features = selector.get_support(indices=True)

# 获取选择的特征名称
selected_feature_names = X.columns[selected_features]

print(selected_feature_names)