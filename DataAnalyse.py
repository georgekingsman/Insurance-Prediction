# !/user/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 读取表格数据
path = r'./test.xlsx'
data = pd.read_excel(path)  # 替换为你的表格文件路径和名称

# 将特征列和目标列分开
X = data.drop('86 CARAVAN Number of mobile home policies 0 - 1', axis=1)  # 替换'interested_in_insurance'为你的目标列名称
y = data['86 CARAVAN Number of mobile home policies 0 - 1']  # 替换'interested_in_insurance'为你的目标列名称

# 使用随机森林进行特征选择
model = RandomForestClassifier()
model.fit(X, y)

# 获取特征重要性
feature_importance = model.feature_importances_

# 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# 打印特征重要性排名
# 在运行时接受用户输入的特征数量N
N = int(input("请输入要列出的前N个重要性："))
top_n_features = feature_importance_df.head(N)
print(top_n_features)