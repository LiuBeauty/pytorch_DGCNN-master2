from sklearn.svm import LinearSVC
import numpy as np

# 训练数据
X_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y_train = [0, 1, 0]

# 使用线性SVM训练模型
svm = LinearSVC()
svm.fit(X_train, y_train)

# 获取特征权重（系数）
coef = svm.coef_[0]

# 计算特征的绝对权重值
abs_coef = np.abs(coef)

# 获取特征重要性排名
feature_importance_ranking = np.argsort(abs_coef)[::-1]

# 打印特征重要性排名
for rank, feature_index in enumerate(feature_importance_ranking):
    print(f"Rank {rank+1}: Feature {feature_index}, Importance: {abs_coef[feature_index]}")