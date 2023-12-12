from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import pandas as pd
seed = 0


#-----------本脚本用传统机器学习方法计算分类精度

# np.set_printoptions(threshold=np.inf)
#
num_class = 5
data_train = list()
data_test = list()
label_train = list()
label_test = list()

total_expression_path = 'data/BRCA/BRCA_gene_expression.csv'
total_expression_feature = pd.read_csv(total_expression_path, header=0, index_col=0)

total_bodyMeth_path = 'data/BRCA/BRCA_bodyMeth.csv'
total_bodyMeth_feature = pd.read_csv(total_bodyMeth_path, header=0, index_col=0)

total_proMeth_path = 'data/BRCA/BRCA_proMeth.csv'
total_proMeth_feature = pd.read_csv(total_proMeth_path, header=0, index_col=0)

col_names = list(total_proMeth_feature.columns)
col_names.remove('subtype')
ncol = total_proMeth_feature.shape[1] -1

expression_feature = (total_expression_feature.iloc[:,0:ncol]).to_numpy()
label = (total_expression_feature.iloc[:,-1]).to_numpy()
subtypedic = {'Basal':0,'Normal':1,'Her2':2,'LumA':3,'LumB':4}
label = [subtypedic[item] for item in label]
label = np.array(label)

label2 = (total_proMeth_feature.iloc[:,-1]).to_numpy()
proMeth_feature = (total_proMeth_feature.iloc[:,0:ncol]).to_numpy()

label3 = (total_bodyMeth_feature.iloc[:,-1]).to_numpy()
bodyMeth_feature = (total_bodyMeth_feature.iloc[:,0:ncol]).to_numpy()

indice = np.arange(0,total_expression_feature.shape[0]-1 )
data_train_index, data_test_index, label_train_index, label_test_index = train_test_split(indice, indice, random_state=seed, test_size=0.2)
data_train.append(expression_feature[data_train_index])
label_train = label[data_train_index]
data_test.append(expression_feature[data_test_index])
label_test = label[data_test_index]

data_train.append(proMeth_feature[data_train_index])
data_test.append(proMeth_feature[data_test_index])

data_train.append(bodyMeth_feature[data_train_index])
data_test.append(bodyMeth_feature[data_test_index])

#-----------MOGONET dataset
# train_expression_path = 'D:/MOGONET-main/MOGONET-main/BRCA/1_tr.csv'
# train_expression_feature = pd.read_csv(train_expression_path, header=None)
# data_train.append(train_expression_feature.values)
# train_expression_label_path = 'D:/MOGONET-main/MOGONET-main/BRCA/labels_tr.csv'
# train_expression_label = ((pd.read_csv(train_expression_label_path,header=None)).iloc[:,0]).values.astype(np.uint8)
# label_train = train_expression_label
#
# test_expression_path = 'D:/MOGONET-main/MOGONET-main/BRCA/1_te.csv'
# test_expression_feature = pd.read_csv(test_expression_path, header=None)
# data_test.append(test_expression_feature.values)
# test_expression_label_path = 'D:/MOGONET-main/MOGONET-main/BRCA/labels_te.csv'
# test_expression_label = ((pd.read_csv(test_expression_label_path,header=None))).iloc[:,0].values.astype(np.uint8)
# label_test = test_expression_label
#
# train_meth_path = 'D:/MOGONET-main/MOGONET-main/BRCA/2_tr.csv'
# train_meth_feature = pd.read_csv(train_meth_path, header=None)
# data_train.append(train_meth_feature.values)
#
# test_meth_path = 'D:/MOGONET-main/MOGONET-main/BRCA/2_te.csv'
# test_meth_feature = pd.read_csv(test_meth_path, header=None)
# data_test.append(test_meth_feature.values)
#
# train_miRNA_path = 'D:/MOGONET-main/MOGONET-main/BRCA/3_tr.csv'
# train_miRNA_feature = pd.read_csv(train_miRNA_path, header=None)
# data_train.append(train_miRNA_feature.values)
#
# test_miRNA_path = 'D:/MOGONET-main/MOGONET-main/BRCA/3_te.csv'
# test_miRNA_feature = pd.read_csv(test_miRNA_path, header=None)
# data_test.append(test_miRNA_feature.values)


def SelectModel(modelname):
    if modelname == "SVM":
        from sklearn.svm import SVC

        model = SVC(C=1.0,kernel='linear',probability=True,random_state=seed)

    elif modelname == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()

    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

    elif modelname == "XGBOOST":
        import xgboost as xgb
        model = xgb.XGBClassifier(seed = seed,max_depth=3)

    elif modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        model = knn()
    else:
        pass
    return model


def get_oof(clf, n_folds, X_train, y_train, X_test):
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    classnum = len(np.unique(y_train))
    kf = KFold(n_splits=n_folds, shuffle=False)
    oof_train = np.zeros((ntrain, classnum))
    oof_test = np.zeros((ntest, classnum))

    pd.set_option('display.max_rows', None)  # 显示全部行
    pd.set_option('display.max_columns', None)  # 显示全部列
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index]  # 数据
        kf_y_train = y_train[train_index]  # 标签
        kf_X_test = X_train[test_index]  # k-fold的验证集

        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index] = clf.predict_proba(kf_X_test)
        oof_test += clf.predict_proba(X_test)

    oof_test = oof_test / float(n_folds)
    return oof_train, oof_test,clf

def top_n(lst, n):
    """
    返回前 N 大的元素及其在列表中的位置
    """
    # 将列表转化为 numpy 数组
    arr = np.array(lst)
    # 获取元素在排序之后的位置
    sorted_indexes = np.argsort(arr)[::-1][:n]
    # 获取元素和其位置
    top_n_values = [(lst[i], i) for i in sorted_indexes]
    top_values = [lst[i] for i in sorted_indexes]
    top_indices= [i for i in sorted_indexes]
    return top_indices,top_values




# 训练模型并计算特征重要性

modelist = ['SVM','SVM','SVM','XGBOOST']
newfeature_list = []
newtestdata_list = []
newtraindata = []
newtestdata=[]
print('-------First layer calculating--------')
imp_gene = []
omics_dic = {0:'Feature importance for expression',1:'Feature importance for Promoter methylation',2:'Feature importance for Body methylation'}

for dataset_index in range(0,3):
    clf_first = SelectModel(modelist[dataset_index])
    oof_train_, oof_test_ ,clf_trained= get_oof(clf=clf_first, n_folds=5, X_train=data_train[dataset_index], y_train=label_train,
                                    X_test=data_test[dataset_index])
    print(oof_train_)
    coef = clf_trained.coef_
    to_sort = np.abs(coef[0])
    top_indices, top_values = top_n(to_sort,10)
    gene = [col_names[i] for i in top_indices]
    title = omics_dic[dataset_index]

    plt.bar(range(10),top_values)
    plt.xticks(range(10), gene, rotation=90)
    plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)

    plt.title(title)
    plt.savefig(f'./result/stacking_{title}.png')
    plt.clf()
    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)

    top_indices, top_values = top_n(to_sort, 200)
    gene = [col_names[i] for i in top_indices]

    imp_gene.append(gene)


newtraindata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newfeature_list)
newtestdata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newtestdata_list)
print(newtraindata)
imp_gene =  [j for i in imp_gene for j in i]

import csv
imp_gene =  [[x] for x in imp_gene]

# 写入 CSV 文件函数
filename = './result/important_gene_by_traid_stacking.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in imp_gene:
        writer.writerow(row)

# 特征组合
# 第二级，使用上一级输出的当做训练集
# clf_second1 = RandomForestClassifier()
print('---------Second layer caculating--------')
clf_second1 = SelectModel(modelist[3])
clf_second1.fit(newtraindata, label_train)
fig, ax = plt.subplots(figsize=(12, 8))
xgb.plot_importance(clf_second1, ax=ax, importance_type='weight', max_num_features=20)
pred = clf_second1.predict(newtestdata)
plt.savefig('./result/xgboost_feature_importance_by_tradition.png')

cancer_type = 'brca'
from sklearn.metrics import roc_auc_score

if num_class == 2:
    F1_score = f1_score(label_test, pred)
    accuracy = accuracy_score(label_test, pred)
    score = roc_auc_score(label_test, pred)
    print(f'{cancer_type} : {modelist} accuracy:  {accuracy}\n  F1_macro: {F1_score} \n roc_auc_score: {score}')

else:
    F1_score = f1_score(label_test,pred,average='macro')
    F1_weight = f1_score(label_test,pred,average='weighted')
    right_number = 0
    for i in range(0,len(pred)-1):
        if pred[i] == label_test[i]:
            right_number+=1
    accuracy = right_number/len(label_test)
    print(f'{cancer_type} : {modelist} accuracy:  {accuracy}\n  F1_macro: {F1_score} \n F1_weight: {F1_weight} \n')

import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.set()
fig, ax = plt.subplots()
# for i in range(len(pred)):
#     if pred[i] == 0:
#         pred[i] = 1
# for i in range(len(label_test)):
#     if label_test[i] == 0:
#         label_test[i] = 1

# print(label_test)
# print(pred)

C2 = confusion_matrix(label_test, pred, labels=[0, 1, 2, 3, 4])
im = ax.imshow(C2,cmap = plt.cm.Blues)
ax.set_xticks([0,1,2,3,4])
ax.set_yticks([0,1,2,3,4])
ax.set_xticklabels([0,1,2,3,4])
ax.set_yticklabels([0,1, 2, 3, 4])
for i in range(len([0,1,2,3,4])):
    for j in range(len([0,1,2,3,4])):
        text = ax.text(j,i,C2[i,j],ha = 'center',va = 'center',color = 'white' if C2[i,j] > (C2.max() / 2) else 'black')

ax.set_title('BRCA confusion matrix with traditional stacking')
ax.set_xlabel('predict')
ax.set_ylabel('true')
plt.colorbar(im)
plt.savefig('result/BRCA_tradition_result.png')