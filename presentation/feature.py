import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 绘图包
import numpy as np
color = sns.color_palette()
sns.set_style('darkgrid')

def log_con(dataframe,features):
    for i in dataframe:
        for j in range(0,len(features[i])):
           if  features[i][j] <= 0:
               features[i][j] = -np.log((1 - features[i][j]))
           else:
               features[i][j] = np.log((1 + features[i][j]))
    return features


data = pd.read_csv('UCI_Credit_Card.csv')
features = data.iloc[:, 0:-1]
x_label = data['ID']
PAY = data.iloc[:,6:12]
BILL = data.iloc[:,12:18]
AMT =  data.iloc[:,18:-1]

for i in range(0,len(features['EDUCATION'])):
    if (features['EDUCATION'][i] == 0) or (features['EDUCATION'][i] == 5) or (features['EDUCATION'][i] == 6):
        features['EDUCATION'][i] = 4
    continue
for i in PAY:
    for j in range(0, len(features[i])):
        if features[i][j] <= 0:
            features[i][j] = -1
        continue
for i in range(0,len(features['MARRIAGE'])):
    if features['MARRIAGE'][i] == 0:
        features['MARRIAGE'][i] = 3
    continue

features_change_edu = pd.get_dummies(features['EDUCATION'],dummy_na= False)
features_change_edu.columns = ['EDU_1','EDU_2','EDU_3','EDU_4']
del features['EDUCATION']
features_change_marry = pd.get_dummies(features['MARRIAGE'],dummy_na= False)
features_change_marry.columns = ['MAR_1','MAR_2','MAR_3']
del features['MARRIAGE']
features = pd.merge(features,features_change_edu,left_index=True,right_index=True)
features = pd.merge(features,features_change_marry,left_index=True,right_index=True)

features = log_con(PAY,features)
features = log_con(BILL,features)
features = log_con(AMT,features)
features['LIMIT_BAL'] = np.log((1 + features['LIMIT_BAL']))

plt.figure(figsize=(16, 9))
plt.subplots_adjust(left=0.1, bottom=0.1, wspace=0.6, hspace=0.6)
for i, feature in enumerate(features):
    plt.subplot(4,8, i + 1)
    sns.scatterplot(x=x_label, y=feature, data=features)
    plt.ylabel(feature)
    plt.xlabel('ID')
plt.savefig('1-1.png')
plt.show()

heat = features.corr()
plt.figure(figsize=(16, 9))
sns.heatmap(heat, annot=True)
plt.savefig('1-2.png')
plt.show()
features.to_csv('features.csv',index=False)