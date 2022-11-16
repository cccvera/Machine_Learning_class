import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 绘图包
import numpy as np
color = sns.color_palette()
sns.set_style('darkgrid')

data = pd.read_csv('UCI_Credit_Card.csv')
features = data.iloc[:, 1:]
x_label = data['ID']


plt.figure(figsize=(16, 9))
plt.subplots_adjust(left=0.1, bottom=0.1, wspace=0.6, hspace=0.6)
for i, feature in enumerate(features):
    plt.subplot(4, 6, i + 1)
    sns.scatterplot(x=y_label, y=feature, data=features,alpha = 0.7)
    plt.ylabel(feature)
    plt.xlabel('ID')
plt.savefig('1.png')
plt.show()

heat = features.corr()
plt.figure(figsize=(16, 9))
sns.heatmap(heat, annot=True)
plt.savefig('2.png')
plt.show()