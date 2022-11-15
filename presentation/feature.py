import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 绘图包

color = sns.color_palette()
sns.set_style('darkgrid')


# heat map
def heat_mapping(features):
    plt.figure(figsize=(16, 9))
    plt.subplots_adjust(left=0.1, bottom=0.1, wspace=0.6, hspace=0.6)
    for i, feature in enumerate(features):
        plt.subplot(4, 6, i + 1)
        sns.scatterplot(x=y_label, y=feature, data=features)
        plt.ylabel(feature)
        plt.xlabel('default.payment.next.month')
    plt.savefig('1-1.png')
    plt.show()

    heat = features.corr()
    plt.figure(figsize=(16, 9))
    sns.heatmap(heat, annot=True)
    plt.savefig('1-2.png')
    plt.show()






data = pd.read_csv('UCI_Credit_Card.csv')
features = data.iloc[:, 1:-1]
y_label = data['default.payment.next.month']
PAY = data.iloc[:,6:12]

features_change_edu = pd.get_dummies(features['EDUCATION'],dummy_na= False)
del features['EDUCATION']
features_change_marry = pd.get_dummies(features['MARRIAGE'],dummy_na= False)
del features['MARRIAGE']
features = pd.merge(features,features_change_edu,left_index=True,right_index=True)
features = pd.merge(features,features_change_marry,left_index=True,right_index=True)
