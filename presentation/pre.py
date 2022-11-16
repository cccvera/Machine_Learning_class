import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as RMSE
from xgboost import plot_importance
import matplotlib.pyplot as plt

import numpy as np
from decimal import Decimal
import shap
from lime import lime_tabular



def picture():
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(model,
                    height=0.5,
                    ax=ax,
                    max_num_features=64)
    plt.savefig('3.png')
    plt.show()
def lime():
    explainer = lime_tabular.LimeTabularExplainer(x_train.values, mode='classification',
                                                  feature_names=x_train.columns.tolist(), categorical_features=None,
                                                  verbose=False, class_names=None)
    i = np.random.randint(0, x_train.shape[0])
    exp = explainer.explain_instance(x_train.values[i], model_xgb.predict_proba, num_features=6)
    exp.show_in_notebook(show_table=True, show_all=False)
    exp.save_to_file('LIME.html')

    list = exp.as_list()
    x = []
    y = []
    for j in list:
        x.append(j[0])
        y.append(j[1])

    plt.figure(figsize=(16, 9))
    plt.barh(x, y, color=["#4CAF50", "red", "hotpink", "#556B2F"])
    plt.title("ID:{}".format(ID[i]), fontdict={'weight': 'normal', 'size': 18})
    plt.xlabel("Value")
    plt.ylabel("Features")
    plt.savefig('ID={}.png'.format(ID[i]))
    plt.show()


def decimal_1(shap_values):
    for i in range(0, len(shap_values)):
        for j in range(0, len(shap_values[i])):
            shap_values[i][j] = round(shap_values[i][j], 2)
    return shap_values


data = pd.read_csv('features.csv')
train,test = TTS(data,test_size=0.2)
x = data.iloc[:,1:-1]
y = data['default.payment.next.month']
x_train = train.iloc[:,1:-1]
y_train = train['default.payment.next.month']
x_test = test.iloc[:,1:-1]
y_test = test['default.payment.next.month']
ID = train['ID']

model = XGBClassifier()
model_xgb = model.fit(x_train,y_train)
y_pre = model.predict(x_test)
#print(model.score(x_test,y_test), RMSE(y_test,model.predict(x_test)), CVS(model,x_train,y_train,cv=5).mean())

k = np.random.randint(0, x_train.shape[0])
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)
shap_values = decimal_1(shap_values)

shap.summary_plot(shap_values, x_train,show=False)
plt.show()
shap.summary_plot(shap_values, x_train, plot_type="bar")
plt.show()