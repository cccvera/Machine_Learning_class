import pandas as pd

data = pd.read_csv('UCI_Credit_Card.csv')
features = data.iloc[:, 1:-1]
y_label = data['default.payment.next.month']
PAY = data.iloc[:, 6:12]
for i in PAY:
    for j in range(0, len(features[i])):
        if features[i][j] <= 0:
            features[i][j] = -1
        continue
PAY = features.iloc[:, 5:11]
print(PAY)