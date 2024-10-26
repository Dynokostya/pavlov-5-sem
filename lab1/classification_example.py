import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

# завантажити набір даних
data_path = os.path.join(os.path.dirname(__file__), 'IRIS.csv')
iris = pd.read_csv(data_path)

x = iris.drop('species', axis=1)
y = iris['species']

# обчислення середніх і ковариаційної матриці для Mahalanobis розрахунків
mean_vec = np.mean(x, axis=0)
cov_matrix = np.cov(x, rowvar=False)
inv_cov_matrix = inv(cov_matrix)

# визначення Mahalanobis відстаней і вибір 30% найвіддаленіших як тестовий набір
x['mahalanobis_dist'] = x.apply(lambda row: mahalanobis(row, mean_vec, inv_cov_matrix), axis=1)
x['is_test'] = x['mahalanobis_dist'] >= x['mahalanobis_dist'].quantile(0.7)

# розбиття на навчальний і тестовий набір
x_train, x_test = x[x['is_test'] == False].drop(columns=['mahalanobis_dist', 'is_test']), \
                  x[x['is_test'] == True].drop(columns=['mahalanobis_dist', 'is_test'])
y_train, y_test = y[x['is_test'] == False], y[x['is_test'] == True]

# Створення моделі логістичної регресії на основі отриманих підвибірок
log_regr = LogisticRegression(random_state=0, max_iter=1000, penalty='l1', solver='liblinear')
log_regr.fit(x_train, y_train)
y_pred_lr = log_regr.predict(x_test)
lr_train = round(log_regr.score(x_train, y_train) * 100, 4)
lr_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 4)
val_score = cross_val_score(estimator=log_regr, X=x_train, y=y_train, cv=10)

print(f"Точність тренування: {lr_train} %, тесту: {lr_accuracy} %, крос валідації: {val_score.mean() * 100} %")
