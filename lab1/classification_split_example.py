import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# завантажити набір даних IRIS
data_path = os.path.join(os.path.dirname(__file__), 'IRIS.csv')
iris = pd.read_csv(data_path)
print(f"Загальний розмір вибірки: {len(iris)}")

# визначення цільової змінної та набору характеристик
x = iris.drop('species', axis=1)
y = iris['species']

# розбиття набору даних на тренувальний, тестовий, валідаційний та калібраційний за допомогою train_test_split
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.25, random_state=0)
x_test, x_valid_calib, y_test, y_valid_calib = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)
x_valid, x_calib, y_valid, y_calib = train_test_split(x_valid_calib, y_valid_calib, test_size=0.5, random_state=0)

# Збереження індексів з однаковою довжиною для DataFrame
max_len = max(len(x_train), len(x_test), len(x_valid), len(x_calib))
subsample_indices = {
    'train_ind': x_train.index.tolist() + [np.nan] * (max_len - len(x_train)),
    'test_ind': x_test.index.tolist() + [np.nan] * (max_len - len(x_test)),
    'valid_ind': x_valid.index.tolist() + [np.nan] * (max_len - len(x_valid)),
    'calib_ind': x_calib.index.tolist() + [np.nan] * (max_len - len(x_calib))
}

# Запис результатів у файл Excel
save_path = os.path.join(os.path.dirname(__file__), 'subsample_indices.xlsx')
pd.DataFrame(subsample_indices).to_excel(save_path, index=False)
print(f"Результат розбиття збережено в файл 'subsample_indices.xlsx'")
