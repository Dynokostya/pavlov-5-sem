import random

import numpy as np

import pandas as pd

# Головний метод розбиття
# Параметр split_method приймає на вхід 2 значення: 'deterministic' (обрання центральної точки страти)
# або 'random' (обрання випадкової точки страти)
def split_data_for_classification(x, y, split_method, test_size, valid_size=0.0, calibr_size=0.0):
    if test_size == 0.0:
        raise ValueError('Test size should be in range [0., 1.] should be provided')
    if test_size < 0.0 or valid_size < 0.0 or calibr_size < 0.0:
        raise ValueError('Test, valid, calibration sizes should be float numbers in range [0., 1.]')
    if calibr_size > 0.0 and valid_size == 0.0:
        raise ValueError('Calibration size cannot be provided without validation size.')

    train_size = 1 - test_size - valid_size - calibr_size
    if train_size < 0.0:
        raise ValueError(
            'Train, test, valid, calibration sizes should be float numbers in range [0., 1.] and sum up to 1.')
    if (train_size + test_size + valid_size + calibr_size) != 1:
        raise ValueError(
            'Train, test, valid, calibration sizes should be float numbers in range [0., 1.] and sum up to 1.')

    x_indexed = [[x.iloc[i], i] for i in range(len(x))] if isinstance(x, pd.DataFrame) else [[x[i], i] for i in
                                                                                             range(len(x))]

    train_ind, test_ind = classification_splitter(x_indexed, y, test_size, split_method)

    if valid_size == 0.0:
        return train_ind, test_ind

    recalculated_valid_size = (len(y) * valid_size) / len(train_ind)
    leftover_x_data_to_split = [x_indexed[i] for i in train_ind]
    leftover_y_data_to_split = [y[i] for i in train_ind]
    train_ind, valid_ind = classification_splitter(leftover_x_data_to_split,
                                                   leftover_y_data_to_split,
                                                   recalculated_valid_size,
                                                   split_method)

    if calibr_size == 0.0:
        return train_ind, test_ind, valid_ind

    recalculated_calibr_size = (len(y) * calibr_size) / len(train_ind)
    leftover_x_data_to_split = [x_indexed[i] for i in train_ind]
    leftover_y_data_to_split = [y[i] for i in train_ind]
    train_ind, calibr_ind = classification_splitter(leftover_x_data_to_split,
                                                    leftover_y_data_to_split,
                                                    recalculated_calibr_size,
                                                    split_method)

    return train_ind, test_ind, valid_ind, calibr_ind


def classification_splitter(x_indexed, y, test_size, method):
    dataset_row_size = len(x_indexed)
    dataset_col_size = len(x_indexed[0][0])
    dataset_length = dataset_col_size - 1

    unique_y = set(y)
    unique_y = set(map(str, unique_y))

    split_target_list = []
    for value in unique_y:
        list_by_target = []
        for row_ind in range(0, dataset_row_size):
            if str(y[row_ind]) == value:
                list_by_target.append(x_indexed[row_ind])
        split_target_list.append(list_by_target)

    dataset_train_ind = []
    dataset_test_ind = []
    for cur_target_list in split_target_list:
        cur_target_list = np.array(cur_target_list)
        cur_target_list_without_ind = []
        for i in range(0, len(cur_target_list)):
            cur_target_list_without_ind.append(cur_target_list[i][0])
        cur_target_list_x = np.array(cur_target_list_without_ind)[:, :dataset_length + 1]

        centroid = cur_target_list_x.mean(axis=0)
        covariance_matrix = np.cov(cur_target_list_x, rowvar=False, ddof=1)
        if abs(np.linalg.det(covariance_matrix)) <= approx(0., sign=+1):
            print("Data row warning: The determinant is 0. The matrix is irreversible.")
        inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        target_list_x_indexed_sorted = get_sorted_mahab_distances(centroid, cur_target_list, inverse_covariance_matrix)
        temp_train_ind, temp_test_ind = resolve_split_method(method, target_list_x_indexed_sorted, test_size)

        dataset_train_ind = dataset_train_ind + temp_train_ind
        dataset_test_ind = dataset_test_ind + temp_test_ind

    return dataset_train_ind, dataset_test_ind


def resolve_split_method(method, target_list_x_indexed_sorted, test_size):
    if method == "random":
        return rand_stratified_split(target_list_x_indexed_sorted, test_size)
    elif method == "deterministic":
        return deterministic_stratified_split(target_list_x_indexed_sorted, test_size)


def deterministic_stratified_split(y, test_size):
    y_test_ind = []
    strata_size_list = get_strata_size_list(len(y), test_size)

    for i in range(0, len(strata_size_list)):
        y_indexed_stratified_begin = sum(strata_size_list[:i])
        y_indexed_stratified_end = y_indexed_stratified_begin + strata_size_list[i]
        cur_strata_size = y_indexed_stratified_end - y_indexed_stratified_begin
        mid_strata_ind = int(cur_strata_size / 2)

        y_test_ind.append(y[y_indexed_stratified_begin + mid_strata_ind][1])

    all_y_indexes = []
    for i in range(0, len(y)):
        all_y_indexes.append(y[i][1])

    y_train_ind = [x for x in all_y_indexes if x not in y_test_ind]
    return y_train_ind, y_test_ind


def rand_stratified_split(y, test_size):
    y_test_ind = []
    strata_size_list = get_strata_size_list(len(y), test_size)

    for i in range(0, len(strata_size_list)):
        y_indexed_stratified_begin = sum(strata_size_list[:i])
        y_indexed_stratified_end = y_indexed_stratified_begin + strata_size_list[i]
        cur_strata_size = y_indexed_stratified_end - y_indexed_stratified_begin
        rand_strata_ind = random.randint(0, cur_strata_size - 1)

        y_test_ind.append(y[y_indexed_stratified_begin + rand_strata_ind][1])

    all_y_indexes = []
    for i in range(0, len(y)):
        all_y_indexes.append(y[i][1])

    y_train_ind = [x for x in all_y_indexes if x not in y_test_ind]
    return y_train_ind, y_test_ind


def get_sorted_mahab_distances(centroid, cur_target_list_x, inverse_covariance_matrix):
    target_list_x_indexed_sorted = sorted(cur_target_list_x,
                                          key=lambda i: mahalanobis(i[0], centroid, inverse_covariance_matrix))
    return target_list_x_indexed_sorted


def mahalanobis(point_from, point_to, inverse_covariance_matrix):
    delta = point_from - point_to
    return max(np.float64(0), np.dot(np.dot(delta, inverse_covariance_matrix), delta)) ** 0.5


def approx(number, *, sign, epsilon=1e-4):
    return number + np.sign(sign) * epsilon


def get_strata_size_list(y_length, test_size):
    min_strata_size = get_min_strata_size(test_size if test_size < 0.5 else (1 - test_size))
    fit_min_stratas = int(y_length / min_strata_size)
    remainder = y_length - (fit_min_stratas * min_strata_size)

    strata_size_list = [min_strata_size for i in range(0, fit_min_stratas)]
    if remainder == 1:
        strata_size_list[fit_min_stratas - 1] += 1
    elif remainder != 0:
        strata_size_list.append(remainder)

    return strata_size_list


def get_min_strata_size(data_size):
    return round(1 / data_size)  # 1 / 0.15 = round(6.66) -> 6 whole numbers fit
