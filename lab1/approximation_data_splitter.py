import random

import numpy as np
import pandas as pd

# Головний метод розбиття
# Параметр split_method приймає на вхід 2 значення: 'deterministic' (обрання центральної точки страти)
# або 'random' (обрання випадкової точки страти)
def split_data_for_approximation_mahalanobis(x, split_method, test_size, valid_size=0.0, calibr_size=0.0):
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

    x_indexed = [[x.iloc[i], i] for i in range(len(x))] if isinstance(x, pd.DataFrame) else [[x[i], i] for i in range(len(x))]

    centroid = x.mean(axis=0)
    covariance_matrix = np.cov(x, rowvar=False, ddof=1)
    if abs(np.linalg.det(covariance_matrix)) <= approx(0., sign=+1):
        print("Определитель равен 0. Матрица необратима.")
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    target_list_x_indexed_sorted = get_sorted_mahab_distances(centroid, x_indexed, inverse_covariance_matrix)

    train_ind, test_ind = resolve_split_method(split_method, target_list_x_indexed_sorted, test_size)

    if valid_size == 0.0:
        return train_ind, test_ind

    recalculated_valid_size = (len(x) * valid_size) / len(train_ind)
    leftover_data_to_split = [item for item in x_indexed if item[1] in train_ind]
    train_ind, valid_ind = resolve_split_method(split_method, leftover_data_to_split, recalculated_valid_size)

    if calibr_size == 0.0:
        return train_ind, test_ind, valid_ind

    recalculated_calibr_size = (len(x) * calibr_size) / len(train_ind)
    leftover_data_to_split = [item for item in x_indexed if item[1] in train_ind]
    train_ind, calibr_ind = resolve_split_method(split_method, leftover_data_to_split, recalculated_calibr_size)

    return train_ind, test_ind, valid_ind, calibr_ind


def split_data_for_approximation_euclidian(y, split_method, test_size, valid_size=0.0, calibr_size=0.0):
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

    y_indexed = [[y[i], i] for i in range(len(y))]
    train_ind, test_ind = consecutive_split_for_approximation(y_indexed, test_size, split_method)

    if valid_size == 0.0:
        return train_ind, test_ind

    recalculated_valid_size = (len(y) * valid_size) / len(train_ind)
    leftover_data_to_split = [item for item in y_indexed if item[1] in train_ind]
    train_ind, valid_ind = consecutive_split_for_approximation(leftover_data_to_split, recalculated_valid_size,
                                                               split_method)

    if calibr_size == 0.0:
        return train_ind, test_ind, valid_ind

    recalculated_calibr_size = (len(y) * calibr_size) / len(train_ind)
    leftover_data_to_split = [item for item in y_indexed if item[1] in train_ind]
    train_ind, calibr_ind = consecutive_split_for_approximation(leftover_data_to_split, recalculated_calibr_size,
                                                                split_method)

    return train_ind, test_ind, valid_ind, calibr_ind


# split_method: 'deterministic' or 'random'
def consecutive_split_for_approximation(y_indexed, test_size, method):
    y_values = [y_indexed[i][0] for i in range(0, len(y_indexed))]
    mean = np.mean(y_values)
    y_indexed_left = [item for item in y_indexed if item[0] < mean]
    y_indexed_right = [item for item in y_indexed if item[0] > mean]

    y_sorted_left = sorted(y_indexed_left.copy(), key=lambda i: i[0] - mean)
    y_sorted_right = sorted(y_indexed_right.copy(), key=lambda i: mean - i[0])

    y_test_ind = []

    # process data from Y values < mean
    left_strata_size_list = get_strata_size_list(len(y_sorted_left), test_size)
    for i in range(0, len(left_strata_size_list)):
        y_indexed_stratified_begin = sum(left_strata_size_list[:i])
        y_indexed_stratified_end = y_indexed_stratified_begin + left_strata_size_list[i]
        cur_strata_size = y_indexed_stratified_end - y_indexed_stratified_begin
        strata_ind = resolve_strata_elem_number(method, cur_strata_size)

        y_test_ind.append(y_sorted_left[y_indexed_stratified_begin + strata_ind][1])

    # process data from Y values > mean
    right_strata_size_list = get_strata_size_list(len(y_sorted_right), test_size)
    for i in range(0, len(right_strata_size_list)):
        y_indexed_stratified_begin = sum(right_strata_size_list[:i])
        y_indexed_stratified_end = y_indexed_stratified_begin + right_strata_size_list[i]
        cur_strata_size = y_indexed_stratified_end - y_indexed_stratified_begin
        strata_ind = resolve_strata_elem_number(method, cur_strata_size)

        y_test_ind.append(y_sorted_right[y_indexed_stratified_begin + strata_ind][1])

    y_train_ind = [item[1] for item in y_indexed if item[1] not in y_test_ind]

    return y_train_ind, y_test_ind


def get_strata_size_list(y_length, test_size):
    min_strata_size = get_min_strata_size(test_size)
    fit_min_stratas = int(y_length / min_strata_size)
    remainder = y_length - (fit_min_stratas * min_strata_size)

    strata_size_list = [min_strata_size for i in range(0, fit_min_stratas)]
    if remainder == 1:
        strata_size_list[fit_min_stratas - 1] += 1
    elif remainder != 0:
        strata_size_list.append(remainder)

    return strata_size_list


def resolve_strata_elem_number(method, cur_strata_size):
    if method == "random":
        return random.randint(0, cur_strata_size - 1)
    elif method == "deterministic":
        return int(cur_strata_size / 2)


def get_min_strata_size(data_size):
    return int(1 / data_size)  # 1 / 0.15 = 6.66 -> 6 whole numbers fit


# ------------------------ below are methods used for Mahalanobis split -------------------------

def approx(number, *, sign, epsilon=1e-4):
    return number + np.sign(sign) * epsilon


def get_sorted_mahab_distances(centroid, cur_target_list_x, inverse_covariance_matrix):
    target_list_x_indexed_sorted = sorted(cur_target_list_x,
                                          key=lambda i: mahalanobis(i[0], centroid, inverse_covariance_matrix))
    return target_list_x_indexed_sorted


def mahalanobis(point_from, point_to, inverse_covariance_matrix):
    delta = point_from - point_to
    return max(np.float64(0), np.dot(np.dot(delta, inverse_covariance_matrix), delta)) ** 0.5


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
