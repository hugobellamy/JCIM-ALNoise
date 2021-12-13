# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:26:57 2021

@author: Hugo
"""

import pandas as pd
import numpy as np


def data_import(n):
    # load data from a csv file in standard format
    data_set = pd.read_csv("CHEMBL_Data/" + 'CHEMBL'+str(n)+'.csv')
    X_data = data_set.drop(['molecule_id', 'pXC50'], axis=1).to_numpy()
    Y_given = data_set['pXC50'].to_numpy()
    return([X_data, Y_given])


def split(X, y_true, y_noise, key):
    # split data according to given key
    X_train = []
    y_train_true = []
    y_train_noise = []
    X_bulk = []
    y_bulk_true = []
    y_bulk_noise = []

    for i in range(len(y_true)):
        if i in key:
            X_train.append(X[i])
            y_train_true.append(y_true[i])
            y_train_noise.append(y_noise[i])
        else:
            X_bulk.append(X[i])
            y_bulk_noise.append(y_noise[i])
            y_bulk_true.append(y_true[i])

    return(X_train, y_train_noise, y_train_true, X_bulk, y_bulk_noise,
           y_bulk_true)


def normal_noise(y_true, sigma, seed=0):
    # produce another version of y that intoduces noise
    np.random.seed(seed)
    rands = np.random.normal(0, sigma, len(y_true))
    y_new = y_true+rands
    return(y_new)


def n_percentile(vector, n):
    # find the boundry for the top n% of the vector
    sort = np.sort(vector)
    return(sort[int((1-n)*len(vector))])


def over_threshold(vector, threshold):
    # return number objects over a given threshold
    count = (vector > threshold).sum()
    return(count)


def combine_training_batch(data, batch_info, bulk_info):
    """
   combine data allowing for next model to be made

    data is [X_train, y_train, y_train_true, X_Bulk, y_Bulk, y_Bulk_true]
    where the true values are required due to the implementation of noise

    info is [X, activity, rankings]
    where activity is [y, y_true, predicted_values, predicted_variance]
    """
    X_train = np.vstack((data[0], batch_info[0]))
    y_train = np.append(data[1], batch_info[1][0])

    y_train_true = np.append(data[2], batch_info[1][1])

    X_Bulk = bulk_info[0]
    y_Bulk = bulk_info[1][0]

    y_Bulk_true = bulk_info[1][1]

    new_data = [X_train, y_train,  y_train_true, X_Bulk, y_Bulk,  y_Bulk_true]

    return(new_data)


def active_learner(data, ranking_function, predictions, n, threshold):
    """
    Function to perform an active learning batch.

    Input Details
    predictions are [activity predictions, variance predictions]
    data is [X_train, y_train, y_train_true, X_Bulk, y_Bulk, y_Bulk_true]
    where the true values are required due to the implementation of noise
    """
    rankings = ranking_function(predictions, threshold)

    activity_information = combine_data(data[4], data[5], predictions)

    All_data = combine_data(data[3], activity_information, rankings)

    sorted_data = sort_by_last(All_data)

    batch_data = sorted_data[len(sorted_data)-n:]

    bulk_data = sorted_data[:len(sorted_data)-n]

    batch_info = split_data(batch_data, 1024, 2, 1)
    bulk_info = split_data(bulk_data, 1024, 2, 1)

    batch_info[1] = np.transpose(batch_info[1])
    batch_info.append(np.zeros(len(batch_info[2])))

    bulk_info[1] = np.transpose(bulk_info[1])
    """
    # info is a list object like [chemical info, activity info(training data,
     true data), predictions, retest_counters(all 0)]
    """
    return(batch_info, bulk_info)


def retest_check(batch_info, thresh):
    # check current batch to see if any molecules meet requirement for retest
    length = len(batch_info[2])
    retests = []
    for j in range(length):
        measured_value = batch_info[1][0][j]
        predicted_val = batch_info[2][j]
        counter = batch_info[3][j]
        if measured_value < thresh and predicted_val > thresh and counter == 0:
            retests.append(j)
    return(retests)


def make_retests(batch_info, retests, noise_level, seed):
    # make retest info object from retests and generate new activity values
    chem_info = []
    true_activities = []
    measured_activities = []
    predictions = []
    retest_counters = []

    for item in retests:
        chem_info.append(batch_info[0][item])
        true_activities.append(batch_info[1][1][item])
        new_seed = seed + 50*batch_info[1][1][item]
        np.random.seed(abs(int(new_seed))+1)
        rands = np.random.normal(0, noise_level)
        new_measurement = batch_info[1][1][item]+rands
        measured_activities.append(new_measurement)
        predictions.append(batch_info[2][item])
        retest_counters.append(batch_info[3][item]+1)

    total_info = [chem_info, [measured_activities, true_activities],
                  predictions, retest_counters]
    return(total_info)


def combine_retest_batch(batch_info, retest_info, total_ents):
    # combine the new batch info with the retest info
    if len(batch_info[1][0]) == total_ents:
        return(batch_info)
    else:
        chem_info = np.vstack((batch_info[0], retest_info[0]))
        acti_info = np.hstack((batch_info[1], retest_info[1]))
        predictes = np.append(batch_info[2], retest_info[2])
        rcounters = np.append(batch_info[3], retest_info[3])
        return([chem_info, acti_info, predictes, rcounters])


def prediction_variance(model, X):
    # return variance in predictions between models
    individual_prediction = []
    for m in model.estimators_:
        individual_prediction.append(m.predict(X))

    y_var = np.var(individual_prediction, axis=0)**2
    return(y_var)


def combine_data(*args):
    # combine inputs by adding later imputs as new columns
    if len(np.shape(args[1])) == 1:

        result = hstack(args[0], args[1])
    else:
        result = np.hstack((args[0], args[1]))

    total_elements = len(args)

    if total_elements > 2:
        for i in range(total_elements-2):
            result = hstack(result, args[i+2])
    return(result)


def split_data(matrix, *lengths):
    # seperate the columns of a matrix in the given increments
    data = np.transpose(matrix)

    output = []

    for i in lengths:
        output.append(np.transpose(data[:i]))
        data = data[i:]
    return(output)


def sort_by_last(a):
    # sort input by the values in last column
    row_to_sort_by = np.shape(a)[1]-1

    result = a[a[:, row_to_sort_by].argsort()]
    return(result)


def hstack(obj_1, obj_2):
    # np.hstack that works with vectors
    obj_1_trans = np.transpose(obj_1)

    combined = np.vstack((obj_1_trans, obj_2))
    return(np.transpose(combined))


def true_over_threshold(vector_true, vector_measured, threshold):
    # return number objects over a given threshold
    count = 0
    for i in range(len(vector_true)):
        if vector_true[i] > threshold and vector_measured[i] > threshold:
            count += 1
    return(count)
