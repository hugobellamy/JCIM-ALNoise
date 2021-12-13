# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 16:43:08 2021

@author: Hugo
"""
import active_learning_func as af
from sklearn.ensemble import RandomForestRegressor
import ranking_functions as rf
import numpy as np


def prep_data(data_n, sigma, seed):
    """
    import data, setup noise and perfrom initial split
    note data object (D0) is
    [X_train, y_train, y_train_true, X_Bulk, y_Bulk, y_Bulk_true]
    """
    data = af.data_import(data_n)
    initial_batch = 100
    y_bulk_noise = af.normal_noise(data[1], sigma, seed)
    total = len(y_bulk_noise)

    np.random.seed(3*seed)

    starting = np.random.choice(np.linspace(0, total-1, total), initial_batch)

    D0 = af.split(data[0], data[1], y_bulk_noise, starting)
    return(D0)


def active_learn(data, ranking_function):
    # code that perfroms active learning batches with retests
    model = RandomForestRegressor(15)
    n_batch = 100
    active_percent = 0.1

    active_threshold = af.n_percentile(np.append(data[2], data[5]),
                                       active_percent)
    n = 0
    hits = []
    true_hits = []
    hits.append(af.over_threshold(data[2], active_threshold))
    true_hits.append(af.true_over_threshold(data[2], data[1],
                                            active_threshold))

    while n < 22:
        n += 1
        model.fit(data[0], data[1])

        activity_predictions = model.predict(data[3])
        activity_variance = af.prediction_variance(model, data[3])
        predictions = [activity_predictions, activity_variance**0.5]

        batch_info, bulk_info = af.active_learner(data, ranking_function,
                                                  predictions, n_batch,
                                                  active_threshold)

        data = af.combine_training_batch(data, batch_info, bulk_info)

        hits.append(af.over_threshold(data[2], active_threshold))
        true_hits.append(af.true_over_threshold(data[2], data[1],
                                                active_threshold))

    return(hits, true_hits)

"""
Above function was used with following paramaters in experiments, each possible
combination was used

datasets = [203, 204, 205, 228, 233, 251, 253, 260, 267, 313, 333, 339, 340]
methods = [rf.random, rf.greedy, rf.UCB, rf.EI, rf.PI]
noiselevels = [0, 0.5, 1, 1.5, 2, 2.5, 3]
seeds = [64,67,74,3,96,32,22,60,6,99]



Example, using CHEMBL205 dataset,
noise of sigma2 = 0
random seed  = 6
ranking function = rf.greedy

"""
current_data = prep_data(205, 0, 6)
hits, true_hits = active_learn(current_data, rf.greedy)
