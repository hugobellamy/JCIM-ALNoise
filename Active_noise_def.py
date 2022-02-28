# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:53:20 2021

@author: Hugo
"""

import numpy as np
import retest_metrics as rt

def list_to_dict(X, keys=None):
    output = {}
    if keys is None:
        keys = np.linspace(0, len(X), len(X))
    for i in range(len(X)):
        output[i] = X[i]
    return(output)


def upack(Y):
    out = []
    leng = len(Y)
    
    if leng == 1:
        return(list(Y[0]))
    else:
        
        out = list(Y[0])
        for j in range(leng-1):
            out += list(Y[j+1])
    
        return(np.array(out))

def build_train_data(X_dict, Y_dict, indice_list):

    all_ind = upack(indice_list)
    X_train = []
    Y_train = np.array([])
    for a in all_ind:
        L = len(Y_dict[a])
        for i in range(L):
            
            X_train.append(X_dict[a])
      
        Y_train = np.append(Y_train, Y_dict[a])
    Y_train = np.array(Y_train).ravel()
    return(X_train, Y_train)


def build_test_data(X_dict, indice_list, total):

    all_ind = upack(indice_list)
    X_test = []
    keys = []
    for a in range(total):
        if a not in all_ind:
            X_test.append(X_dict[a])
            keys.append(a)
    return(X_test, keys)


def new_indices(scores, keys, n):
    combines = np.transpose([scores, keys])
    sort_combine = combines[np.argsort(combines[:, 0])]
    new_keys = np.transpose(sort_combine)[1][-n:]
    return(new_keys)


def make_noise(Y, noise, seed):
    # produce another version of y that intoduces noise
    np.random.seed(seed)
    rands = np.random.normal(0, noise, len(Y))
    return((Y+rands).reshape(len(Y),1))


def retest(active, y_pred, y_var, keys, batch):
    
    batch_pred = []
    batch_var = [] 
    batch_obs = []
    
    for i in batch:
        position = keys.index(i)
        batch_pred.append(y_pred[position])
        batch_var.append(y_var[position])
        batch_obs.append([active.Y_noise[i]])
        
    return(active.retest_metric(batch, batch_pred, batch_var, batch_obs,
                                active.crit))    


   
    
class Active_learner:

    def __init__(self, Model, acq_metric, batch_size, noise_level,
                 retest_metric=rt.empty, max_retests=0, crit_value=None):
        self.batch_n = 0
        self.Model = Model
        self.metric = acq_metric
        self.batch_size = batch_size
        self.batch_details = []
        self.crit = crit_value
        self.noise = noise_level
        self.next_retests = []
        self.retests = []
        self.retest_metric = retest_metric
        self.max_retests = max_retests

    def load_data(self, X, Y, seed=0):
        self.X = list_to_dict(X)
        self.Y_true = list_to_dict(Y)
        self.Y_noise = list_to_dict(make_noise(Y, self.noise, seed))
        self.retest_n = list_to_dict(np.zeros(len(Y)))
        self.total_entries = len(Y)
        self.seed = seed

    def initial_batch(self, initial=None):
        # if there is no proived intial batch select one randomly based on size
        # of a regular batch
        if initial is None:
            initial = np.random.choice(np.linspace(0, self.total_entries -
                                                   1, self.total_entries),
                                       self.batch_size)
        self.batch_details.append(initial)
        self.batch_n = 1

    def predict_untested(self, var_est):
        X_train, y_train = build_train_data(self.X, self.Y_noise,
                                            self.batch_details)
        
        self.Model.fit(X_train, y_train)
        
        if self.crit is None and self.batch_n == 1:
            self.crit = np.amax(y_train)
            
        X_test, keys = build_test_data(self.X, self.batch_details,
                                       self.total_entries)
        y_pred = self.Model.predict(X_test)
        # plan to change this to allow a range of variance estimate techniques
        y_var = var_est(self.Model, X_test)
        return(y_pred, y_var, keys)

    def select_batch(self, y_pred, y_var, keys):
        # this is a greedy policy to select batch - all entries indepent of
        # eachother
        self.add_retests(self.next_retests)
        self.retests.append(self.next_retests)
        scores = self.metric([y_pred, y_var], self.crit)
        new_batch = new_indices(scores, keys,
                                self.batch_size-len(self.next_retests))
        self.next_retests = retest(self, y_pred, y_var, keys, new_batch)
        if self.max_retests != 0:
            self.next_retests = self.update_retests(self.next_retests)
        self.batch_details.append(new_batch)
        self.batch_n += 1

    def perform_batch(self, var_est):
        y_pred, y_var, keys = self.predict_untested(var_est)
        self.select_batch(y_pred, y_var, keys)

    def active_learn(self, n, var_est, init=None):
        self.initial_batch(init)
        for i in range(n):
            self.perform_batch(var_est)

    def add_retests(self, retest_keys):
        for j in retest_keys:
            new_seed = self.seed + j + self.retest_n[j]
            np.random.seed(int(new_seed))
            new_val = np.random.normal(0, self.noise) + self.Y_true[j]
            self.Y_noise[j] = np.append(self.Y_noise[j], new_val)
            
    def update_retests(self, next_re):
        next_retests = [] 
        for i in next_re:
            current = self.retest_n[i]
            if current<self.max_retests:
                next_retests.append(i)
                self.retest_n[i]+=1
        return(next_retests)