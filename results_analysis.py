# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:08:30 2022

@author: Hugo
"""

import numpy as np
import pickle


def get_y(indices, Y_dict):
    ys = []
    for i in indices:
        ys.append(Y_dict[i])
    return(ys)

def over_T(ys, thresh):
    n = 0
    for i in ys:
        if i >= thresh:
            n += 1
    return(n)

def true_over_T(y_true,y_noise,thresh):
    n = 0
    for i in range(len(y_true)):
        if y_true[i]>=thresh and np.amax(y_noise[i])>=thresh:
            n += 1
    return(n)

def actives_per_batch(learner, active_value=None):
    if learner.batch_n <= 1:
        print('Need to perfrom active learning first')
    else:
        if active_value is None:
            active_value = learner.crit
        hits = []
        for j in learner.batch_details:
            ys = get_y(j, learner.Y_true)
            hits.append(over_T(ys, active_value))
    return(hits)


def true_actives_per_batch(learner, active_value=None):
    if learner.batch_n <= 1:
        print('Need to perfrom active learning first')
    else:
        if active_value is None:
            active_value = learner.crit
        hits = []
        for j in learner.batch_details:
            y1 = get_y(j, learner.Y_true)
            y2 = get_y(j, learner.Y_noise)
            hits.append(true_over_T(y1, y2, active_value))
    return(hits)

"""

def missed_active(self, active_value=None):
        if active_value is None:
                active_value = self.crit
        X, keys = build_test_data(self.X, self.batch_details,
                                  self.total_entries)
        ys = get_y(keys, self.Y_true)
        return(over_T(ys, active_value))
    
def build_test_data(X_dict, indice_list, total):

    all_ind = upack(indice_list)
    X_test = []
    keys = []
    for a in range(total):
        if a not in all_ind:
            X_test.append(X_dict[a])
            keys.append(a)
    return(X_test, keys)

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
"""

def to_cumulative(y):
    a = []
    s = 0
    
    for i in y:
        s += i
        a.append(s)
    return(a)

    

def load_cumulative_data(source, noise, n_repeats, labels):
    
    results = {}
    for label in labels:
        results[label] = []
    
    for i in range(n_repeats):
        filename = source + str(noise) +'R'+str(i)+'.pkl'
        
        data = pickle.load(open(filename,'rb'))
        
        for j in labels:
            apb = actives_per_batch(data[j])
            capb = to_cumulative(apb)
            results[j].append(capb)
    
    return(results)


def load_true_cumulative_data(source, noise, n_repeats, labels):
    
    results = {}
    for label in labels:
        results[label] = []
    
    for i in range(n_repeats):
        filename = source + str(noise) +'R'+str(i)+'.pkl'
        
        data = pickle.load(open(filename,'rb'))
        
        for j in labels:
            apb = true_actives_per_batch(data[j])
            capb = to_cumulative(apb)
            results[j].append(capb)
    
    return(results)

def int_cumulative_data(data, labels):
    
    results = {}
        
    for j in labels:
        apb = actives_per_batch(data[j])
        capb = to_cumulative(apb)
        results[j] = capb
    
    return(results)

def int_true_cumulative_data(data, labels):
    
    results = {}
           
    for j in labels:
        apb = true_actives_per_batch(data[j])
        capb = to_cumulative(apb)
        results[j]=capb
    
    return(results)


def dataset(name, retests, noise, index, true):
    """
    

    Parameters
    ----------
    name : str
        dataset name. folder name where data is 
    retests : Bool
        if results wiht retests are required
    noise : float
        noise of experiment
    index : int
        want results after how many batches
    true : Bool
        if want true hits results

    Returns the mean enrichment factor for a given dataset, noise levels, retests
    and index
    -------
    """
   
    source = 'results_10%/'+name
    
    base_source = source+'/noR/AL_noise'+str(noise)+'R'
    
    if retests == True:
        source += '/withR/'
    else:
        source += '/noR/'
    
    source += 'AL_noise'+str(noise)+'R'
       
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    
    repeats = 10 
    
    results = {}
    
    for i in labels:
        results[i] = []
    
    for i in range(repeats):
        
        file = source + str(i) +'.pkl'
        
        base_file = base_source + str(i) +'.pkl'
        
        data = pickle.load(open(file,'rb'))
        
        base_data = pickle.load(open(base_file,'rb'))

        base = base_data[0]['random'][index]
        
        
        if base!=0:
            
            for j in labels:
                results[j].append(data[true][j][index]/base)
    
    res = {}
    for k in labels:
        
        res[k]=np.mean(results[k])
    
    
    return(res, results)
                


    