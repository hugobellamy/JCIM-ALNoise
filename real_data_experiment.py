# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:26:02 2022

@author: Hugo Bellamy
"""

import CSV_loader as csv
from joblib import Parallel, delayed
import pickle
import os
import numpy as np 
import pandas as pd
import Active_noise_def as alf
import ranking_functions as rnk
from sklearn.ensemble import RandomForestRegressor
import retest_metrics as rtm
import results_analysis as resa
import multiprocessing as mp
import time

def prediction_variance(model, X):
    individual_predictions = []
    for m in model.estimators_:
        individual_predictions.append(m.predict(X))
    y_var = np.var(individual_predictions, axis=0)
    return(y_var)


def test(path):
    """
    perform complete active learning experiment for a given dataset and saves
    result

    Parameters
    ----------
    path : string
        location of csv file to be tested

    Returns
    -------
    None.

    """
    # Load dataset
    dataset = csv.data_obj('qsar_data/'+path)
    X, y = dataset.total_data()
    
    CHEMBL_name = path[:-4]
    
    # sort paramaters for active learning 
    # first what is top 10%
    a = y
    a = 2*a 
    a.sort() 
    crit = a[int(len(a)*0.99)]/2 # this is the only line to change when going between 10% and 1% 
    # how many batches to perfrom
    batch_n =int(len(y)/200)+1
    #range of values for noise
    noise_range = (np.amax(y)-np.amin(y)) 
    noise_factors = np.linspace(0,0.25,6)
    # acquistion metrics to try
    methods = [rnk.greedy, rnk.random, rnk.UCB, rnk.EI, rnk.PI]
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']

    seeds = [658, 682, 533, 27, 889, 224, 205, 338, 559, 163]
    
    repeats = 10
    
    
    for i in noise_factors:
        noise = i*noise_range
        
        for j in range(repeats):
            
            # first run with no retests
            data = {} 
            for k in range(len(labels)):
                
                base = RandomForestRegressor(100, n_jobs=-1, random_state=seeds[repeats-j-1])
                
                AL = alf.Active_learner(base, methods[k], 100, noise,
                                        crit_value=crit, 
                                        retest_metric=rtm.empty)
                # load the data, this sets the seed used for the noise generation
                AL.load_data(X,y,seeds[j])

                AL.active_learn(batch_n, prediction_variance)
                
                data[labels[k]] = AL
                
            hits_list = resa.int_cumulative_data(data, labels)
            true_hits_list = resa.int_true_cumulative_data(data, labels)
            
            all_hits = [hits_list, true_hits_list]
            
            fname = 'results_1%/'+CHEMBL_name+'/noR/AL_noise'+str(i)+'R'+str(j)+'.pkl'
    
            pickle.dump(all_hits, open(fname,'wb'))
            
            # second, repeat with retests
            data2 = {} 
            for k in range(len(labels)):
                base = RandomForestRegressor(100, n_jobs=-1, random_state=seeds[repeats-j-1])
                
                AL2 = alf.Active_learner(base, methods[k], 100, noise,
                                        crit_value=crit, 
                                        retest_metric=rtm.simple)

                AL2.load_data(X,y, seeds[j])

                AL2.active_learn(batch_n, prediction_variance)
                
                data2[labels[k]] = AL2
                
            
            hits_list = resa.int_cumulative_data(data2, labels)
            true_hits_list = resa.int_true_cumulative_data(data2, labels)
            
            all_hits = [hits_list, true_hits_list]
            
            fname = 'results_1%/'+CHEMBL_name+'/withR/AL_noise'+str(i)+'R'+str(j)+'.pkl'
    
            pickle.dump(all_hits, open(fname,'wb'))

        
def exp(name):
    length = len(pd.read_csv('qsar_data/'+name))
    if length>800:  
        CHEMBL_name = name[:-4]
        print('New Dataset '+CHEMBL_name)
        os.mkdir('results_1%/'+CHEMBL_name)
        os.mkdir('results_1%/'+CHEMBL_name+'/noR')
        os.mkdir('results_1%/'+CHEMBL_name+'/withR')
        test(name)
        
    os.remove('qsar_data/'+name)
        
def main():
    
    all_files = os.listdir('qsar_data')
    num_cores = mp.cpu_count()
    
    Parallel(n_jobs=num_cores)(delayed(exp)(all_files[i]) for i in range(len(all_files)))
   
   
            
        
if __name__ == '__main__':
    main()
