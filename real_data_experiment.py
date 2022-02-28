# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:26:02 2022

@author: Hugo Bellamy
"""

import CSV_loader as csv
import pickle
import os
import numpy as np 
import pandas as pd
import Active_noise_def as alf
import ranking_functions as rnk
from sklearn.ensemble import RandomForestRegressor
import retest_metrics as rtm
import results_analysis as resa


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
    dataset = csv.data_obj('data/'+path)
    X, y = dataset.total_data()
    
    CHEMBL_name = path[:-4]
    
    # sort paramaters for active learning 
    # first what is top 10%
    a = y
    a = 2*a 
    a.sort() 
    crit = a[int(len(a)*0.90)]/2 
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
        print('Starting noise = ', noise, end=' ')
        for j in range(repeats):
            if j!= repeats-1:
                print(j, end=' ')
            else:
                print(j)
            # first run with no retests
            data = {} 
            for k in range(len(labels)):
                
                base = RandomForestRegressor(15, n_jobs=-1)
                
                AL = alf.Active_learner(base, methods[k], 100, noise,
                                        crit_value=crit, 
                                        retest_metric=rtm.empty)

                AL.load_data(X,y,seeds[j])

                AL.active_learn(batch_n, prediction_variance)
                
                data[labels[k]] = AL
                
            hits_list = resa.int_cumulative_data(data, labels)
            true_hits_list = resa.int_true_cumulative_data(data, labels)
            
            all_hits = [hits_list, true_hits_list]
            
            fname = 'results/'+CHEMBL_name+'/noR/AL_noise'+str(i)+'R'+str(j)+'.pkl'
    
            pickle.dump(all_hits, open(fname,'wb'))
            
            # second, repeat with retests
            data2 = {} 
            for k in range(len(labels)):
                base = RandomForestRegressor(15, n_jobs=-1)
                
                AL2 = alf.Active_learner(base, methods[k], 100, noise,
                                        crit_value=crit, 
                                        retest_metric=rtm.simple)

                AL2.load_data(X,y, seeds[j])

                AL2.active_learn(batch_n, prediction_variance)
                
                data2[labels[k]] = AL2
                
            
            hits_list = resa.int_cumulative_data(data2, labels)
            true_hits_list = resa.int_true_cumulative_data(data2, labels)
            
            all_hits = [hits_list, true_hits_list]
            
            fname = 'results/'+CHEMBL_name+'/withR/AL_noise'+str(i)+'R'+str(j)+'.pkl'
    
            pickle.dump(all_hits, open(fname,'wb'))
        
        
def main():
    
    all_files = os.listdir('data')
    
    
    for j in all_files:
        length = len(pd.read_csv('data/'+j))
        if length>800:  
            CHEMBL_name = j[:-4]
            print('New Dataset '+CHEMBL_name)
            os.mkdir('results/'+CHEMBL_name)
            os.mkdir('results/'+CHEMBL_name+'/noR')
            os.mkdir('results/'+CHEMBL_name+'/withR')
            
            test(j)
            
        os.remove('data/'+j)
        
if __name__ == '__main__':
    main()