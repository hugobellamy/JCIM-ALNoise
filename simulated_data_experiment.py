# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:38:47 2022

@author: Hugo
"""

from sklearn import datasets
import pickle
import numpy as np 
import Active_noise_def as alf
import ranking_functions as rnk
from sklearn.ensemble import RandomForestRegressor
import retest_metrics as rtm

def prediction_variance(model, X):
    individual_predictions = []
    for m in model.estimators_:
        individual_predictions.append(m.predict(X))
    y_var = np.var(individual_predictions, axis=0)
    return(y_var)
 

def main(percent, batch_size=100):
    # generate dataset - noise added later
    X, y = datasets.make_regression(n_samples = 5000, n_features = 10,
                                          noise = 0.0)
    # sort paramaters for active learning 
    # first what is top percent
    a = y.copy()
    a.sort() 
    crit = a[int(len(a)*(100-percent)/100)]
    # how many batches to perfrom
    batch_n = int(len(y)/(2*batch_size))+1
    y = y.reshape(len(y),1)
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
            if j!= 9:
                print(j, end=' ')
            else:
                print(j)
            # first run with no retests
            data = {} 
            for k in range(len(labels)):
                base = RandomForestRegressor(100, n_jobs=-2, random_state=seeds[repeats-j-1])
                AL = alf.Active_learner(base, methods[k], batch_size, noise,
                                        crit_value=crit, 
                                        retest_metric=rtm.empty)
                AL.load_data(X,y,seeds[j])
                AL.active_learn(batch_n, prediction_variance)
                data[labels[k]] = AL
            
            if batch_size == 100:
                fname =f'results_simulated/{percent}%/noR/AL_noise{i}R{j}.pkl'
            else:
                fname = f'results_simulated/batchsize{batch_size}/noR/AL_noise{i}R{j}.pkl'
            pickle.dump(data, open(fname,'wb'))
            
            # second, repeat with retests
            data2 = {} 
            for k in range(len(labels)):
                base = RandomForestRegressor(100, n_jobs=-2, random_state=seeds[repeats-j-1])
                AL2 = alf.Active_learner(base, methods[k], batch_size, noise,
                                        crit_value=crit, 
                                        retest_metric=rtm.simple)
                AL2.load_data(X,y, seeds[j])
                AL2.active_learn(batch_n, prediction_variance)
                data2[labels[k]] = AL2
            if batch_size == 100:
                fname = f'results_simulated/{percent}%/wR/AL_noise{i}R{j}.pkl'
            else:
                fname = f'results_simulated/batchsize{batch_size}/wR/AL_noise{i}R{j}.pkl'
            pickle.dump(data2, open(fname,'wb'))
            
if __name__ == '__main__':
    main(10, batch_size=300)
    main(10, batch_size=500)
