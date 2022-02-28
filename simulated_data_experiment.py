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


def main():
    # generate dataset - noise added later
    
    X, y, coef = datasets.make_regression(n_samples = 5000, n_features = 10,
                                          n_informative = 5, n_targets = 1,
                                          noise = 0.0, coef = True)

    # sort paramaters for active learning 
    # first what is top 10%
    a = y
    a = 2*a 
    a.sort() 
    crit = a[int(len(a)*0.9)]/2 
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
            if j!= 9:
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
            fname = 'results/simulated_noR/AL_noise'+str(i)+'R'+str(j)+'.pkl'
    
            pickle.dump(data, open(fname,'wb'))
            
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
            fname = 'results/simulated/AL_noise'+str(i)+'R'+str(j)+'.pkl'
    
            pickle.dump(data2, open(fname,'wb'))
            
if __name__ == '__main__':
    main()