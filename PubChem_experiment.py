# -*- coding: utf-8 -*-
import pickle
import numpy as np 
import Active_noise_def as alf
import ranking_functions as rnk
from sklearn.ensemble import RandomForestRegressor
import retest_metrics as rtm
import pandas as pd

def prediction_variance(model, X):
    individual_predictions = []
    for m in model.estimators_:
        individual_predictions.append(m.predict(X))
    y_var = np.var(individual_predictions, axis=0)
    return(y_var)
 

def main(batch_size=100):
    # load dataset - noise added later
    X = pd.read_csv('PubChem_data/Real1_fingerprints.csv')
    X = X.drop(columns=['Unnamed: 0'])
    X = X.to_numpy()
    y = pd.read_csv('PubChem_data/Real1_targets.csv')
    y = y.drop(columns=['Unnamed: 0'])
    y = y.to_numpy()
    # sort paramaters for active learning 
    crit = 40 # Defined by pubchem as actives
    # how many batches to perfrom
    batch_n = int(len(y)/(2*batch_size))+1
    #range of values for noise
    noise_range = (np.amax(y)-np.amin(y)) 
    noise_factors = np.linspace(0,0.25,6)
    # acquistion metrics to try
    methods = [rnk.greedy, rnk.random, rnk.UCB, rnk.EI, rnk.PI]
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    seeds = [658, 682, 533, 27, 889, 224, 205, 338, 559, 163]
    repeats = 10
    file_base = f'results_PubChem/set1/{batch_size}'
    
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
            
            fname = file_base+ f'/noR/AL_noise{i}R{j}.pkl'
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
            fname = file_base + f'/wR/AL_noise{i}R{j}.pkl'
            pickle.dump(data2, open(fname,'wb'))
            
if __name__ == '__main__':
    main()
    main(300)
    main(500)
    main(1000)

