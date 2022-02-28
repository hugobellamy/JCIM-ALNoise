# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:10:50 2021

@author: Hugo Bellamy
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


class data_obj:
    
    def __init__(self, source):
        
        data_set = pd.read_csv(source)
        X_data = data_set.drop(['molecule_id', 'pXC50'], axis=1).to_numpy()
        Y_data = data_set['pXC50'].to_numpy()
        
        split_data_1= train_test_split(X_data, Y_data, test_size=0.35,
                                       random_state=12)
        
        self.X_train, X_val_test, self.Y_train, Y_val_test = split_data_1
        
        split_data_2 = train_test_split(X_val_test, Y_val_test,
                                         test_size = 2/3.5, random_state=15)
        
        self.X_val, self.X_test, self.Y_val, self.Y_test = split_data_2
    

    def data(self):
        return(self.X_train, self.Y_train, self.X_val, self.Y_val)


    def total_data(self):
        return(np.vstack((self.X_train,self.X_val)), np.append(self.Y_train, 
                                                               self.Y_val))


    def test_data(self):
        return(self.X_test)


    def test_predictions(self,Y):
        return(mean_squared_error(Y, self.Y_test))
    
    def y_true(self):
        return(self.Y_test)
