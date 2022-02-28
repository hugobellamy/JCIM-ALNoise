# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:08:31 2021

@author: Hugo Bellamy
"""


import numpy as np


def empty(keys, y_pred, y_var, y_measured, crit):
    return([])



def simple(keys, y_pred, y_var, y_measured, crit):
    to_retest = []
    for i in range(len(keys)):
        best_mes = np.amax(y_measured[i])
        if y_pred[i]>=crit and best_mes<crit:
            to_retest.append(keys[i])
    return(to_retest)