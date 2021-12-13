# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:10:07 2021

@author: Hugo
"""
from scipy.stats import norm
import numpy as np

"""
this code runs the acquistion functions as described in the methods section
of the report
"""


def pdf(z):
    return(np.exp(-z**2/2)/(2*np.pi)**0.5)


def gamma(predictions, y):
    E = 0.01
    return(predictions[0]-y+E)
    

def greedy(predictions, y=None):
    return(predictions[0])


def random(predictions, y=None):
    return(np.random.uniform(size=len(predictions[0])))


def UCB(predictions, y):
    B = 2
    return(predictions[0]+B*predictions[1])


def EI(predictions, y):
    Gamma = gamma(predictions, y)
    result = []
    for i in range(len(Gamma)):
        if predictions[1][i] > 0:
            z = Gamma[i]/predictions[1][i]
            j = Gamma[i]*norm.cdf(z)+predictions[1][i]*pdf(z)
        else:
            j = Gamma[i]
        result.append(j)
    return(result)


def PI(predictions, y):
    Gamma = gamma(predictions, y)
    result = []
    for i in range(len(Gamma)):
        if predictions[1][i] > 0:
            z = Gamma[i]/predictions[1][i]
            j = norm.cdf(z)
        elif Gamma[i] > 0:
            j = 1
        else:
            j = 0
        result.append(j)
    return(result)
