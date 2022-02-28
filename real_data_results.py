# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:33:57 2022

@author: Hugo Bellamy
"""
import numpy as np 
import results_analysis as ra
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

def iqr(a):
    
    q75, q25 = np.percentile(a, [75 ,25])
    iqr = q75 - q25
    return(iqr)


def figure6_data():
    # show average enrichment factor after approx 25% of dataset added to data
    
    datasets = os.listdir('results_10%/')
    
    noise_2 = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
    
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    
    all_res = {}
    all_var = {}
    
    for i in labels:
        all_res[i]=[]
        all_var[i]=[]
    
    
    for j in noise_2:
        
        noise_res = {}
        for l in labels:
            noise_res[l] = []
    
        for i in datasets:
            if i[0] =='C':
            
                length = len(pd.read_csv('data/'+i+'.csv'))
            
                index = int(np.round(length/100)*0.25+1)
            
            
            
                data, o = ra.dataset(i, False, j, index, False)
                
        
                
                for k in labels:
                    
                    if data[k]>5:
                        print(i, data[k], k)
                    
                    noise_res[k].append(data[k])
    
        for k in labels:
            all_res[k].append(np.mean(noise_res[k]))
            all_var[k].append(np.std(noise_res[k]))
            
            
    
    filename = open('results/summarised/figure6.pkl','wb')
    
    pickle.dump([all_res, all_var], filename)
    
    
def figure6_graph():
    
    all_res, all_var = pickle.load(open('results/summarised/figure6.pkl','rb'))
    
    
    
    noise_levels = [0.0, 5e-02, 1e-01, 1.5000000000000002e-01,2e-01, 2.5e-01 ]
        
        
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
        
    for i in labels:
        plt.errorbar(noise_levels, all_res[i], np.array(all_var[i]), label=i, capsize=3)
        
    plt.xlabel(r'noise level($\alpha$)')
    plt.ylabel('Enrichment Factor')
    plt.legend()
    plt.savefig('figures/fig6.png', dpi=600)
    plt.show()
    
   

def figure7_data():
    # show true hits/hits after approx 20% of batches 
    datasets = os.listdir('results_10%/')
    
    noise_2 = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
    
    labels = ['greedy', 'PI', 'EI', 'UCB']
    
    all_res = {}
    all_var = {}
    all_rest = {}
    all_vart = {}
    
    for i in labels:
        all_res[i]=[]
        all_var[i]=[]
        all_rest[i]=[]
        all_vart[i]=[]
    
    
    for j in noise_2:
        
        noise_res = {}
        noise_rest = {}
        for l in labels:
            noise_res[l] = []
            noise_rest[l] = []
    
        for i in datasets:
            if i[0] =='C':
            
                length = len(pd.read_csv('data/'+i+'.csv'))
            
                index = int(np.round(length/100)*0.25+1)
            
            
            
                data_hits, o = ra.dataset(i, False, j, index, False)
                data_true, o = ra.dataset(i, False, j, index, True)
                
                for k in labels:
                    noise_res[k].append(data_hits[k])
                    noise_rest[k].append(data_true[k])
    
        for k in labels:
            all_res[k].append(np.mean(noise_res[k]))
            all_var[k].append(np.std(noise_res[k]))
            all_rest[k].append(np.mean(noise_rest[k]))
            all_vart[k].append(np.std(noise_rest[k]))
    
    filename = open('results/summarised/figure7.pkl','wb')
    
    pickle.dump([all_res, all_var, all_rest, all_vart], filename)
    
def figure7_graph():
    
    all_res, all_var, all_rest, all_vart = pickle.load(open('results/summarised/figure7.pkl','rb'))
    
    labels = ['greedy', 'PI', 'EI', 'UCB']
    
    noise_levels = np.linspace(0,0.25,6)
    
    for i in labels:
        plt.errorbar(noise_levels, all_res[i], all_var[i], label=i, capsize=3)
        plt.errorbar(noise_levels, all_rest[i], all_vart[i], label=i+'(TH)', capsize=3)
        
    plt.xlabel(r'noise level($\alpha$)')
    plt.ylabel('Enrichment Factor')
    plt.legend()
    plt.savefig('figures/fig7.png', dpi=600)
    plt.show()

def figure8_data(fraction):
    fraction = fraction
    
    datasets = os.listdir('results_10%/')
    
    noise_2 = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
    
    labels = ['greedy', 'PI']
    
    repeats = 10
    
    all_res = {}
    all_var = {}
    all_var2 = {}
    all_var3 = {}
    all_los = {}
    all_equ = {}
    n = 0
    
    for i in labels:
        all_res[i]=[]
        all_var[i]=[]
        all_equ[i]=[]
        all_var2[i] = [] 
        all_var3[i] = []
        all_los[i] = []
        
    
    
    for j in noise_2:
        
        noise_res = {}
        noise_equ = {}
        noise_los = {}
       
        for l in labels:
            noise_res[l] = np.zeros(10)
            noise_equ[l] = np.zeros(10)
            noise_los[l] = np.zeros(10)
            
    
        for i in datasets:
            if i[0] =='C':
                n+=1
            
                length = len(pd.read_csv('data/'+i+'.csv'))
            
                index = int(np.round(length/100)*fraction+1)
                            
                o, data_true = ra.dataset(i, False, j, index, True)
                o, data_re = ra.dataset(i, True, j, index, True)
                
               
                
                for k in labels:
                    ls = [len(data_re[k]), len(data_true[k])]
                    
                    for l in range(np.amin(ls)):
                        if data_re[k][l]>data_true[k][l]:
                            noise_res[k][l] = noise_res[k][l]+1
                        elif data_re[k][l]==data_true[k][l]:
                            noise_equ[k][l] = noise_equ[k][l]+1
                        else:
                            noise_los[k][l] = noise_los[k][l]+1
                
                        
                        
    
        for k in labels:
            all_res[k].append(np.mean(noise_res[k]))
            all_var[k].append(np.std(noise_res[k]))
            all_equ[k].append(np.mean(noise_equ[k]))
            all_var2[k].append(np.std(noise_equ[k]))
            all_var3[k].append(np.std(noise_los[k]))
            all_los[k].append(np.mean(noise_los[k]))
        print(all_res)
        print(all_equ)
        print(all_los)
            
    filename = open('results/summarised/figure8'+str(fraction)+'.pkl','wb')
    
    pickle.dump([[all_res, all_los, all_equ], [all_var, all_var2, all_var3]], filename)
    
def figure8_graph(fraction):
    
    all_res,all_var = pickle.load(open('results/summarised/figure8'+str(fraction)+'.pkl','rb'))
    
    print(all_res)
    
    labels = ['greedy', 'PI']
    
    l2 = ['retest win', 'retest loss', 'draw']
    n = 288
    
    noise_levels = np.linspace(0,0.25,6)
    
    
 
    
    for i in labels:
       for k in range(2):
           plt.errorbar(noise_levels, np.array(all_res[k][i])*100/n, np.array(all_var[k][i])*100/n, label=i+' - '+l2[k], capsize=3)
       
    #plt.ylim(0,220)
    plt.xlabel(r'noise level($\alpha$)')
    plt.ylabel('retest wins (%)')
    plt.legend()
    plt.savefig('figures/fig8'+str(fraction)+'.png', dpi=600)
    plt.show()
    

"""
if __name__ == '__main__':
    # figure6_data()
    figure6_graph()
    # figure7_data()
    figure7_graph()
    # figure8_data(0.25)
    figure8_graph(0.25)
    #figure8_data(0.4)
    figure8_graph(0.4)
"""
    