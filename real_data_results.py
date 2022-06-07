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

percent = 10


def figure6_data():
    # show average enrichment factor after approx 20% of dataset added to data
    
    datasets = os.listdir('results_'+str(percent)+'%/')
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
            
                length = len(pd.read_csv('qsar_data/'+i+'.csv'))
            
                index = int(np.round(length/100)*0.2-1)
            
                data, _ = ra.dataset(i, False, j, index, False, percent)
                
                for k in labels:
                    noise_res[k].append(data[k])
    
        for k in labels:
            all_res[k].append(np.mean(noise_res[k]))
            all_var[k].append(np.std(noise_res[k]))
            
    filename = open('figures/'+str(percent)+'%summarised/figure6.pkl','wb')
    pickle.dump([all_res, all_var], filename)
    
    
def figure6_graph():
    
    filename = open('figures/'+str(percent)+'%summarised/figure6.pkl','rb')
    all_res, all_var = pickle.load(filename)
    noise_levels = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
        
    for i in labels:
        plt.errorbar(noise_levels, all_res[i], all_var[i], label=i, capsize=3)
        
    plt.xlabel(r'noise level($\alpha$)')
    plt.ylabel('Enrichment Factor')
    plt.legend()
    plt.savefig('figures/real'+str(percent)+'%/fig6.png', dpi=600)
    plt.show()
    
   

def figure7_data():
    # show true hits/hits after approx 20% of batches 
    datasets = os.listdir('results_10%/')
    
    noise_2 = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
    
    labels = ['greedy', 'PI']
    
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
            
                index = int(np.round(length/100)*0.25-1)
            
            
            
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
    
    labels = ['greedy', 'PI']
    
    noise_levels = np.linspace(0,0.25,6)
    
    for i in labels:
        plt.errorbar(noise_levels, all_res[i], all_var[i], label=i)
        plt.errorbar(noise_levels, all_rest[i], all_vart[i], label=i+'(TH)')
        
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
    all_equ = {}
    n = 0
    
    for i in labels:
        all_res[i]=[]
        all_var[i]=[]
        all_equ[i]=[]
        
    
    
    for j in noise_2:
        
        noise_res = {}
        noise_equ = {}
       
        for l in labels:
            noise_res[l] = np.zeros(10)
            noise_equ[l] = np.zeros(10)
            
            
    
        for i in datasets:
            if i[0] =='C':
                n+=1
            
                length = len(pd.read_csv('data/'+i+'.csv'))
            
                index = int(np.round(length/100)*fraction-1)
                            
                o, data_true = ra.dataset(i, False, j, index, True)
                o, data_re = ra.dataset(i, True, j, index, True)
                
               
                
                for k in labels:
                    for l in range(repeats):
                        if data_re[k][l]>data_true[k][l]:
                            noise_res[k][l] = noise_res[k][l]+1
                        elif data_re[k][l]==data_true[k][l]:
                            noise_equ[k][l] = noise_equ[k][l]+1
                        
                        
    
        for k in labels:
            all_res[k].append(np.mean(noise_res[k]))
            all_var[k].append(np.std(noise_res[k]))
            all_equ[k].append(np.mean(noise_equ[k]))
        print(all_res)
        print(all_equ)
            
    filename = open('results/summarised/figure8'+str(fraction)+'.pkl','wb')
    
    pickle.dump([all_res, all_var, all_equ, n], filename)
    
def figure8_graph(fraction):
    
    all_res, all_var, all_eq, n = pickle.load(open('results/summarised/figure8'+str(fraction)+'.pkl','rb'))
    
    
    
    labels = ['greedy', 'PI']
    
    n = 288
    
    noise_levels = np.linspace(0,0.25,6)
    
    
    for i in labels:
        res = []
        var = []
        for j in range(len(all_res[i])):
            res.append(100*all_res[i][j]/(n-all_eq[i][j]))
            var.append(100*all_var[i][j]/(n-all_eq[i][j]))
        
        print(res)
        
        plt.errorbar(noise_levels, res, var, label=i)
       
        
    plt.xlabel(r'noise level($\alpha$)')
    plt.ylabel('retest wins (%)')
    plt.legend()
    plt.savefig('figures/fig8'+str(fraction)+'.png', dpi=600)
    plt.show()
    

"""
if __name__ == '__main__':
    figure6_data()
    figure6_graph()
    figure7_data()
    figure7_graph()
    figure8_data(0.25)
    figure8_graph(0.25)
    figure8_data(0.4)
    figure8_graph(0.4)
"""
    
