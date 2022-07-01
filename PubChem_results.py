# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:58:04 2022

@author: Hugo Bellamy
"""
import numpy as np 
import results_analysis as ra
import matplotlib.pyplot as plt
import pickle

dataset = 1
batch_size = 100

INDEX = int(750/batch_size)+1


source = f'results_PubChem_C/set{dataset}/{batch_size}/noR/'
sourceR = f'results_PubChem_C/set{dataset}/{batch_size}/wR/'

repeats = 10

NOISE_LEVELS = ['0.0', '0.05', '0.1', '0.15000000000000002','0.2', '0.25']


fig_loc = f'figures/PubChem/set{dataset}/{batch_size}/'

def figure1():
    noise_levels = ['0.0', '0.1','0.2']
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
    for i in noise_levels:
        data = pickle.load(open(source+f'LCD{i}.pkl', 'rb')) 
        
        
        length = len(data[labels[0]][0])
        
        x = np.linspace(1, length, length)
        
        for j in range(len(labels)):
            
            
            plt.errorbar(x, np.mean(data[labels[j]], axis=0), np.std(data[labels[j]], axis=0),
                          label=labels[j], capsize=3, color=colors[j])
            
        plt.legend(loc=4, ncol=3)
        plt.xlabel('batch number')
        plt.ylabel('hits')
        # plt.xlim(-0,12)
        plt.savefig(fig_loc + f'fig1-'+str(i)+'.png', dpi=600)
        plt.show()


def figure2():
    noise_levels = NOISE_LEVELS
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
    colors = dict(zip(labels,colors))
    res = {}
    x = np.linspace(0,0.25,6)
    index = INDEX
    for i in labels:
        res[i] = []
    for j in range(6):
        data = pickle.load(open(source+f'LCD{noise_levels[j]}.pkl', 'rb')) 
        for k in labels:
            
            enrich = []
            
            for l in range(repeats):
                hits = data[k][l][index]
                
                enrich.append(hits)
                
            res[k].append(enrich)
            
    for i in labels:
        plt.errorbar(x, np.mean(res[i], axis=1), 
                     np.std(res[i], axis=1), label=i, capsize=3, color=colors[i],
                     ls='none', fmt='o')
    plt.legend(loc=8, ncol=3)
    plt.ylim(25,135)
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('hits')
    plt.savefig(fig_loc + 'fig2.png', dpi=600)
    plt.show()
    


def figure3():
    noise_levels = NOISE_LEVELS
    labels2 = ['greedy', 'PI']
    c2 = ['tab:blue', 'tab:orange']
    c3 = ['tab:red', 'tab:green']
    c2 = dict(zip(labels2,c2))
    c3 = dict(zip(labels2,c3))
    res = {}
    res_t = {}
    index = INDEX
    x = np.linspace(0,0.25,6) 
    for i in labels2:
        res[i] = []
        res_t[i] = []
        
    for j in noise_levels:
        data = pickle.load(open(source+f'LCD{j}.pkl', 'rb')) 
        data_t = pickle.load(open(source+f'LTCD{j}.pkl', 'rb')) 
        
        for k in labels2:
            
            enrich = []
            enrich_t = []
            
            for l in range(repeats):
                hits = data[k][l][index]
                hits_t = data_t[k][l][index]
                
                
                enrich.append(hits)
                enrich_t.append(hits_t)
                
            res[k].append(enrich)
            res_t[k].append(enrich_t)
            
                
    for i in labels2:
        plt.errorbar(x, np.mean(res[i], axis=1), 
                     np.std(res[i], axis=1), label=i, capsize=3, color=c2[i])
        plt.errorbar(x, np.mean(res_t[i], axis=1), 
                     np.std(res_t[i], axis=1), label=i+'(TH)', capsize=3, color=c3[i])
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('hits/true hits')
    plt.legend(loc=8, ncol=2)
    plt.savefig(fig_loc + 'fig3.png', dpi=600)
    plt.show()
    
def figure4():
    noise_levels = ['0.2']
    labels = ['greedy', 'PI']
    c2 = ['tab:blue', 'tab:orange']
    c3 = ['tab:red', 'tab:green']
    c2 = dict(zip(labels,c2))
    c3 = dict(zip(labels,c3))
    res = {}
    repeats = 10 # cannot be 1
    for i in noise_levels:
        data = pickle.load(open(source+f'LTCD{i}.pkl', 'rb')) 
        data2 = pickle.load(open(sourceR+f'LTCD{i}.pkl', 'rb')) 
        length = len(data[labels[0]][0])
        x = np.linspace(1, length, length)
        for j in labels:
            plt.errorbar(x, np.mean(data[j], axis=0), np.std(data[j], axis=0),
                          label=j, capsize=3, color=c2[j])
            plt.errorbar(x, np.mean(data2[j], axis=0), np.std(data2[j], axis=0),
                          label=j+'(retests)', capsize=3, color=c3[j])
            
        plt.legend(loc=8, ncol=2)
        plt.xlabel('batch number')
        # plt.xlim(-0,12)
        plt.ylabel('true hits')
        plt.savefig(fig_loc + f'fig4-'+str(i)+'.png', dpi=600)
        plt.show()
    

def figure5(a, L=8):
    # should be run with 
    noise_levels = NOISE_LEVELS
    labels = ['greedy', 'PI']
    res_t = {}
    x = np.linspace(0,0.25,6)
    res_retest = {}
    repeats = 10
    c2 = ['tab:blue', 'tab:orange']
    c3 = ['tab:red', 'tab:green']
    c2 = dict(zip(labels,c2))
    c3 = dict(zip(labels,c3))
    index = a
    
    for i in labels:
        res_t[i] = []
        res_retest[i] = []
        
    for j in noise_levels:
        data_t = pickle.load(open(source+f'LTCD{j}.pkl', 'rb')) 
        data_retest = pickle.load(open(sourceR+f'LTCD{j}.pkl', 'rb')) 
        for k in labels:
            enrich_t = []
            enrich_retest = []
            for l in range(repeats):
                hits_t = data_t[k][l][index]
                hits_retest = data_retest[k][l][index]
                enrich_t.append(hits_t)
                enrich_retest.append(hits_retest)
            res_t[k].append(enrich_t)
            res_retest[k].append(enrich_retest)
    
    for i in labels:
        plt.errorbar(x, np.mean(res_t[i], axis=1), 
                     np.std(res_t[i], axis=1), label=i, capsize=3, color=c2[i],
                     ls='none', fmt='o')
        plt.errorbar(x, np.mean(res_retest[i], axis=1), 
                     np.std(res_retest[i], axis=1), label=i+'(retests)', capsize=3, color=c3[i],
                     ls='none', fmt='o')
        
    plt.legend(loc=L, ncol=2)
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('true hits')
    #plt.ylim(24,54.5)
    plt.savefig(fig_loc + f'fig5'+str(a)+'.png', dpi=600)
    plt.show()
"""
if __name__=='__main__':
    figure1()
    figure2()
    figure3()
    figure4()
    figure5(INDEX)
    figure5(int(INDEX/2))
"""
