# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:58:04 2022

@author: Hugo Bellamy
"""
import numpy as np 
import results_analysis as ra
import matplotlib.pyplot as plt

percent = 1

source = f'results_simulated/{percent}%/noR/AL_noise'
sourceR = f'results_simulated/{percent}%/wR/AL_noise'

repeats = 10

noise_levels = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']

def figure1():
    noise_levels = ['0.0', '1e-01','2e-01']
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
    for i in noise_levels:
        data = ra.load_cumulative_data(source, i,
                                       repeats, labels)
        
        
        length = len(data[labels[0]][0])
        
        x = np.linspace(1, length, length)
        
        for j in range(len(labels)):
            
            
            plt.errorbar(x, np.mean(data[labels[j]], axis=0), np.std(data[labels[j]], axis=0),
                          label=labels[j], capsize=3, color=colors[j])
            
        plt.legend(loc=4, ncol=3)
        plt.xlabel('batch number')
        plt.ylabel('hits')
        # plt.xlim(-0,12)
        plt.savefig(f'figures/simulated{percent}%/fig1-'+str(i)+'.png', dpi=600)
        plt.show()


def figure2():
    noise_levels = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
    colors = dict(zip(labels,colors))
    res = {}
    x = np.linspace(0,0.25,6)
    index = 8
    for i in labels:
        res[i] = []
    for j in range(6):
      
        data = ra.load_cumulative_data(source, noise_levels[j],
                                       repeats, labels)
        
        for k in labels:
            
            enrich = []
            
            for l in range(repeats):
                hits = data[k][l][index]
                
                enrich.append(hits)
                
            res[k].append(enrich)
            
    for i in labels:
        plt.errorbar(x, np.mean(res[i], axis=1), 
                     np.std(res[i], axis=1), label=i, capsize=3, color=colors[i])
    plt.legend(loc=8, ncol=3)
    plt.ylim(-5,55)
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('hits')
    plt.savefig(f'figures/simulated{percent}%/fig2.png', dpi=600)
    plt.show()
    


def figure3():
    noise_levels = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
    labels2 = ['greedy', 'PI']
    c2 = ['tab:blue', 'tab:orange']
    c3 = ['tab:red', 'tab:green']
    c2 = dict(zip(labels2,c2))
    c3 = dict(zip(labels2,c3))
    res = {}
    res_t = {}
    index = 8
    x = np.linspace(0,0.25,6) 
    for i in labels2:
        res[i] = []
        res_t[i] = []
        
    for j in noise_levels:
        data = ra.load_cumulative_data(source, j,
                                       repeats, labels2)
        data_t = ra.load_true_cumulative_data(source,
                                              j, repeats, labels2)
        
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
    plt.savefig(f'figures/simulated{percent}%//fig3.png', dpi=600)
    plt.show()
    
def figure4():
    noise_levels = ['2e-01']
    labels = ['greedy', 'PI']
    c2 = ['tab:blue', 'tab:orange']
    c3 = ['tab:red', 'tab:green']
    c2 = dict(zip(labels,c2))
    c3 = dict(zip(labels,c3))
    res = {}
    repeats = 10 # cannot be 1
    for i in noise_levels:
        data = ra.load_true_cumulative_data(source, i,
                                       repeats, labels)
        
        data2 = ra.load_true_cumulative_data(sourceR, i,
                                       repeats, labels)
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
        plt.savefig(f'figures/simulated{percent}%/fig4-'+str(i)+'.png', dpi=600)
        plt.show()
    

def figure5(a, L=8):
    # should be run with a=4 and a=8
    noise_levels = ['0.0', '5e-02', '1e-01', '1.5000000000000002e-01','2e-01', '2.5e-01']
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
        data_t = ra.load_true_cumulative_data(source,
                                              j, repeats, labels)
        data_retest = ra.load_true_cumulative_data(sourceR,
                                              j, repeats, labels)
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
                     np.std(res_t[i], axis=1), label=i, capsize=3, color=c2[i])
        plt.errorbar(x, np.mean(res_retest[i], axis=1), 
                     np.std(res_retest[i], axis=1), label=i+'(retests)', capsize=3, color=c3[i])
        
    plt.legend(loc=L, ncol=2)
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('true hits')
    plt.ylim(24,54.5)
    plt.savefig(f'figures/simulated{percent}%/fig5'+str(a)+'.png', dpi=600)
    plt.show()
"""
if __name__=='__main__':
    figure1()
    figure2()
    figure3()
    figure4()
    figure5(4)
    figure5(8)
"""
