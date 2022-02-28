# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:58:04 2022

@author: Hugo Bellamy
"""
import numpy as np 
import results_analysis as ra
import matplotlib.pyplot as plt


def figure1():
    noise_levels = [0., 0.1,0.2]
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    repeats = 10 # cannot be 1
    for i in noise_levels:
        data = ra.load_cumulative_data('results/simulated_noR/AL_noise', i,
                                       repeats, labels)
        
        
        length = len(data[labels[0]][0])
        
        x = np.linspace(1, length, length)
        
        for j in labels:
            
            
            plt.errorbar(x, np.mean(data[j], axis=0), np.std(data[j], axis=0),
                          label=j, capsize=3)
        plt.legend()
        plt.xlabel('batch number')
        plt.ylabel('hits')
        # plt.xlim(-0,12)
        plt.savefig('figures/fig1-'+str(i)+'.png', dpi=600)
        plt.show()


def figure2():
    noise_levels = np.linspace(0,0.25,6)
    labels = ['greedy', 'random', 'UCB', 'EI', 'PI']
    res = {}
    repeats = 10
    index = 8
    for i in labels:
        res[i] = []
    for j in range(6):
      
        data = ra.load_cumulative_data('results/simulated_noR/AL_noise', noise_levels[j],
                                       repeats, labels)
        
        for k in labels:
            
            enrich = []
            
            for l in range(repeats):
                hits = data[k][l][index]
                
                enrich.append(hits)
                
            res[k].append(enrich)
            
    for i in labels:
        plt.errorbar(noise_levels, np.mean(res[i], axis=1), 
                     np.std(res[i], axis=1), label=i, capsize=3)
    plt.legend()
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('hits')
    plt.savefig('figures/fig2.png', dpi=600)
    plt.show()
    


def figure3():
    
    noise_levels = np.linspace(0,0.25,6)
    
    labels2 = ['greedy', 'PI']
    res = {}
    res_t = {}
    repeats = 10
    index = 8
    
    for i in labels2:
        res[i] = []
        res_t[i] = []
        
    for j in noise_levels:
      
        data = ra.load_cumulative_data('results/simulated_noR/AL_noise', j,
                                       repeats, labels2)
        data_t = ra.load_true_cumulative_data('results/simulated_noR/AL_noise',
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
        plt.errorbar(noise_levels, np.mean(res[i], axis=1), 
                     np.std(res[i], axis=1), label=i, capsize=3)
        
        plt.errorbar(noise_levels, np.mean(res_t[i], axis=1), 
                     np.std(res_t[i], axis=1), label=i+'(TH)', capsize=3)
    
    
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('hits/true hits')
        
    plt.legend()
    plt.savefig('figures/fig3.png', dpi=600)
    plt.show()
    
def figure4():
    noise_levels = [0.2]
    labels = ['greedy', 'PI']
    repeats = 10 # cannot be 1
    for i in noise_levels:
        data = ra.load_true_cumulative_data('results/simulated_noR/AL_noise', i,
                                       repeats, labels)
        
        data2 = ra.load_true_cumulative_data('results/simulated/AL_noise', i,
                                       repeats, labels)
        
        
        
        length = len(data[labels[0]][0])
        
        x = np.linspace(1, length, length)
        
        for j in labels:
            
            
            plt.errorbar(x, np.mean(data[j], axis=0), np.std(data[j], axis=0),
                          label=j, capsize=3)
            
            plt.errorbar(x, np.mean(data2[j], axis=0), np.std(data2[j], axis=0),
                          label=j+'(retests)', capsize=3)
            
        plt.legend()
        plt.xlabel('batch number')
        # plt.xlim(-0,12)
        plt.ylabel('true hits')
        plt.savefig('figures/fig4-'+str(i)+'.png', dpi=600)
        plt.show()
    

def figure5(a):
    
    noise_levels = np.linspace(0,0.25,6)
    labels = ['greedy', 'PI']
    res_t = {}
    res_retest = {}
    repeats = 10
    index = a
    
    for i in labels:
        res_t[i] = []
        res_retest[i] = []
        
    for j in noise_levels:
      

        data_t = ra.load_true_cumulative_data('results/simulated_noR/AL_noise',
                                              j, repeats, labels)
        
        data_retest = ra.load_true_cumulative_data('results/simulated/AL_noise',
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
        plt.errorbar(noise_levels, np.mean(res_t[i], axis=1), 
                     np.std(res_t[i], axis=1), label=i, capsize=3)
        
        plt.errorbar(noise_levels, np.mean(res_retest[i], axis=1), 
                     np.std(res_retest[i], axis=1), label=i+'(retests)', capsize=3)
        
        
    plt.legend()
    plt.xlabel(r'noise fraction ($\alpha$)')
    plt.ylabel('true hits')
    plt.savefig('figures/fig5'+str(a)+'.png', dpi=600)
    plt.show()
    
    

def main():
    figure1()
    figure2()
    figure3()
    figure4()   
    figure5(4)
    figure5(8)
    
    
if __name__ == '__main__':
    main()