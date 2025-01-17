import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import os
def main():
    
    
    tasks = ['push_bar', 'pick_bar', 'open_box', 'turn_faucet']
    # tasks = ['pick_bar', 'open_box', 'turn_faucet']
    
    files = [
        'naive',
        'rej_only',
        'rej_only_soft',
        'ret_origin',
        'ret_scratch',
        'all',
        'all_soft'
    ]
    
    categories = [
        'Naive',
        'Rej',
        'Rej Soft',
        'Ret Origin',
        'Ret Scratch',
        'All',
        'All Soft'
    ]
    
    avg_values = []
    avg_errors = []
    
    for task in tasks:
        values = []
        errors = []
        for file in files:
            
            file_name = f'jsons/{task}/{file}.json'
        
            with open(file_name, 'r') as f:
                print(file_name)
                results = json.load(f)
                

            replan_list = []
            
            
            for result in results:
                replan_list.append(result['trials'])
            
            mean_replan_retrieve = np.mean(replan_list) 
            std_replan_retrieve = np.std(replan_list)
            se_replan_retrieve = std_replan_retrieve / np.sqrt(len(replan_list))
            
                
            values.append(mean_replan_retrieve)
            errors.append(se_replan_retrieve)
            
        factor = values[0]
        values = [value / factor for value in values]
        errors = [error / factor for error in errors]
        avg_values.append(values)
        avg_errors.append(errors)
        plt.bar(categories, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')

        plt.savefig(f'figures/normal/{task}.png')
        plt.clf()
    
    avg_values = np.mean(avg_values, axis=0)
    avg_errors = np.mean(avg_errors, axis=0)
    
    print(avg_values)
    print(avg_errors)
    
    plt.bar(categories, avg_values, yerr=avg_errors, capsize=5, color='skyblue', edgecolor='black')
    plt.savefig(f'figures/normal/all.png')
    plt.clf()
if __name__ == '__main__':
    main()