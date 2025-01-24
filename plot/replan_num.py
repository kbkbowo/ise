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
        'ret_origin',
        'all',
    ]

    categories = [
        'Naive',
        'Rej',
        'Ret Origin',
        'All',
    ]
    
    p_dict = {'push_bar': 0.124, 'pick_bar': 0.201, 'open_box': 0.5, 'turn_faucet': 0.5}
    
    avg_values = []
    avg_errors = []
    
    for task in tasks:
        values = []
        errors = []
        
        p = p_dict[task]
        
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
            
        print("VALUES[0]", values[0])
        upper = (1-p)/p
        lower = 1-p
        factor = upper - lower
        values = [(value - lower) / factor for value in values]
        errors = [error / factor for error in errors]
        avg_values.append(values)
        avg_errors.append(errors)
        plt.bar(categories, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
        plt.title(task)
        plt.savefig(f'figures/normal/{task}.png')
        plt.clf()
    
    avg_values = np.mean(avg_values, axis=0)
    avg_errors = np.mean(avg_errors, axis=0) / 2
    
    print(avg_values)
    print(avg_errors)
    
    plt.bar(categories, avg_values, yerr=avg_errors, capsize=5, color='skyblue', edgecolor='black')
    plt.title('all')
    plt.savefig(f'figures/normal/all.png')
    
    plt.clf()
if __name__ == '__main__':
    main()