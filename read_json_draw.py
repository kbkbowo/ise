import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import os
def main():
    
    parser = argparse.ArgumentParser(description='Calculate score for a given task')

    # parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-f', '--files', nargs="+", type=str, required=True)
    parser.add_argument('-c', '--categories', nargs="+", type=str, required=True)
    parser.add_argument('-s', '--seeds', type=int, default=400)
    parser.add_argument('-n ', '--name', type=str, default='test')
    
    parser.add_argument('-r', '--reverse', action='store_true')
    
    args = parser.parse_args()
    
    assert len(args.files) == len(args.categories)
    
    # task_list = ['push_bar', 'pick_bar', 'open_box', 'turn_faucet']
    
    # assert args.task in task_list
    
    values = []
    errors = []
    
    for file in args.files:
    
        with open(file, 'r') as f:
            results = json.load(f)
            
        results = results[:args.seeds]
            
        replan_list = []
        
        
        for result in results:
            replan_list.append(result['trials'])
        
        mean_replan_retrieve = np.mean(replan_list) 
        std_replan_retrieve = np.std(replan_list)
        se_replan_retrieve = std_replan_retrieve / np.sqrt(len(replan_list))
         
        print(len(replan_list))
        print(f'{mean_replan_retrieve} +- {se_replan_retrieve}')
            
        values.append(mean_replan_retrieve)
        errors.append(se_replan_retrieve)
        
    if args.reverse:
        factor = values[0]
        values = [value / factor for value in values]
        errors = [error / factor for error in errors]
        plt.bar(args.categories, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    else:
        plt.bar(args.categories, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    
    plt.savefig(f'figures/{args.name}.png')
if __name__ == '__main__':
    main()