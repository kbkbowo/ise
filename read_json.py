import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import os
def main():
    
    parser = argparse.ArgumentParser(description='Calculate score for a given task')

    # parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-r', '--running', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()
    
    # task_list = ['push_bar', 'pick_bar', 'open_box', 'turn_faucet']
    
    # assert args.task in task_list
    
    with open(args.file, 'r') as f:
        results = json.load(f)
        
    if args.running:
        results = results[:-1]
        
    if args.seed != 0:
        results = results[:args.seed]
        
    replan_list = []
    
    
    for result in results:
        replan_list.append(result['trials'])
    
    mean_replan_retrieve = np.mean(replan_list) 
    std_replan_retrieve = np.std(replan_list)
    se_replan_retrieve = std_replan_retrieve / np.sqrt(len(replan_list))
    
    categoreis = ['Retrieve', 'Retrieve+Refine', 'Retrieve+Refine+From_Scratch']
    
    values = [mean_replan_retrieve]
    errors = [se_replan_retrieve]
    
    print(len(replan_list))
    print(f'{values} +- {errors}')
            
    
if __name__ == '__main__':
    main()