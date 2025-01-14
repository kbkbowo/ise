import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import os
def main():
    
    parser = argparse.ArgumentParser(description='Calculate score for a given task')

    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-r', '--retrieve', type=str, required=True)
    parser.add_argument('-f', '--refine', type=str, required=True)
    parser.add_argument('-s', '--scratch', type=str, required=True)
    
    args = parser.parse_args()
    
    task_list = ['push_bar', 'pick_bar', 'open_box', 'turn_faucet']
    
    assert args.task in task_list
    
    with open(args.retrieve, 'r') as f:
        results_retrieve = json.load(f)
    with open(args.refine, 'r') as f:
        results_refine = json.load(f)
    with open(args.scratch, 'r') as f:
        results_scratch = json.load(f)
        
    replan_list_retrieve = []
    
    replan_list_refine = []
    
    replan_list_scratch = []
    
    for result in results_retrieve:
        replan_list_retrieve.append(result['trials'])
    
    for result in results_refine:
        replan_list_refine.append(result['trials'])
        
    for result in results_scratch:
        replan_list_scratch.append(result['trials'])
    
    mean_replan_retrieve = np.mean(replan_list_retrieve) 
    std_replan_retrieve = np.std(replan_list_retrieve)
    se_replan_retrieve = std_replan_retrieve / np.sqrt(len(replan_list_retrieve))
    
    mean_replan_refine = np.mean(replan_list_refine)
    std_replan_refine = np.std(replan_list_refine)
    se_replan_refine = std_replan_refine / np.sqrt(len(replan_list_refine))
    
    mean_replan_scratch = np.mean(replan_list_scratch)
    std_replan_scratch = np.std(replan_list_scratch)
    se_replan_scratch = std_replan_scratch / np.sqrt(len(replan_list_scratch))
    
    categoreis = ['Retrieve', 'Retrieve+Refine', 'Retrieve+Refine+From_Scratch']
    
    values = [mean_replan_retrieve, mean_replan_refine, mean_replan_scratch]
    errors = [se_replan_retrieve, se_replan_refine, se_replan_scratch]
    
    plt.bar(categoreis, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
    
    plt.xticks(fontsize=8)
    plt.xlabel('Baselines')
    plt.ylabel('Number of Replans')
    plt.title('Average Number of Replans, Task: ' + args.task)
    # plt.legend()
    plt.savefig(f'figures/retrieve/main_experiment_replan_number_{args.task}.png')
            
    
if __name__ == '__main__':
    main()