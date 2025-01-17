import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import os
def main():
    
    parser = argparse.ArgumentParser(description='Calculate score for a given task')

    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-n', '--naive', type=str, required=True)
    parser.add_argument('-rj', '--rejection', type=str, required=True)
    parser.add_argument('-rt', '--retrieval', type=str, required=True)
    parser.add_argument('-a', '--all', type=str, required=True)
    parser.add_argument('-tt', '--total_trial', type=int, default=16)
    
    parser.add_argument('-r', '--reverse', action='store_true')
    
    args = parser.parse_args()
    
    task_list = ['push_bar', 'pick_bar', 'open_box', 'turn_faucet']
    
    assert args.task in task_list
    
    with open(args.naive, 'r') as f:
        results_naive = json.load(f)
    with open(args.rejection, 'r') as f:
        results_rejection = json.load(f)
    with open(args.retrieval, 'r') as f:
        results_retrieval = json.load(f)
    with open(args.all, 'r') as f:
        results_all = json.load(f)
        
    replan_list_naive = []
    
    replan_list_rejection = []
    
    replan_list_retrieval = []
    
    replan_list_all = []
    
    for result in results_naive:
        replan_list_naive.append(min(result['trials'], args.total_trial))
    
    for result in results_rejection:
        replan_list_rejection.append(min(result['trials'], args.total_trial))
        
    for result in results_retrieval:
        replan_list_retrieval.append(min(result['trials'], args.total_trial))
        
    for result in results_all:
        replan_list_all.append(min(result['trials'], args.total_trial))
    
    mean_replan_naive = np.mean(replan_list_naive) 
    std_replan_naive = np.std(replan_list_naive)
    se_replan_naive = std_replan_naive / np.sqrt(len(replan_list_naive))
    
    mean_replan_rejection = np.mean(replan_list_rejection)
    std_replan_rejection = np.std(replan_list_rejection)
    se_replan_rejection = std_replan_rejection / np.sqrt(len(replan_list_rejection))
    
    mean_replan_retrieval = np.mean(replan_list_retrieval)
    std_replan_retrieval = np.std(replan_list_retrieval)
    se_replan_retrieval = std_replan_retrieval / np.sqrt(len(replan_list_retrieval))
    
    mean_replan_all = np.mean(replan_list_all)
    std_replan_all = np.std(replan_list_all)
    se_replan_all = std_replan_all / np.sqrt(len(replan_list_all))
    
    categoreis = ['Naive', 'Rejection', 'Retrieval', 'All (ours)']
    
    if args.reverse:
        
        values = [1, mean_replan_rejection/mean_replan_naive, mean_replan_retrieval/mean_replan_naive, mean_replan_all/mean_replan_naive]
        errors = [se_replan_naive/mean_replan_naive, se_replan_rejection/mean_replan_naive, se_replan_retrieval/mean_replan_naive, se_replan_all/mean_replan_naive]
        
        categoreis = ['Naive', 'Rejection', 'Retrieval', 'All (ours)']
        
        plt.bar(categoreis, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
        

        plt.xlabel('Baselines')
        plt.ylabel('Number of Replans')
        plt.title('Average Number of Replans, Task: ' + args.task)
        # plt.legend()
        plt.savefig(f'figures/main/main_experiment_replan_number_{args.task}.png') 
    
    else:

        values = [mean_replan_naive, mean_replan_rejection, mean_replan_retrieval, mean_replan_all]
        errors = [se_replan_naive, se_replan_rejection, se_replan_retrieval, se_replan_all]
        
        plt.bar(categoreis, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')
        

        plt.xlabel('Baselines')
        plt.ylabel('Number of Replans')
        plt.title('Average Number of Replans, Task: ' + args.task)
        # plt.legend()
        plt.savefig(f'figures/main/main_experiment_replan_number_{args.task}.png')
           
    print(mean_replan_all) 
    
if __name__ == '__main__':
    main()