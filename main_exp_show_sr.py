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
        
    success_replan_list_naive = []
    replan_list_naive = []
    
    success_replan_list_rejection = []
    replan_list_rejection = []
    
    success_replan_list_retrieval = []
    replan_list_retrieval = []
    
    success_replan_list_all = []
    replan_list_all = []
    
    success_trail_cdf_naive = np.zeros(16)
    success_trail_cdf_rejection = np.zeros(16)
    success_trail_cdf_retrieval = np.zeros(16)
    success_trail_cdf_all = np.zeros(16)
    
    is_success_list_naive = []
    is_success_list_rejection = []
    is_success_list_retrieval = []
    is_success_list_all = []
    
    for i in range(16):
        is_success_list_naive.append([])
        is_success_list_rejection.append([])
        is_success_list_retrieval.append([])
        is_success_list_all.append([])
        
    for result in results_naive:
        # if result['trials'] == 0:
        #     continue
        if result['success']:
            success_replan_list_naive.append(result['trials'])
        replan_list_naive.append(result['trials'])
        
        for i in range(16):
            if result['trials'] <= i and result['success']:
                success_trail_cdf_naive[i] += 1
                is_success_list_naive[i].append(1)
            else:
                is_success_list_naive[i].append(0)
                
    for result in results_rejection:
        # if result['trials'] == 0:
        #     continue
        if result['success']:
            success_replan_list_rejection.append(result['trials'])
        replan_list_rejection.append(result['trials'])
        
        for i in range(16):
            if result['trials'] <= i and result['success']:
                success_trail_cdf_rejection[i] += 1
                is_success_list_rejection[i].append(1)
            else:
                is_success_list_rejection[i].append(0)
                
    for result in results_retrieval:
        # if result['trials'] == 0:
        #     continue
        if result['success']:
            success_replan_list_retrieval.append(result['trials'])
        replan_list_retrieval.append(result['trials'])
        
        for i in range(16):
            if result['trials'] <= i and result['success']:
                success_trail_cdf_retrieval[i] += 1
                is_success_list_retrieval[i].append(1)
            else:
                is_success_list_retrieval[i].append(0)
                
    for result in results_all:
        # if result['trials'] == 0:
        #     continue
        if result['success']:
            success_replan_list_all.append(result['trials'])
        replan_list_all.append(result['trials'])
        
        for i in range(16):
            if result['trials'] <= i and result['success']:
                success_trail_cdf_all[i] += 1
                is_success_list_all[i].append(1)
            else:
                is_success_list_all[i].append(0)
    
    success_trail_cdf_naive = success_trail_cdf_naive / len(results_naive)
    success_trail_cdf_rejection = success_trail_cdf_rejection / len(results_rejection)
    success_trail_cdf_retrieval = success_trail_cdf_retrieval / len(results_retrieval)
    success_trail_cdf_all = success_trail_cdf_all / len(results_all)

    error_naive = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list_naive]
    error_rejection = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list_rejection]
    error_retrieval = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list_retrieval]
    error_all = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list_all]
    
    plt.plot(np.arange(16), success_trail_cdf_naive, label='Naive')
    plt.fill_between(np.arange(16), success_trail_cdf_naive - error_naive, success_trail_cdf_naive + error_naive, alpha=0.3)
    
    plt.plot(np.arange(16), success_trail_cdf_rejection, label='Rejection')
    plt.fill_between(np.arange(16), success_trail_cdf_rejection - error_rejection, success_trail_cdf_rejection + error_rejection, alpha=0.3)
    
    plt.plot(np.arange(16), success_trail_cdf_retrieval, label='Retrieval')
    plt.fill_between(np.arange(16), success_trail_cdf_retrieval - error_retrieval, success_trail_cdf_retrieval + error_retrieval, alpha=0.3)
    
    plt.plot(np.arange(16), success_trail_cdf_all, label='All (ours)')
    plt.fill_between(np.arange(16), success_trail_cdf_all - error_all, success_trail_cdf_all + error_all, alpha=0.3)
    
    plt.xlabel('Number of Replans')
    plt.ylabel('Success Rate')
    plt.title('Success Rate (CDF) v.s. Number of Replans, Task: ' + args.task)
    plt.legend()
    plt.savefig(f'figures/main/main_experiment_success_rate_{args.task}.png')
            
    print(success_trail_cdf_all[-1])
    
if __name__ == '__main__':
    main()