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
    parser.add_argument('-tt', '--total_trial', type=int, default=16)
    
    args = parser.parse_args()
    
    task_list = ['push_bar', 'pick_bar', 'open_box', 'turn_faucet']
    
    assert args.task in task_list
    
    with open(args.retrieve, 'r') as f:
        results_retrieve = json.load(f)
    with open(args.refine, 'r') as f:
        results_refine = json.load(f)
    with open(args.scratch, 'r') as f:
        results_scratch = json.load(f)
        
    success_replan_list_retrieve = []
    replan_list_retrieve = []
    
    success_replan_list_refine = []
    replan_list_refine = []
    
    success_replan_list_scratch = []
    replan_list_scratch = []
    
    success_trail_cdf_retrieve = np.zeros(16)
    success_trail_cdf_refine = np.zeros(16)
    success_trail_cdf_scratch = np.zeros(16)
    
    is_success_list_retrieve = []
    is_success_list_refine = []
    is_success_list_scratch = []
    
    for i in range(16):
        is_success_list_retrieve.append([])
        is_success_list_refine.append([])
        is_success_list_scratch.append([])
        
    for result in results_retrieve:
        # if result['trials'] == 0:
        #     continue
        if result['success']:
            success_replan_list_retrieve.append(result['trials'])
        replan_list_retrieve.append(result['trials'])
        
        for i in range(16):
            if result['trials'] <= i and result['success']:
                success_trail_cdf_retrieve[i] += 1
                is_success_list_retrieve[i].append(1)
            else:
                is_success_list_retrieve[i].append(0)
                
    for result in results_refine:
        # if result['trials'] == 0:
        #     continue
        if result['success']:
            success_replan_list_refine.append(result['trials'])
        replan_list_refine.append(result['trials'])
        
        for i in range(16):
            if result['trials'] <= i and result['success']:
                success_trail_cdf_refine[i] += 1
                is_success_list_refine[i].append(1)
            else:
                is_success_list_refine[i].append(0)
                
    for result in results_scratch:
        # if result['trials'] == 0:
        #     continue
        if result['success']:
            success_replan_list_scratch.append(result['trials'])
        replan_list_scratch.append(result['trials'])
        
        for i in range(16):
            if result['trials'] <= i and result['success']:
                success_trail_cdf_scratch[i] += 1
                is_success_list_scratch[i].append(1)
            else:
                is_success_list_scratch[i].append(0)
                

    success_trail_cdf_retrieve = success_trail_cdf_retrieve / len(results_retrieve)
    success_trail_cdf_refine = success_trail_cdf_refine / len(results_refine)
    success_trail_cdf_scratch = success_trail_cdf_scratch / len(results_scratch)


    error_retrieve = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list_retrieve]
    error_refine = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list_refine]
    error_scratch = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list_scratch]

    plt.plot(np.arange(16), success_trail_cdf_retrieve, label='Retrieve')
    plt.fill_between(np.arange(16), success_trail_cdf_retrieve - error_retrieve, success_trail_cdf_retrieve + error_retrieve, alpha=0.3)
    
    plt.plot(np.arange(16), success_trail_cdf_refine, label='Retrieve+Refine')
    plt.fill_between(np.arange(16), success_trail_cdf_refine - error_refine, success_trail_cdf_refine + error_refine, alpha=0.3)
    
    plt.plot(np.arange(16), success_trail_cdf_scratch, label='Retrieve+Refine+From_Scratch')
    plt.fill_between(np.arange(16), success_trail_cdf_scratch - error_scratch, success_trail_cdf_scratch + error_scratch, alpha=0.3)
    
    plt.xlabel('Number of Replans')
    plt.ylabel('Success Rate')
    plt.title('Success Rate (CDF) v.s. Number of Replans, Task: ' + args.task)
    plt.legend()
    plt.savefig(f'figures/retrieve/main_experiment_success_rate_{args.task}.png')
            
    
if __name__ == '__main__':
    main()