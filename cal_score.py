import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import os
def main():
    
    parser = argparse.ArgumentParser(description='Calculate score for a given task')
    parser.add_argument('-f', '--file', type=str, nargs='+', help='Path to the file containing the task')
    parser.add_argument('-n', '--name', type=str, help='Name of the task')
    
    args = parser.parse_args()
    
    for file in args.file:
    
        with open(file, 'r') as f:
            results = json.load(f)

        success_replan_list = []
        replan_list = []
        
        results = results[:-1]
        
        success_trail_cdf = np.zeros(15)
        is_success_list = []
        
        for i in range(15):
            is_success_list.append([])
        
        for result in results:
            # if result['trials'] == 0:
            #     continue
            if result['success']:
                success_replan_list.append(result['trials'])
            replan_list.append(result['trials'])
            
            for i in range(15):
                if result['trials'] <= i:
                    success_trail_cdf[i] += 1
                    is_success_list[i].append(1)
                else:
                    is_success_list[i].append(0)
                    
        
        success_trail_cdf = success_trail_cdf / len(results)
        
        print('Number of plans:', len(results))
        print('Success rate:', len(success_replan_list) / len(results))
        print('Average replan trials:', sum(replan_list) / len(replan_list))
        print('Standard deviation:', np.std(replan_list))
        print('Standard Error:', np.std(replan_list) / np.sqrt(len(replan_list)))
        print('Success CDF:', success_trail_cdf)
        
        plt.plot(np.arange(15), success_trail_cdf, label=os.path.basename(file))
        
        # error = np.sqrt(success_trail_cdf * (1 - success_trail_cdf)) / np.sqrt(len(replan_list))
        error = [np.std(is_success)/np.sqrt(len(is_success)) for is_success in is_success_list]
        
        # for is_success in is_success_list:
        #     print(len(is_success))
        #     print(np.std(is_success)/np.sqrt(len(is_success)))
        
        plt.fill_between(np.arange(15), success_trail_cdf - error, success_trail_cdf + error, alpha=0.3)
            
    plt.legend()
    plt.savefig(f'figures/success_cdf_{args.name}.png')
    
    
if __name__ == '__main__':
    main()