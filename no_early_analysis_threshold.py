import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    
    parser = argparse.ArgumentParser(description='Calculate score for a given task')
    
    parser.add_argument('-d', '--data_file', type=str, help='Path to the file containing the task')   
    parser.add_argument('-t', '--task_json', type=str, help='Path to the file containing the task')
    # parser.add_argument('-o', '--origin_json', type=str, help='Path to the file containing the task')
    parser.add_argument('-n', '--name', type=str, help='Name of the task')
    
    args = parser.parse_args()
    
    with open(args.task_json, 'r') as f:
        results = json.load(f)
        
    origin_results = results
        
    origin_success_cnt = 0
    origin_replan_num = 0
    for origin_result in origin_results:
        if any(origin_result['success']):
            origin_success_cnt += 1
            origin_replan_num += next((i for i, value in enumerate(origin_result['success']) if value), None)
        else:
            origin_replan_num += 16
        
    origin_success_rate = origin_success_cnt / len(origin_results)
    origin_replan_num_avg = origin_replan_num / len(origin_results)
            
    data = np.load(args.data_file)
    
    print(data.shape)
    
    thresholds = np.linspace(0.6, 0.9, 31)
    
    accuracy_list = []
    early_list = []
    late_list = []
    replan_list = []
    
    for threshold in tqdm(thresholds):
        
        guess_list = [] 
        
        '''
        1 for guess accurately
        0 for guess lately
        -1 for guess early
        '''

        num_replan_list = []

        for i, result in enumerate(results):
            pass
        
            similarity = data[16*i:16*(i+1)]
        
            for j in range(16):
                if similarity[j] > threshold and result['success'][j]:
                    guess_list.append(1)
                    num_replan_list.append(j)
                    break
                elif similarity[j] > threshold and not result['success'][j]:
                    guess_list.append(-1)
                    # num_replan_list.append(j)
                    break
                # elif similarity[j] <= threshold and result['success'][j]:
                #     guess_list.append(0)
                #     break
                else:
                    if j == 15:
                        guess_list.append(-1)
                        num_replan_list.append(16)
                        
                    
        
        # print(len(guess_list))
        if len(guess_list) == 0:
            accuracy_list.append(float('nan'))
            early_list.append(float('nan'))
            replan_list.append(np.mean(num_replan_list))
            continue
        
        guess_list = np.array(guess_list)
        accuracy = np.sum(guess_list == 1) / len(guess_list)
        early = np.sum(guess_list == -1) / len(guess_list)
        # late = np.sum(guess_list == 0) / len(guess_list)
        replan_list.append(np.mean(num_replan_list))
        
        accuracy_list.append(accuracy)
        early_list.append(early)
        # late_list.append(late)
    # print(len(replan_list))
    print(thresholds)
    print(accuracy_list)
    plt.plot(thresholds, accuracy_list, label='Threshold')
    # plt.plot(thresholds, early_list, label='Early')
    plt.plot(thresholds, [origin_success_rate]*len(thresholds), label='Origin')
    # plt.plot(thresholds, late_list, label='Late')
    plt.title(f'side_exp-{args.name}-success_rate')
    plt.xlabel('cosine similarity threshold')
    plt.ylabel('success rate')
    plt.legend()
    plt.savefig(f'figures/side/side_exp-success_rate_{args.name}.png')
    
    plt.clf()
    
    plt.plot(thresholds, replan_list, label='Threshold')
    plt.plot(thresholds, [origin_replan_num_avg]*len(thresholds), label='Origin')
    plt.title(f'side_exp-{args.name}-replan_num')
    plt.xlabel('cosine similarity threshold')
    plt.ylabel('number of replans')
    plt.legend()
    plt.savefig(f'figures/side/side_exp-replan_num_{args.name}.png')            
    
    np.save(f'data/side_exp-{args.name}-sr.npy', np.array(accuracy_list))
    np.save(f'data/side_exp-{args.name}-rp.npy', np.array(replan_list))
    
    print("Threshold: ", thresholds[20], "Accuracy: ", accuracy_list[20], "Replan: ", replan_list[20])
    print("Threshold: ", thresholds[22], "Accuracy: ", accuracy_list[22], "Replan: ", replan_list[22])
    print("Threshold: ", thresholds[24], "Accuracy: ", accuracy_list[24], "Replan: ", replan_list[24])
    print("Threshold: ", thresholds[26], "Accuracy: ", accuracy_list[26], "Replan: ", replan_list[26])
    print("Threshold: ", thresholds[28], "Accuracy: ", accuracy_list[28], "Replan: ", replan_list[28])
    print("Threshold: ", thresholds[30], "Accuracy: ", accuracy_list[30], "Replan: ", replan_list[30])

if __name__ == '__main__':
    main()