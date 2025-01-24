import argparse
import json

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_df_data(files, categories):
    seeds_num_list = []
    trials_list = []
    
    for i, file in enumerate(files):
        with open(file) as f:
            results = json.load(f)
        
        seeds_num_list.append(len(results))
        trials_list.extend([result['trials'] for result in results])
        
    category_list = []
    for i, category in enumerate(categories):
        category_list.extend([category for _ in range(seeds_num_list[i])])
        
    data = {
        "Category": category_list,
        "Values": trials_list
    }
    
    return pd.DataFrame(data)
    

def main():
    # Generate example data
    
    parser = argparse.ArgumentParser(description='Violin Plot Example')
    
    parser.add_argument('-f1', '--file1', type=str, nargs='+', help='json_file')
    parser.add_argument('-c1', '--category1', type=str, nargs='+', help='category')
    parser.add_argument('-n1', '--name1', type=str, help='name', default="Task")
    
    parser.add_argument('-f2', '--file2', type=str, nargs='+', help='json_file')
    parser.add_argument('-c2', '--category2', type=str, nargs='+', help='category')
    parser.add_argument('-n2', '--name2', type=str, help='name', default="Task")
    
    parser.add_argument('-f3', '--file3', type=str, nargs='+', help='json_file')
    parser.add_argument('-c3', '--category3', type=str, nargs='+', help='category')
    parser.add_argument('-n3', '--name3', type=str, help='name', default="Task")
    
    parser.add_argument('-f4', '--file4', type=str, nargs='+', help='json_file')
    parser.add_argument('-c4', '--category4', type=str, nargs='+', help='category')
    parser.add_argument('-n4', '--name4', type=str, help='name', default="Task")
    
    args = parser.parse_args()
    
    assert len(args.file1) == len(args.category1), 'The number of files and categories should be the same'
    if args.file2 is not None:
        assert len(args.file2) == len(args.category2), 'The number of files and categories should be the same'
    if args.file3 is not None:
        assert len(args.file3) == len(args.category3), 'The number of files and categories should be the same'
    if args.file4 is not None:
        assert len(args.file4) == len(args.category4), 'The number of files and categories should be the same'

    # Convert to a DataFrame for Seaborn
    df_list = []
    name_list = []
    df1 = get_df_data(args.file1, args.category1)
    df_list.append(df1)
    name_list.append(args.name1)
    if args.file2 is not None:
        df2 = get_df_data(args.file2, args.category2)
        df_list.append(df2)
        name_list.append(args.name2)
    if args.file3 is not None:
        df3 = get_df_data(args.file3, args.category3)
        df_list.append(df3)
        name_list.append(args.name3)
    if args.file4 is not None:
        df4 = get_df_data(args.file4, args.category4)
        df_list.append(df4)
        name_list.append(args.name4)

    subplots = int(args.file1 is not None) + int(args.file2 is not None) + int(args.file3 is not None) + int(args.file4 is not None)
    print(subplots)
    rows = int(subplots == 4) + 1
    cols = subplots if subplots < 4 else 2
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    axs = np.atleast_2d(axs)
    print(axs)
    
    for i in range(rows):
        for j in range(cols):
            sns.violinplot(hue="Category", y="Values", data=df_list[i*cols+j], inner="quartile", bw_method=0.1, palette="muted", ax=axs[i, j])
            axs[i, j].set_title(f"Violin Plot of {name_list[i*cols+j]}")
            axs[i, j].set_xlabel("Baselines")
            axs[i, j].set_ylabel("Number of Replans")

    # Create a violin plot
    # sns.violinplot(hue="Category", y="Values", data=df1, inner="quartile", bw_method=0.1, palette="muted")

    # # Add title and labels
    # plt.title(f"Violin Plot of {args.name}")
    # plt.xlabel("Baselines")
    # plt.ylabel("Number of Replans")

    # Show the plot
    # Show the plot
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.15)
    plt.savefig('test.png')

if __name__ == '__main__':
    main()