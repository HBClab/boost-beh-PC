# run Quality Check against new sub data

import os
import sys
import pandas as pd

def parse_cmd_args():
    import argparse
    parser = argparse.ArgumentParser(description='QC for ATS')
    parser.add_argument('-s', type=str, help='Path to submission')
    parser.add_argument('-o', type=str, help='Path to output for QC plots and Logs')
    parser.add_argument('-sub', type=str, help='Subject ID')

    return parser.parse_args()

def df(submission):
    submission = pd.read_csv(submission)
    return submission

def qc(submission):
    # convert submission to DataFrame
    submission = df(submission)
     # check if submission is a DataFrame
    if not isinstance(submission, pd.DataFrame):
        raise ValueError('Submission is not a DataFrame. Could not run QC')
    # check if submission is empty
    if submission.empty:
        raise ValueError('Submission is empty')

    
def plots(submission, output, sub):
    import pandas as pd 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from math import pi

    # Load the data
    df = pd.read_csv(submission)
    
    # Filter the data
    test = df[df['condition'] == 'test']
    
    # Print statements for debugging
    print("DataFrame head:")
    print(df.head())
    print("Filtered DataFrame (test):")
    print(test)
    
    def plot_circular_bar_graph(percentages, name, output_name):
        startangle = 90
        colors = ['#4393E5', '#43BAE5', '#7AE6EA', '#E5A443']
        
        # Convert data to fit the polar axis
        ys = [i *1.1 for i in range(len(percentages))]   # One bar for each block
        left = (startangle * pi * 2) / 360  # This is to control where the bar starts

        # Figure and polar axis
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = plt.subplot(projection='polar')

        # Plot bars and points at the end to make them round
        for i, (block, percentage) in enumerate(percentages.items()):
            ax.barh(ys[i], percentage * 2 * pi, left=left, height=0.5, color=colors[i % len(colors)], label=block)
            ax.text(percentage + left + 0.02, ys[i], f'{percentage:.0%}', va='center', ha='left', color='black', fontsize=12)

        plt.ylim(-1, len(percentages))

        # Custom legend
        ax.legend(loc='center', bbox_to_anchor=(0.5, -0.1), frameon=True) 

        # Clear ticks and spines
        plt.xticks([])
        plt.yticks([])
        ax.spines.clear()
        plt.title(name, fontsize=15, pad=20, color="white")

        plt.savefig(os.path.join(output, f'{sub}_'+output_name+'.png'))
        plt.close()

    correct_by_block = test.groupby('block_c')['correct'].mean()
    print(test.head())

    print(correct_by_block)
    plot_circular_bar_graph(correct_by_block, 'Correct Rate by Block', output_name='_acc_test')
    sns.set(style="whitegrid")
    sns.set_palette("husl")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='block_c', y='response_time', data=test, showfliers=False)
    sns.swarmplot(x='block_c', y='response_time', data=test, color=".25", hue='correct')
    plt.title('Response Time by Block')
    plt.xlabel('Block')
    plt.ylabel('Response Time')
    plt.savefig(os.path.join(output, f'{sub}_rt.png'))

def main():

    #parse command line arguments
    args = parse_cmd_args()
    submission = args.s
    output = args.o
    sub = args.sub

    # check if submission is a csv
    if not submission.endswith('.csv'):
        raise ValueError('Submission is not a csv')
    # check if submission exists
    if not os.path.exists(submission):
        raise ValueError('Submission does not exist')
    # run QC
    qc(submission)
    
    print(f'QC passed for {submission}, generating plots...')
    # generate plots
    plots(submission, output, sub)
    return submission
    
    
if __name__ == '__main__':
    main()
