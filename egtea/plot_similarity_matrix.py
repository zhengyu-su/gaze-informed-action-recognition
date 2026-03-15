import numpy as np
import time
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import random

def needleman_wunsch_algo(matrix1, matrix2, gap=0, match=2, missmatch=1):
    # wholestart = time.time()
    rows = matrix1.shape[0] + 1
    columns = matrix2.shape[0] + 1

    # Initialization
    tab = np.zeros((rows, columns), dtype=float)
    for i in range(1, rows):
        tab[i][0] = tab[i - 1][0] + gap
    for i in range(1, columns):
        tab[0][i] = tab[0][i - 1] + gap
    
    # Update the table
    for i in range(1, rows):
        for j in range(1,columns):
            if matrix1[i-1] == matrix2[j-1]:
                score = tab[i - 1][j - 1] + match
            else:
                score = tab[i - 1][j - 1] + missmatch

            gap_score_1 = tab[i - 1][j] + gap
            gap_score_2 = tab[i][j - 1] + gap
            tab[i][j] = max(gap_score_1, gap_score_2, score)
    
    tab = tab / max(rows, columns)    # Normalization

    similarity = tab[rows-1][columns-1]

    # wholeend = time.time()
    # print("Processing Time", (wholeend - wholestart))
    return similarity, tab

def get_similarity_matrix(exp_name, action_name):
    # Get the token scan paths for the action
    # action_folder = os.path.join('/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Results/EGTEAGaze+/', exp_name,'train', action_name)
    action_folder = os.path.join(r"C:\Users\Su\Documents\Codes\Results\EGTEA", exp_name,'train', action_name)
    print(f"action folder: {action_folder}")
    scan_path_files = []
    for root, dirs, files in os.walk(action_folder):
        for file in files:
            if file.endswith('.csv'):
                scan_path_files.append(os.path.join(root, file))
    n = len(scan_path_files)
    print(f"scanpath files: {scan_path_files}")

    print(f'Calculating similarity for {action_name}...')
    similarity_matrix = np.zeros((n, n))
    for i, file_1 in enumerate(scan_path_files):
        for j, file_2 in enumerate(scan_path_files):
            # Load the scan path from files
            scan_path_1 = np.array(pd.read_csv(file_1)['Label'])
            scan_path_2 = np.array(pd.read_csv(file_2)['Label'])

            # Calculate the similarity
            similarity, _ = needleman_wunsch_algo(scan_path_1, scan_path_2)
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix, n

def get_all_similarity_matrix(exp_name):
    # Get all action names
    # exp_folder = os.path.join('/Users/xiaoxiao/Documents/Studium/Lab/GazeEstimation/Results/EGTEAGaze+/', exp_nr)
    exp_folder = os.path.join(r"C:\Users\Su\Documents\Codes\Results\EGTEA", exp_name)
    train_folder = os.path.join(exp_folder, 'train')
    action_names = [os.path.basename(f.path) for f in os.scandir(train_folder) if f.is_dir()]
    print(f"action names: {action_names}")
    plot_folder = os.path.join(exp_folder, 'plots')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for action_name in action_names:
        similarity_matrix, n = get_similarity_matrix(exp_name, action_name)
        # Plot the similarity matrix
        plot_similarity_matrix(similarity_matrix, plot_folder, action_name, n)

def plot_similarity_matrix(similarity_matrix, plot_folder, action_name, num_samples):
    plt.figure(figsize=(6, 6))
        
    sns.heatmap(similarity_matrix, cmap='viridis', square=True,)
        
    # plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout(pad=2)

    plt.title(f"Similarity Matrix Heatmap for {action_name} ({num_samples} samples)", pad=20)
        
    plt.subplots_adjust(top=0.9, bottom=0.3)
    plt.savefig(os.path.join(plot_folder, f"{action_name}_similarity_matrix.png"))

    plt.close()

###############################   Combined Similarity Matrix   ################################

def get_combined_similarity_matrix(exp_name, action_name, k_train=10, k_test=6, random_seed=42):
    random.seed(random_seed)

    # Scanpaths folder
    train_folder = os.path.join(r"C:\Users\Su\Documents\Codes\Results\EGTEA", exp_name, 'train', action_name)
    test_folder = os.path.join(r"C:\Users\Su\Documents\Codes\Results\EGTEA", exp_name, 'test', action_name)

    # Get scanpath files
    def get_sampled_files(folder, k):
        all_files = []
        for root, _, files in os.walk(folder):
            all_files.extend([os.path.join(root, file) for file in files if file.endswith('.csv')])
        return random.sample(all_files, min(k, len(all_files)))
    
    train_files = get_sampled_files(train_folder, k_train)
    test_files = get_sampled_files(test_folder, k_test)

    all_files = train_files + test_files
    n = len(all_files)
    print(f"Total files: {n} (Train: {len(train_files)}, Test: {len(test_files)})")

    # Initialize similarity matrix
    combined_similarity_matrix = np.zeros((n, n))

    # Calculate pairwise similarities
    for i in range(n):
        scan_path_1 = np.array(pd.read_csv(all_files[i])['Label'])
        for j in range(i, n):   # compute upper triangle only
            scan_path_2 = np.array(pd.read_csv(all_files[j])['Label'])
            similarity, _ = needleman_wunsch_algo(scan_path_1, scan_path_2)
            combined_similarity_matrix[i, j] = similarity
            combined_similarity_matrix[j, i] = similarity  # symmetry   
    
    combined_similarity_matrix = normalize_matrix(combined_similarity_matrix)

    return combined_similarity_matrix, train_files, test_files

def plot_combined_similarity_matrix(similarity_matrix, plot_folder, action_name, train_files, test_files, k_train=10, k_test=6):
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(similarity_matrix, cmap='viridis', square=True, cbar_kws={"shrink": .8})

    # Add diving lines between train and test sets
    ax.axhline(k_train, color='white', linewidth=2)
    ax.axvline(k_train, color='white', linewidth=2)

    # Tick labels
    ticks = [k_train // 2, k_train + k_test // 2]
    labels = ['Train', 'Test']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=12)

    plt.title(f"Combined Similarity Matrix for {action_name} \n ({len(train_files)} train files, {len(test_files)} test files)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"{action_name}_combined_similarity_matrix.png"))
    plt.close()

def get_all_combined_similarity_matrices(exp_name, k_train=10, k_test=6):
    # Get all action names
    exp_folder = os.path.join(r"C:\Users\Su\Documents\Codes\Results\EGTEA", exp_name)
    train_folder = os.path.join(exp_folder, 'train')
    action_names = [os.path.basename(f.path) for f in os.scandir(train_folder) if f.is_dir()]
    print(f"action names: {action_names}")
    
    plot_folder = os.path.join(exp_folder, 'combined_plots_10_6')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for action_name in action_names:
        similarity_matrix, train_files, test_files = get_combined_similarity_matrix(exp_name, action_name, k_train, k_test)
        plot_combined_similarity_matrix(similarity_matrix, plot_folder, action_name, train_files, test_files, k_train, k_test)

def normalize_matrix(matrix):
    """Normalize the matrix to range [0, 1]."""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val == 0:
        return matrix  # Avoid division by zero
    return (matrix - min_val) / (max_val - min_val)

def main(args):
    # Get all subfolder names
    # get_all_similarity_matrix(args.exp_name)
    get_all_combined_similarity_matrices(args.exp_name, k_train=10, k_test=6)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the similarity matrix for all actions in an experiment')
    args = parser.add_argument('--exp_name', type=str, default='experiment_3', help='Experiment name, e.g., experiment_3')
    
    args = parser.parse_args()
    main(args)