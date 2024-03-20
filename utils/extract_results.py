import os
import re
import numpy as np
import pandas as pd

# Base directory where the seed folders are located
base_dir = 'Logs/GLEM_train/citeseer/GNN/reg(True)/iter(2)/'

# Define the beta values
beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@100']

# Dictionary to store test MRR values, organized by beta
test_mrr_by_beta = {beta: [] for beta in beta_values}

# Function to extract test MRR from a log file
def extract_test_mrr_from_file(file_path, metric):
    test_mrr_pattern = rf"{metric} result: Train: [\d.]+ ± nan, Valid: [\d.]+ ± nan, Test: ([\d.]+) ± nan"
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(test_mrr_pattern, line)
            if match:
                return float(match.group(1))
    return None
result_dict = {}
# Loop over each seed directory and file
for metric in metrics:
    for seed in range(1, 11):  # Seeds 1 through 10
        for beta in beta_values:
            file_name = f'bert-small_es(2)_beta({beta})_seed{seed}.log'
            file_path = os.path.join(base_dir, f'seed({seed})', file_name)
            if os.path.exists(file_path):
                test_mrr = extract_test_mrr_from_file(file_path, metric)
                if test_mrr is not None:
                    test_mrr_by_beta[beta].append(test_mrr)

    #test_mrr_stats = {beta: {'mean': np.mean(test_mrr_by_beta[beta]), 'std': np.std(test_mrr_by_beta[beta])}
    #               for beta in beta_values}
    test_mrr_stats = {beta: f'{np.mean(test_mrr_by_beta[beta]):.2f} ± {np.std(test_mrr_by_beta[beta]):.2f}' for beta in beta_values}
    result_dict[metric] = test_mrr_stats
    #print(test_mrr_stats)
    test_mrr_by_beta = {beta: [] for beta in beta_values}


df = pd.DataFrame(result_dict)
print(df)
