#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file


import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

glacier_list = ['RGI2000-v7.0-G-11-01182', 'RGI2000-v7.0-G-11-01457',
                'RGI2000-v7.0-G-11-01706', 'RGI2000-v7.0-G-11-02596']

hyperparameter = 'inflation'
for glacier_name in glacier_list:
    experiment_dir = os.path.join('../../Experiments/', glacier_name,
                                  'inflation_robustness')
    experiment_dirs = os.listdir(experiment_dir)

    inflation_data = {
        'inflation': [],
        'seed': [],
        'final_mean_ela': [],
        'final_mean_grad_abl': [],
        'final_mean_grad_acc': [],
        'final_std_ela': [],
        'final_std_grad_abl': [],
        'final_std_grad_acc': [],
        # 'num_bins': []

    }
    for exp_dir in experiment_dirs:
        if os.path.isdir(os.path.join(experiment_dir, exp_dir)):

            files = os.listdir(os.path.join(experiment_dir, exp_dir))

            result_files = [file for file in files if file.startswith("result")]
            if len(result_files) == 0:
                continue
            else:
                result_file = result_files[0]

            result = json.load(
                open(os.path.join(experiment_dir, exp_dir, result_file), 'r'))
            print(exp_dir, result_file)
            inflation_data['seed'].append(result['seed'])
            inflation_data['inflation'].append(result['inflation'])
            # inflation_data['num_bins'].append(result['num_bins'])
            inflation_data['final_mean_ela'].append(result['final_mean'][0])
            inflation_data['final_mean_grad_abl'].append(result['final_mean'][1])
            inflation_data['final_mean_grad_acc'].append(result['final_mean'][2])
            inflation_data['final_std_ela'].append(result['final_std'][0])
            inflation_data['final_std_grad_abl'].append(result['final_std'][1])
            inflation_data['final_std_grad_acc'].append(result['final_std'][2])

    df = pd.DataFrame(inflation_data)

    filter_values = {
        'Equilibrium_Line_Altitude': (1000, 4000, 'final_mean_ela', 'final_std_ela'),
        'Ablation_Gradient': (
            -100, 100, 'final_mean_grad_abl', 'final_std_grad_abl'),
        'Accumulation_Gradient': (
            -100, 100, 'final_mean_grad_acc', 'final_std_grad_acc')
    }
    filtered_df = df[
        (df['final_mean_ela'] >= filter_values['Equilibrium_Line_Altitude'][0]) &
        (df['final_mean_ela'] <= filter_values['Equilibrium_Line_Altitude'][1]) &
        (df['final_mean_grad_abl'] >= filter_values['Ablation_Gradient'][0]) &
        (df['final_mean_grad_abl'] <= filter_values['Ablation_Gradient'][1]) &
        (df['final_mean_grad_acc'] >= filter_values['Accumulation_Gradient'][0]) &
        (df['final_mean_grad_acc'] <= filter_values['Accumulation_Gradient'][1])
        ]

    mean_ensemble_spread = filtered_df.groupby(hyperparameter).mean().reset_index()
    std_ensemble_spread = filtered_df.groupby(hyperparameter).std().reset_index()

    import matplotlib.pyplot as plt

    # Create two figures, each with 3 subplots (one for each parameter)
    fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(15, 5))
    fig_spread, axes_spread = plt.subplots(1, 3, figsize=(15, 5))

    for i, parameter in enumerate(filter_values.keys()):
        # Scatter plot for each parameter
        axes_scatter[i].scatter(filtered_df[hyperparameter],
                                filtered_df[filter_values[parameter][2]], marker='o')
        axes_scatter[i].set_xlabel(hyperparameter)
        axes_scatter[i].set_ylabel(f"{parameter} Value")
        axes_scatter[i].set_title(
            f"Final Mean {parameter} of multiple runs\n {glacier_name}")

        # Line plot for mean and standard deviation for each parameter
        axes_spread[i].plot(mean_ensemble_spread[hyperparameter],
                            mean_ensemble_spread[filter_values[parameter][3]],
                            marker='o', linestyle='-',
                            label='Mean std of individual calibration runs')
        axes_spread[i].plot(mean_ensemble_spread[hyperparameter],
                            std_ensemble_spread[filter_values[parameter][2]],
                            marker='o', linestyle='-',
                            label='Std of mean of multiple calibration runs')
        axes_spread[i].set_xlabel(hyperparameter)
        axes_spread[i].set_ylabel(f"{parameter} Standard Deviation")
        axes_spread[i].set_title(f"{parameter} spread of {glacier_name}")
        axes_spread[i].legend()

    # Adjust layout and save both figures
    fig_scatter.tight_layout()
    fig_spread.tight_layout()

    fig_scatter.savefig(
        f'{glacier_name}_compare_{hyperparameter}_scatter.png', dpi=300)
    fig_spread.savefig(
        f'{glacier_name}_compare_{hyperparameter}_spread.png', dpi=300)
