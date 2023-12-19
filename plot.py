import os
import pdb
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from typing import List

# >>>>>>>>>> rfd
def collect_rfd(n2dir: dict, dump_path: str, start_step: int = 125,
    metrics_fn = lambda col: np.mean(sorted(col, reverse=True)[:10]),
    anu_fn = lambda col_train, col_eval: (col_train - col_eval).mean()
):
    def _load_anu_eval(data_dir: str, start_step):
        metrics_path = os.path.join(data_dir, 'metrics')
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(os.path.join(data_dir, 'metrics'))
        else:
            metrics = {
                'stsb': np.array([0])
            }
        eval_anu = pd.read_csv(os.path.join(data_dir, 'eval_alignment_and_uniformity'))
        train_anu = pd.read_csv(os.path.join(data_dir, 'train_alignment_and_uniformity'))

        return metrics, \
            eval_anu[eval_anu['step'] >= start_step], train_anu[train_anu['step'] >= start_step]
    
    results = {
        'label': [],
        'stsb': [],
        'RFD_u': [],
        'RFD_a': []
    }
    for label, data_dir in n2dir.items():
        metrics, eval_anu, train_anu = _load_anu_eval(data_dir, start_step)
        results['label'].append(label)
        results['stsb'].append(metrics_fn(metrics['stsb']))
        results['RFD_u'].append(anu_fn(train_anu['uniformity'], eval_anu['uniformity']))
        results['RFD_a'].append(anu_fn(train_anu['alignment'], eval_anu['alignment']))

    df = pd.DataFrame(results)
    df.to_csv(dump_path, index=False)
    
def plot_rfd(anu_path: str, dump_path: str = None, annotated: bool = False):

    data = pd.read_csv(anu_path)

    roman_font = fm.FontProperties(family='serif', style='normal', 
        size=18, weight='normal', stretch='normal' # size=26, 
    )

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    markers = {
        'Wiki.dropout': 's',
        'Wiki.token_shuffle': 's',
        'Wiki.token_cutoff': 's',
        'Wiki.NLI': '*',
        'Wiki.STS': '*',
        'Wiki.STS_HT': '*',
        'NLI.dropout': 's',
        'NLI.token_shuffle': 's',
        'NLI.token_cutoff': 's',
        'NLI.NLI': '*',
        'NLI.STS': '*',
        'NLI.STS_HT': '*',
        'NLI.SUP': 'o'
    }
    shifts = {
        'Wiki.dropout': [-0.01, -0.005],
        'Wiki.token_shuffle': [0., -0.01],
        'Wiki.token_cutoff': [0., 0.],
        'Wiki.NLI': [0., 0.],
        'Wiki.STS': [0., 0.],
        'Wiki.STS_HT': [0., 0.],
        'NLI.dropout': [0., 0.],
        'NLI.token_shuffle': [0., 0.],
        'NLI.token_cutoff': [0., 0.],
        'NLI.NLI': [0., 0.],
        'NLI.STS': [0., 0.],
        'NLI.STS_HT': [0., 0.],
        'NLI.SUP': [0., 0.]
    }


    data['stsb'] *= 100
    norm = Normalize(vmax=data["stsb"].max(), vmin=(data["stsb"].min() - 2))
    cm = plt.cm.get_cmap('coolwarm')
    # cm = plt.cm.get_cmap('jet')
    # cm = plt.cm.get_cmap('rainbow')

    marker2type = {
        's': 'trained with unsupervised dataset',
        '*': 'trained with hybrid dataset',
        'o': 'trained with supervised dataset',
    }
    marker2size = {
        's': 300,
        '*': 600,
        'o': 300
    }
    used_marker = set()
    legend_elements = []
    for label, marker in markers.items():
        subset = data[data['label'] == label]
        plt.scatter(subset['RFD_u'] + shifts[label][0], 
                    subset['RFD_a'] + shifts[label][1], 
                    marker=marker, color=cm(norm(subset['stsb'])), s=marker2size[marker])
        
        if marker not in used_marker:
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', markersize=marker2size[marker] / 30, label=marker2type[marker]))   
            used_marker.add(marker)

        if annotated:
            for i, txt in enumerate(subset['label']):
                lb = '{}({:.1f})'.format(txt, subset['stsb'].item())
                plt.annotate(lb, (subset['RFD_u'].iloc[i], subset['RFD_a'].iloc[i]), fontsize=9, xytext=(5,0), textcoords='offset points')

    plt.xlabel('RFD$_\mathrm{u}$', fontproperties=roman_font)
    plt.ylabel('RFD$_\mathrm{a}$', fontproperties=roman_font)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16) 

    plt.grid(True, which='both', linestyle='-', linewidth=1.)

    plt.gca().spines['top'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['right'].set_linewidth(1)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')

    # colorbar
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm) # 
    cbar.ax.tick_params(labelsize=14)  
    # cbar.set_label('Spearman\'s Correlation in STS Benchmark', fontproperties=roman_font)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_color('black')

    plt.tight_layout()
    if dump_path:
        plt.savefig(dump_path)
    else:
        plt.show()
# <<<<<<<<<< rfd

# >>>>>>>>>> alignment and uniformity
def collect_ori_anu(n2paths: dict, dump_path: str):
    results = {
        'label':[],
        'stsb': [],
        'alignment': [],
        'uniformity': []
    }

    results['label'].append('BERT.cls')
    results['stsb'].append(0.3173)
    results['alignment'].append(0.4074)
    results['uniformity'].append(-1.1560)
    
    # stsb starts with 0, anu starts with 125
    for label, paths in n2paths.items():
        metrics_df = pd.read_csv(paths[0])
        anu_df = pd.read_csv(paths[1])
        max_idx = np.argmax(metrics_df['stsb']) + 1
        results['label'].append(label)
        results['stsb'].append(metrics_df.iloc[max_idx]['stsb'])
        results['alignment'].append(anu_df.iloc[max_idx]['alignment'])
        results['uniformity'].append(anu_df.iloc[max_idx]['uniformity'])

    df = pd.DataFrame(results)
    df.to_csv(dump_path, index=False)

def plot_anu(anu_path: str, dump_path: str = None, annotated: bool = False):
    
    data = pd.read_csv(anu_path)

    roman_font = fm.FontProperties(family='serif', style='normal', 
        size=26, weight='normal', stretch='normal'
    )

    plt.figure(figsize=(10, 4))
    sns.set(style="whitegrid")
    markers = {
        'BERT.cls': '^',
        'nli.sup': 'o',
        'nli.dropout': 's',
        'nli.token_shuffle': 's',
        'nli.token_cutoff': 's',
        'wiki.dropout': 's',
        'wiki.token_shuffle': 's',
        'wiki.token_cutoff': 's'
    }

    shifts = {
        'BERT.cls': [0., 0.],
        'nli.sup': [0., 0.],
        'nli.dropout': [0.02, -0.02],
        'nli.token_shuffle': [0., -0.03],
        'nli.token_cutoff': [0., 0.],
        'wiki.dropout': [0., -0.005],
        'wiki.token_shuffle': [0., 0.],
        'wiki.token_cutoff': [0., 0.]
    }

    data['stsb'] *= 100
    norm = Normalize(vmin=50.0, vmax=data["stsb"].max())
    
    cm = plt.cm.get_cmap('viridis')

    marker2type = {
        '^': "pretrained",
        'o': 'trained with supervised dataset',
        's': 'trained with unsupervised dataset'
    }
    used_marker = set()
    legend_elements = []
    for label, marker in markers.items():
        subset = data[data['label'] == label]
        plt.scatter(subset['uniformity'] + shifts[label][0], 
                    subset['alignment'] + shifts[label][1], 
                    marker=marker, color=cm(norm(subset['stsb'])), s=400)
        
        if marker not in used_marker:
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', markersize=10, label=marker2type[marker]))   
            used_marker.add(marker)

        if annotated:
            for i, txt in enumerate(subset['label']):
                plt.annotate(txt, (subset['uniformity'].iloc[i], subset['alignment'].iloc[i]), fontsize=9, xytext=(5,0), textcoords='offset points')

    plt.xlabel('$\mathcal{L}_\mathrm{uniform}$', fontproperties=roman_font)
    plt.ylabel('$\mathcal{L}_\mathrm{align}$', fontproperties=roman_font)
    plt.xticks(fontsize=16)
    plt.yticks(np.arange(0.24, 0.45, 0.05), fontsize=16)

    plt.grid(True, which='both', linestyle='-', linewidth=1.)

    plt.gca().spines['top'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['right'].set_linewidth(1)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_color('black')

    # colorbar
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm) # 
    cbar.ax.tick_params(labelsize=14)  
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_color('black')

    plt.tight_layout()
    if dump_path:
        plt.savefig(dump_path)
    else:
        plt.show()
# <<<<<<<<<< alignment and uniformity

# >>>>>>>>>> training process
def collect_anu_process(n2dir: str, dump_path: str):
    results = {
        'label': [],
        'step': [],
        'eval.uniformity': [],
        'eval.alignment': [],
        'train.uniformity': [],
        'train.alignment': []
    }
    for label, dir_path in n2dir.items():
        eval_df = pd.read_csv(os.path.join(dir_path, 'eval_alignment_and_uniformity'))
        train_df = pd.read_csv(os.path.join(dir_path, 'train_alignment_and_uniformity'))
        for eval_row, train_row in zip(eval_df.iloc, train_df.iloc):
            assert eval_row['step'].item() == train_row['step'].item()
            results['label'].append(label)
            results['step'].append(eval_row['step'].item())
            results['eval.uniformity'].append(eval_row['uniformity'])
            results['eval.alignment'].append(eval_row['alignment'])
            results['train.uniformity'].append(train_row['uniformity'])
            results['train.alignment'].append(train_row['alignment'])
    
    df = pd.DataFrame(results)
    df.to_csv(dump_path, index=False)

def plot_anu_process(anup_path: str, dump_path: str = None, 
    labels: list = ['NLI.SUP', 'NLI.dropout', 'NLI.token_cutoff', 'NLI.token_shuffle',
                    'Wiki.dropout', 'Wiki.token_cutoff', 'Wiki.token_shuffle']):

    data = pd.read_csv(anup_path)

    properties = ['alignment', 'uniformity']

    fig, axes = plt.subplots(nrows=2, ncols=len(labels), figsize=(2.5 * len(labels), 4))
    # fig, axes = plt.subplots(nrows=2, ncols=len(labels), figsize=(10,6))
    sns.set(style="whitegrid")

    x_lim = (0, 4000)

    roman_font = fm.FontProperties(family='serif', style='normal', 
        weight='normal', stretch='normal', size=18, 
    )

    for row, prop in enumerate(properties):
        for col, label in enumerate(labels):
            ax = axes[row, col]
            subset_data = data[data['label'] == label]
            ax.plot(subset_data['step'], subset_data[f'train.{prop}'], label=f'train', linestyle='-')
            ax.plot(subset_data['step'], subset_data[f'eval.{prop}'], label=f'eval', linestyle='--')
            
            ax.grid(True, linestyle='--', alpha=0.5)
            # ax.set_xlim(x_lim)

            if row == 0:
                ax.set_ylim(0.15, 0.60)
            
            if row == 1:
                ax.set_ylim(-3.25, -1.25)
            
            if col > 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis='y', labelsize=16)
            
            if row == 0:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', labelsize=16)
            
            if col == 0:
                ytitle = '$\mathcal{L}_\mathrm{align}$' if prop == 'alignment' else '$\mathcal{L}_\mathrm{uniform}$'
                ax.set_ylabel(ytitle, fontsize=26)
            
            if row == 1:
                ax.set_xlabel('step', fontproperties=roman_font)

            if row == 0:
                ax.set_title(label, fontproperties=roman_font)
            

            if col == 0 and row == 0:
                ax.legend(
                    prop=fm.FontProperties(family='serif', style='normal', 
                        weight='normal', stretch='normal', size=16
                    )
                )
            

    plt.tight_layout()

    if dump_path:
        plt.savefig(dump_path)
    else:
        plt.show()
# <<<<<<<<<< training process

# >>>>>>>>>> further study
def plot_m1_m2(m1_path: str, m2_path: str,
                      dump_path: str = None):
    m1_data = pd.read_csv(m1_path)
    m2_data = pd.read_csv(m2_path)

    # Determine the unified y-axis limits
    y_min = min(m1_data['avg'].min(), m2_data['avg'].min()) - 0.1
    y_max = max(m1_data['avg'].max(), m2_data['avg'].max()) + 0.1

    roman_font = fm.FontProperties(family='serif', style='normal', 
        size=18, weight='normal', stretch='normal' # size=26, 
    )

    # Function to plot the line graphs with further updated requirements
    def plot_final_graph(x, y, xlabel, ylabel):
        plt.plot(x, y, marker='o', markersize=4)
        plt.xlabel(xlabel, fontproperties=roman_font)
        plt.ylabel(ylabel, fontproperties=roman_font)
        plt.grid(True)

    plt.figure(figsize=(10, 4)) # Adjusted figure size for a more compact layout

    ax = plt.subplot(1, 2, 1)
    plt.ylim(y_min, y_max)
    plot_final_graph(m1_data['margin_1'], m1_data['avg'], 
        '$m_1$', 'Avg.')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_yticks(np.arange(78., 79.9, 0.5))

    ax = plt.subplot(1, 2, 2)
    plt.ylim(y_min, y_max)
    plot_final_graph(m2_data['margin_2'], m2_data['avg'], 
        '$m_2$', 'Avg')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_yticks(np.arange(78., 79.9, 0.5))
    ax.set_yticklabels([])
    ax.set_ylabel('')

    plt.tight_layout(w_pad=0.5) # Reduced padding between subplots

    if dump_path:
        plt.savefig(dump_path)
    else:
        plt.show()

def plot_number(number_path: str, dump_path: str = None):
    number_data = pd.read_csv(number_path)

    fig = plt.figure(figsize=(10, 4)) # Adjusted figure size for a more compact layout

    sns.set_theme(style="whitegrid")

    sns.lineplot(x="number", y="avg", color='#FF852F',
            #  hue="region", style="event",
             data=number_data)

    roman_font = fm.FontProperties(family='serif', style='normal', 
        size=18, weight='normal', stretch='normal' # size=26, 
    )
    plt.xlabel('The number of LLM-generated data', fontproperties=roman_font)
    plt.ylabel('Avg.', fontproperties=roman_font)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16) 

    for _, spine in fig.gca().spines.items():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.9)
    
    plt.tight_layout()
    if dump_path:
        plt.savefig(dump_path)
    else:
        plt.show()
# <<<<<<<<<< further study

if __name__ == '__main__':
    
    # >>>>>>>>>> plot rfd
    collect_rfd({
        'Wiki.dropout': r'data\wiki\dropout',
        'Wiki.token_shuffle': r'data\wiki\token_shuffle',
        'Wiki.token_cutoff': r'data\wiki\token_cutoff',
        'Wiki.NLI': r'data\wiki\nli',
        'Wiki.STS': r'data\wiki\sts',
        'Wiki.STS_HT': r'data\wiki\sts_ht',
        'NLI.dropout': r'data\nli\dropout',
        'NLI.token_shuffle': r'data\nli\token_shuffle',
        'NLI.token_cutoff': r'data\nli\token_cutoff',
        'NLI.NLI': r'data\nli\nli',
        'NLI.STS': r'data\nli\sts',
        'NLI.STS_HT': r'data\nli\sts_ht',
        'NLI.SUP': r'data\nli\sup'
    }, os.path.join(r'figure\rfd', 'rfd.csv'),
    metrics_fn=lambda col: np.mean(sorted(col, reverse=True)[:5]))

    plot_rfd(os.path.join(r'figure\rfd', 'rfd.csv'), os.path.join(r'figure\rfd', 'rfd.pdf'))
    # <<<<<<<<<< plot rfd

    # >>>>>>>>>> plot alignment anu uniformity
    collect_ori_anu({
        'nli.sup': [r'data\nli\sup\metrics', r'data\nli\sup\eval_alignment_and_uniformity'],
        'nli.dropout': [r'data\nli\dropout\metrics', r'data\nli\dropout\eval_alignment_and_uniformity'],
        'nli.token_shuffle': [r'data\nli\token_shuffle\metrics', r'data\nli\token_shuffle\eval_alignment_and_uniformity'],
        'nli.token_cutoff': [r'data\nli\token_cutoff\metrics', r'data\nli\token_cutoff\eval_alignment_and_uniformity'],
        'wiki.dropout': [r'data\wiki\dropout\metrics', r'data\wiki\dropout\eval_alignment_and_uniformity'],
        'wiki.token_shuffle': [r'data\wiki\token_shuffle\metrics', r'data\wiki\token_shuffle\eval_alignment_and_uniformity'],
        'wiki.token_cutoff': [r'data\wiki\token_cutoff\metrics', r'data\wiki\token_cutoff\eval_alignment_and_uniformity']
    }, os.path.join(r'figure\anu', 'anu.csv'))

    plot_anu(os.path.join(r'figure\anu', 'anu.csv'), os.path.join(r'figure\anu', 'anu.pdf'), annotated=False)
    # <<<<<<<<<< plot alignment anu uniformity

    # >>>>>>>>>> plot training process
    collect_anu_process({
        'NLI.SUP': r'data\nli\sup',
        'NLI.dropout': r'data\nli\dropout',
        'NLI.token_shuffle': r'data\nli\token_shuffle',
        'NLI.token_cutoff': r'data\nli\token_cutoff',
        'Wiki.dropout': r'data\wiki\dropout',
        'Wiki.token_shuffle': r'data\wiki\token_shuffle',
        'Wiki.token_cutoff': r'data\wiki\token_cutoff'
    }, os.path.join(r'figure\process', 'anu_process.csv'))

    plot_anu_process(os.path.join(r'figure\process', 'anu_process.csv'), 
                     os.path.join(r'figure\process', 'anu_process.pdf'),  
                     labels=['NLI.SUP', 'NLI.dropout', 'Wiki.dropout']) 
    # <<<<<<<<<< plot training process

    # >>>>>>>>>> plot further study
    plot_m1_m2(r'data\further\m1.csv', r'data\further\m2.csv', r'figure\further\m1_m2.pdf')

    plot_number(r'data\further\number_multiple.csv', r'figure\further\number.pdf')
    # <<<<<<<<<< plot further study