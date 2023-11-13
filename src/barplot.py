from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def sbplot(ax: plt.Axes,
           method_list: List[str],
           data_dict: Dict[str, np.array],
           show_individual_points: bool = True,
           individal_point_size: float = 8,
           labelsize: int = 16) -> plt.Axes:
    '''
    Statistical Bar Plot.
    Blot the bars and display the significance levels.
    '''

    n_bars = len(method_list)
    colors = sns.color_palette('muted', n_bars)
    data_arr = np.array([data_dict[k] for k in method_list])

    if show_individual_points:
        df = pd.DataFrame(data_arr.T, columns=method_list)
        sns.swarmplot(ax=ax, data=df, size=individal_point_size, palette=colors, alpha=0.5)

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    for i in range(n_bars):
        ax.bar(method_list[i],
               np.mean(data_dict[method_list[i]]),
               yerr=np.std(data_dict[method_list[i]]),
               color='none',
               edgecolor=colors[i],
               width=0.4,
               linewidth=3,
               capsize=8,
               error_kw={'capthick': 3, 'elinewidth': 3, 'alpha': 1})

    return ax


def test_sbplot():
    method_list = ['GRU-D', 'GRU-dt', 'ODE-RNN']

    auroc_dict = {
        'GRU-D': np.clip(np.random.normal(loc=0.92, scale=0.02, size=(10,)), 0, 1),
        'GRU-dt': np.clip(np.random.normal(loc=0.75, scale=0.03, size=(10,)), 0, 1),
        'ODE-RNN': np.clip(np.random.normal(loc=0.98, scale=0.02, size=(10,)), 0, 1),
    }

    acc_dict = {
        'GRU-D': np.clip(np.random.normal(loc=0.89, scale=0.02, size=(10,)), 0, 1),
        'GRU-dt': np.clip(np.random.normal(loc=0.81, scale=0.03, size=(10,)), 0, 1),
        'ODE-RNN': np.clip(np.random.normal(loc=0.94, scale=0.02, size=(10,)), 0, 1),
    }

    f1_dict = {
        'GRU-D': np.clip(np.random.normal(loc=0.92, scale=0.02, size=(10,)), 0, 1),
        'GRU-dt': np.clip(np.random.normal(loc=0.87, scale=0.03, size=(10,)), 0, 1),
        'ODE-RNN': np.clip(np.random.normal(loc=0.95, scale=0.02, size=(10,)), 0, 1),
    }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 12
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 3, 1)
    ax = sbplot(ax, method_list, acc_dict)
    ax.set_ylabel(r'AUROC $\uparrow$', fontsize=18)
    ax.set_xlabel('Methods', fontsize=18)

    ax = fig.add_subplot(1, 3, 2)
    ax = sbplot(ax, method_list, acc_dict)
    ax.set_ylabel(r'Accuracy $\uparrow$', fontsize=18)
    ax.set_xlabel('Methods', fontsize=18)

    ax = fig.add_subplot(1, 3, 3)
    ax = sbplot(ax, method_list, f1_dict)
    ax.set_ylabel(r'F1 Score $\uparrow$', fontsize=18)
    ax.set_xlabel('Methods', fontsize=18)

    fig.tight_layout()
    fig.savefig('../assets/sbplot_example.png')


if __name__ == '__main__':
    test_sbplot()

