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
        sns.swarmplot(ax=ax, data=df, size=individal_point_size, color='k', alpha=0.5)

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
    mse_dict = {
        'GRU-D': np.random.normal(loc=0.3, scale=0.02, size=(10,)),
        'GRU-dt': np.random.normal(loc=0.2, scale=0.05, size=(10,)),
        'ODE-RNN': np.random.normal(loc=0.1, scale=0.01, size=(10,)),
    }

    acc_dict = {
        'GRU-D': np.clip(np.random.normal(loc=0.8, scale=0.02, size=(10,)), 0, 1),
        'GRU-dt': np.clip(np.random.normal(loc=0.8, scale=0.05, size=(10,)), 0, 1),
        'ODE-RNN': np.clip(np.random.normal(loc=0.9, scale=0.02, size=(10,)), 0, 1),
    }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 12
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax = sbplot(ax, method_list, mse_dict)
    ax.set_ylabel(r'Mean Squared Error $\downarrow$', fontsize=18)
    ax.set_xlabel('Methods', fontsize=18)

    ax = fig.add_subplot(1, 2, 2)
    ax = sbplot(ax, method_list, acc_dict)
    ax.set_ylabel(r'Accuracy $\uparrow$', fontsize=18)
    ax.set_xlabel('Methods', fontsize=18)
    fig.tight_layout()
    fig.savefig('toy_data.png')


if __name__ == '__main__':
    test_sbplot()


# # Customizing the bar plot according to the requests

# # Setting up the colors for the bars
# colors = sns.color_palette("husl", 3)

# # Redefining the plot with narrower bars and T-shaped error bars (whiskers)
# plt.figure(figsize=(10, 6))

# bar_width = 0.5  # Narrower bar width
# for i in range(len(data)):
#     plt.bar(labels[i], np.mean(data[i]), width=bar_width, color=colors[i],
#             yerr=np.std(data[i]), capsize=5, error_kw={'capthick': 2, 'elinewidth': 2, 'alpha':0.7})

# # Adding elegant coloring using a Seaborn palette
# sns.set_palette("husl")

# # Drawing horizontal lines for p-value annotations and annotating the p-values
# for i, (pair, p_val) in enumerate(p_values.items()):
#     group1, group2 = [int(x[-1]) - 1 for x in pair.split(' vs ')]
#     y, h, col = y_max + (i+1)*y_step/2, y_step/10, 'k'
#     x1, x2 = group1 - bar_width/2, group2 + bar_width/2

#     plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     plt.text((x1+x2)*.5, y+h, f"p={p_val:.1e}", ha='center', va='bottom', color=col)

# # Finalizing the plot
# plt.title('Sample Data with Pairwise Comparisons')
# plt.ylabel('Value')
# plt.ylim(0, y_max + y_step * (len(p_values) + 1))
# plt.tight_layout()

# # Show plot
# plt.show()
