from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def sbplot(ax: plt.Axes,
           method_list: List[str],
           data_dict: Dict[str, np.array],
           pvals_dict: Dict[str, float] = None,
           show_individual_points: bool = True,
           color_palette: str = 'muted',
           individal_point_size: float = 8,
           labelsize: int = 16,
           y_buffer_ratio: float = 0.05,
           text_buffer_ratio: float = 1e-2,
           ymin: float = None,
           ymax: float = None) -> plt.Axes:
    '''
    Statistical Bar Plot.
    Blot the bars and display the significance levels.

    Arguments
    ---------
    - `method_list`:
        A list of method strings, each corresponding to a bar in the plot.
    - `data_dict`:
        A dict, with each key being a method string and the value being a 1D numpy array.
    - `pvals_dict`:
        A dict with the following format:
        {'$method1 vs $method2': $p-value, ...}
        The key shall use ' vs ' to connect two method strings,
        both of which mentioned in `method_list`.
    '''

    n_bars = len(method_list)
    colors = sns.color_palette(color_palette, n_bars)
    data_arr = np.array([data_dict[k] for k in method_list])

    single_point = len(data_arr.T) == 1

    # Plot the individual points.
    # Seaborn works the best with dataframes, hence the conversion.
    if show_individual_points and not single_point:
        df = pd.DataFrame(data_arr.T, columns=method_list)
        sns.swarmplot(ax=ax, data=df, size=individal_point_size, color='k', marker="$\circ$")

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    # Plot the bars one by one.
    for i in range(n_bars):
        y_error = np.std(data_dict[method_list[i]]) if not single_point else None
        ax.bar(method_list[i],
               np.mean(data_dict[method_list[i]]),
               yerr=y_error,
               edgecolor=(0, 0, 0, 1),
               facecolor=(*colors[i][:3], 0.5), # set facecolor-specific alpha.
               width=0.5,
               linewidth=1.5,
               capsize=10,
               error_kw={'capthick': 3, 'elinewidth': 3, 'alpha': 1})

    # Compute the range of y for display.
    if not single_point:
        ymax_baseline = max(np.max(data_arr),
                            np.max(np.mean(data_arr, axis=1) + np.std(data_arr, axis=1))) if ymax is None else ymax
        ymin_baseline = min(np.min(data_arr),
                            np.min(np.mean(data_arr, axis=1) - np.std(data_arr, axis=1))) if ymin is None else ymin
    else:
        ymax_baseline = np.max(data_arr) if ymax is None else ymax
        ymin_baseline = np.min(data_arr) if ymin is None else ymin
    height_range = ymax_baseline - ymin_baseline
    y_buffer = y_buffer_ratio * height_range
    text_buffer = text_buffer_ratio * height_range
    if ymin is None:
        ymin = np.min(data_arr) - y_buffer

    # Plot the indicators of p-values.
    ymax_baseline += y_buffer

    if pvals_dict is not None:
        for method_pair in pvals_dict.keys():
            method1, method2 = method_pair.split(' vs ')

            # Each method string shall appear in `method_list` once and only once.
            assert (np.array(method_list) == method1).sum() == 1
            assert (np.array(method_list) == method2).sum() == 1
            method1_idx = np.where(np.array(method_list) == method1)[0].item()
            method2_idx = np.where(np.array(method_list) == method2)[0].item()

            # Plot the horizontal lines, p-values, and asterisks.
            pval = pvals_dict[method_pair]
            asterisks = pval_to_asterisk(pval=pval)
            line_x = [method1_idx, method2_idx]
            line_y = [ymax_baseline, ymax_baseline]
            ax.plot(line_x, line_y, 'k', linewidth=1)
            ax.text(np.mean(line_x), ymax_baseline + text_buffer, asterisks, horizontalalignment='center')

            ymax_baseline += y_buffer

    ax.set_ylim([ymin, ymax_baseline])

    return ax

def pval_to_asterisk(pval: float) -> str:
    '''
    Assign the proper number of asterisks to the given p-value.
    '''
    assert pval >= 0, '`pval_to_asterisk`: p-value has to be >= 0.'
    if pval < 0.0001:
        return '* * * *'
    elif pval < 0.001:
        return '* * *'
    elif pval < 0.01:
        return '* *'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

def test_sbplot():
    np.random.seed(1)

    method_list = ['method_1', 'method_2', 'method_3']

    auroc_data_dict = {
        'method_1': np.clip(np.random.normal(loc=0.92, scale=0.04, size=(10,)), 0, 1),
        'method_2': np.clip(np.random.normal(loc=0.75, scale=0.03, size=(10,)), 0, 1),
        'method_3': np.clip(np.random.normal(loc=0.98, scale=0.02, size=(10,)), 0, 1),
    }
    auroc_pvals_dict = {
        'method_1 vs method_3': 0.01,
        'method_2 vs method_3': 5e-5,
    }

    acc_data_dict = {
        'method_1': np.clip(np.random.normal(loc=0.89, scale=0.05, size=(10,)), 0, 1),
        'method_2': np.clip(np.random.normal(loc=0.81, scale=0.04, size=(10,)), 0, 1),
        'method_3': np.clip(np.random.normal(loc=0.94, scale=0.03, size=(10,)), 0, 1),
    }
    acc_pvals_dict = {
        'method_1 vs method_3': 0.01,
        'method_2 vs method_3': 0.001,
    }

    f1_data_dict = {
        'method_1': np.clip(np.random.normal(loc=0.92, scale=0.04, size=(10,)), 0, 1),
        'method_2': np.clip(np.random.normal(loc=0.87, scale=0.03, size=(10,)), 0, 1),
        'method_3': np.clip(np.random.normal(loc=0.95, scale=0.02, size=(10,)), 0, 1),
    }
    f1_pvals_dict = {
        'method_1 vs method_3': 0.01,
        'method_2 vs method_3': 3e-4,
    }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 12
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 3, 1)
    ax = sbplot(ax=ax, method_list=method_list, data_dict=auroc_data_dict, pvals_dict=auroc_pvals_dict, ymin=0)
    ax.set_ylabel('AUROC', fontsize=18)

    ax = fig.add_subplot(1, 3, 2)
    ax = sbplot(ax=ax, method_list=method_list, data_dict=acc_data_dict, pvals_dict=acc_pvals_dict, ymin=0)
    ax.set_ylabel('Accuracy', fontsize=18)

    ax = fig.add_subplot(1, 3, 3)
    ax = sbplot(ax=ax, method_list=method_list, data_dict=f1_data_dict, pvals_dict=f1_pvals_dict, ymin=0)
    ax.set_ylabel('F1 Score', fontsize=18)

    fig.tight_layout(pad=1)
    fig.savefig('../assets/sbplot_example.png')


if __name__ == '__main__':
    test_sbplot()

