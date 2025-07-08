import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any

def spider_plot(data: pd.DataFrame, labels: List[str], title: str, save_path=None):
    """
    data: pd.DataFrame
        Format of the dataframe should be as follows:
        {
            categories: List[str],
            values: List[List[float]],
            stddev: List[List[float]]
        } 
    labels: List[str]
        List of labels for each value in the data. 
        The length of the list should be equal to the number of rows in the data.
    title: str
        Title of the plot
    save_path: str 
        Path to save the plot. If None, the plot is not saved.
    """
    # number of variables
    num_vars = len(data['categories'])
    # split the circle into even parts
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # close the plot
    ax = plt.subplot(111, polar=True)
    for i in range(len(data['values'])):
        values = data['values'][i]
        ub_values = [values[j] + data['stddev'][i][j] for j in range(len(values))]
        lb_values = [values[j] - data['stddev'][i][j] for j in range(len(values))]
        # values = data.loc[i].drop('categories').values.flatten().tolist()
        # values += values[:1]
        # ax.fill(angles, values, alpha=0.15, label=labels[i])
        ax.fill_between(angles+[0], lb_values+lb_values[:1], ub_values+ub_values[:1], alpha=0.3, label=labels[i])
        ax.plot(angles+[0], values+values[:1])
    # Draw one axe per variable + add labels + space it farther
    plt.xticks(angles, data['categories'], color='black', size=7)
    # pad the xticks
    ax.tick_params(axis='x', pad=15)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="black", size=7)
    plt.ylim(0,1)
    # Add a title
    plt.title(title, size=11, color="black", y=1.15)
    # Add legend
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4))
    # Save the plot
    # set tight
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)


