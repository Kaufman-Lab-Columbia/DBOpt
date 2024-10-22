#single_dim_plots

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Any, List, Tuple
import numpy.typing as npt
import matplotlib.figure
def single_dim_plot(
    sweep_arr: npt.NDArray[Any], 
    sel_params: Dict[str, float]
) -> matplotlib.figure:
    """
    Creates a scatter plot with x-axis cooresponing to the single 
    parameter being optimized and z-axis corresponding to the DBCV 
    score, colored to the DBCV score.

    Args:
        sweep_arr ():


        sel_params ():

    Returns:
        MatPlotLib figure with a single plot.

    """
    y_names = sweep_arr[0]
    x_vals, y_vals = output_vals = data_format_single_dim(sweep_arr, method='single')   

    fig, ax = plt.subplots()

    color_map = matplotlib.cm.rainbow
    cm_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    
    ax.scatter(x_vals, y_vals, 
        c=y_vals, cmap='rainbow', vmin=-1, vmax=1, s=10)

    for i in range(len(x_vals) - 1):
        ax.plot([x_vals[i], x_vals[i+1]], [y_vals[i], y_vals[i + 1]], 
            color=color_map(cm_norm((y_vals[i] + y_vals[i + 1]) / 2)))
    
    if sel_params:
        ax.scatter(sel_params[y_names[0]], np.max(y_vals), 
            marker='X', color='white', edgecolor='k', zorder=2, s=50)

    ax.set_ylim(-1,1)        
    ax.set_xlabel(y_names[0])
    ax.set_ylabel('DBCV Score')
    ax.set_title('DBOpt', fontsize=18)
    
    plt.show()
    return fig
    
def dual_single_dim_plot(
    sweep_arr: npt.NDArray[Any], 
    sel_params: Dict[str,float]
    ) -> matplotlib.figure:

    """
    Creates two scatter plots with the x-axis cooresponing to the 
    single parameter being optimized and z-axis corresponding to the 
    DBCV score, colored to the DBCV score. The two plots correspond to 
    HDBSCAN cluster_selection_method 'eom' and 'leaf'.

    Args:
        scoresweep ():

        chosen_parameters ():


    Returns:
        MatPlotLib figure with two plots, corresponding to 'eom' and 'leaf'
        for HDBSCAN cluster_selection_method.

    """

    output_vals = data_format_single_dim(sweep_arr, method='dual')
    
    x_vals = output_vals[0]
    y_vals = output_vals[1]
    min_vals = output_vals[2]
    max_vals = output_vals[3]  

    dy = 2        
    dx = max_vals - min_vals
    aspect_scaling = dx / dy

    dim_names=[]
    if sel_params['cluster_selection_method'] == 'eom':
        sel_params['cluster_selection_method'] = None
        sel_params_0 = sel_params
        sel_params_1 = None
        for i,n in enumerate(sel_params_0):
            dim_names.append(n)
    elif sel_params['cluster_selection_method'] == 'leaf':
        sel_params['cluster_selection_method'] = None
        sel_params_0 = None
        sel_params_1 = sel_params
        for i,n in enumerate(sel_params_1):
            dim_names.append(n)
    
    fig, ax = plt.subplots(1,2,figsize = (9,5))

    color_map = matplotlib.cm.rainbow
    cm_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    

    for i in range(2):

        ax[i].scatter(x_vals[i], y_vals[i], 
        c=y_vals[i], cmap='rainbow', vmin=-1, vmax=1, s=10)

        for j in range(len(x_vals[i]) - 1):
            ax[i].plot([x_vals[i][j], x_vals[i][j + 1]], 
                [y_vals[i][j], y_vals[i][j + 1]], 
                color=color_map(cm_norm((y_vals[i][j] + y_vals[i][j + 1]) / 2)))

        ax[i].set_ylim(-1,1)
        ax[i].set_xlim(min_vals,max_vals)
        ax[i].set_xlabel(dim_names[0])
        ax[i].set_ylabel('DBCV Score')
        ax[i].set_aspect(aspect_scaling)


    if sel_params_0 is not None:
        ax[0].scatter(sel_params_0[dim_names[0]], np.max(y_vals[0]), 
            marker='X', color='white', edgecolor='k', zorder=2, s=50)

    if sel_params_1 is not None:
        ax[1].scatter(sel_params_1[dim_names[0]], np.max(y_vals[1]), 
            marker='X', color='white', edgecolor='k', zorder=2, s=50)

    ax[0].set_title('DBOpt, Method: eom', fontsize=18)       
    ax[1].set_title('DBOpt, Method: leaf', fontsize=18)
    
    # ax[0].set_ylim(-1,1)
    # ax[0].set_xlim(min_vals,max_vals)
    # ax[0].set_xlabel(dim_names[0])
    # ax[0].set_ylabel('DBCV Score')
    # ax[0].set_title('DBOpt, Method: eom', fontsize=18)
    # ax[0].set_aspect(aspect_scaling)
    # ax[1].set_ylim(-1,1)
    # ax[1].set_xlim(min_vals,max_vals)
    # ax[1].set_xlabel(dim_names[0])
    # ax[1].set_ylabel('DBCV Score')
    # ax[1].set_title('DBOpt, Method: leaf', fontsize=18)
    # ax[1].set_aspect(aspect_scaling)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def data_format_single_dim(
    sweep_arr: npt.NDArray[Any], 
    method: str = 'single'
) -> Tuple[
    List[npt.NDArray[np.float_]],
    List[npt.NDArray[np.float_]], 
    float,
    float
]:

    """
    Sorts through optimization data to find values of x and y and splits
    the data when HDBSCAN cluster_selection_method was optimized.

    Args:
        sweep_arr ():

        method (str):
            Indicates whether HDBSCAN cluster_selection_method(eom, 
            leaf) need to be divided ('dual') or not ('single') to 
            create the plot(s).

    Returns:

    """
    if method == 'single':
        all_vals = sweep_arr[1:]
        x_vals = all_vals[:,0][all_vals[:,0].argsort()]
        y_vals = all_vals[:,-1][all_vals[:,0].argsort()]

        return x_vals, y_vals


    elif method == 'dual':

        index_method = np.where(sweep_arr[0] == 'cluster_selection_method')[0][0]

        score_sweep_eom =  np.vstack((sweep_arr[0], 
            sweep_arr[1:][sweep_arr[1:][:,index_method] == 'eom']))

        score_arr_eom = np.delete(score_sweep_eom, index_method, 1)

        score_sweep_leaf =  np.vstack((sweep_arr[0], 
            sweep_arr[1:][sweep_arr[1:][:,index_method] == 'leaf']))

        score_arr_leaf = np.delete(score_sweep_leaf, index_method, 1)

        all_vals = score_arr_eom[1:]
        all_vals_2 = score_arr_leaf[1:]

        min_vals = min(np.min(all_vals[:,0]), np.min(all_vals_2[:,0]))
        max_vals = max(np.max(all_vals[:,0]), np.max(all_vals_2[:,0]))        

        x_vals = all_vals[:,0][all_vals[:,0].argsort()]
        y_vals = all_vals[:,-1][all_vals[:,0].argsort()]

        x_vals_2 = all_vals_2[:,0][all_vals_2[:,0].argsort()]
        y_vals_2 = all_vals_2[:,-1][all_vals_2[:,0].argsort()]

        return [x_vals, x_vals_2], [y_vals, y_vals_2], min_vals, max_vals