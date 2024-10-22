#contour_plots

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


from typing import Dict, Any, Tuple, List
import numpy.typing as npt
import matplotlib.figure

def contour_plot(
    sweep_arr: npt.NDArray[Any], 
    sel_params: Dict[str,float]
) -> matplotlib.figure:

    """
    Creates a contour plot for scored parameters. Countours are 
    colored to the DBCV scores of parameter combinations.

    Args:
        sweep_arr (npt.NDArray[]):
            Array with optimized parameter values and corresponding 
            DBCV scores.
        sel_params (typing.Dict[str,float]):
            Dictionary with parameter names (keys) and values 
            corresponding to the selected optimal parameters

    Returns:

    """
    output_data = data_format_contour(sweep_arr, method = 'single')
    x = output_data[0]
    y = output_data[1]
    xs = output_data[2]
    ys = output_data[3]
    z_resampled = output_data[4]
    contour_step_size  =  output_data[5]
    rescale_x = output_data[6]
    rescale_y = output_data[7]

    param_space = (np.min(x), np.max(x), np.min(y), np.max(y))
    
    x_param = sweep_arr[0][0]
    y_param = sweep_arr[0][1]

    x_sel = sel_params[x_param]
    y_sel = sel_params[y_param]

    if rescale_x == True:
        x_sel = x_sel * 100
    if rescale_y == True:
        y_sel = y_sel * 100
    
    fig, ax = plt.subplots()

    cmap = matplotlib.colormaps["rainbow"].copy()
    

    contour = ax.contourf(xs, ys, z_resampled, cmap=cmap, 
        levels = np.linspace(-1.001, 1, contour_step_size))

    cbar = fig.colorbar(contour, shrink=0.7, label='DBCV score')
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.set_xlim(param_space[0], param_space[1])
    ax.set_ylim(param_space[2], param_space[3])
    
    ax.scatter(x, y, color='k', s=5, marker='o')

    if sel_params:
        ax.scatter(x_sel, y_sel, color='white', s=100, 
            marker='X',  edgecolors='k')
    
    ax.set_title('DBOpt')
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)

    if rescale_x == True:
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks, labels = x_ticks/100)

    if rescale_y == True:
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks, labels = y_ticks/100)

    plt.show()
    
    return fig

def dual_contour_plot(
    sweep_arr: npt.NDArray[Any], 
    sel_params: Dict[str,float]
) -> matplotlib.figure:
    """
    Splits binary HDBSCAN cluster_selection_method (eom,leaf) parameters 
    and plots seperate contour plots for each.

    Args:
        sweep_arr (npt.NDArray[]):
            Array with optimized parameter values and corresponding 
            DBCV scores.
        sel_params (typing.Dict[str,float]):
            Dictionary with parameter names (keys) and values 
            corresponding to the selected optimal parameters

    Returns:


    """

    output_data = data_format_contour(sweep_arr, method = 'dual')
    
    x = output_data[0]
    y = output_data[1]
    xs = output_data[2]
    ys = output_data[3]
    z_resampled = output_data[4]
    contour_step_size  =  output_data[5]
    rescale_x = output_data[6]
    rescale_y = output_data[7]

    if sel_params['cluster_selection_method'] == 'eom':
        sel_params['cluster_selection_method'] = None
        sel_params_0 = sel_params
        sel_params_1 = None
        dim_names = []
        for i,n in enumerate(sel_params_0):
            dim_names.append(n)
    else:
        sel_params['cluster_selection_method'] = None
        sel_params_0 = None
        sel_params_1 = sel_params
        dim_names=[]
        for i,n in enumerate(sel_params_1):
            dim_names.append(n)

    x_param = dim_names[0]
    y_param = dim_names[1]

    param_space = (min(np.min(x[0]), np.min(x[1])), max(np.max(x[0]), np.max(x[1])), 
                   min(np.min(y[0]), np.min(y[1])), max(np.max(y[0]), np.max(y[1])))
    
    dx = param_space[1] - param_space[0]
    dy = param_space[3] - param_space[2]
    aspect_scaling = dx / dy

                              
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  
    cmap = matplotlib.colormaps["rainbow"].copy()

    contour_0 = ax[0].contourf(xs[0], ys[0], z_resampled[0], cmap=cmap, 
        levels = np.linspace(-1.001, 1, contour_step_size[0]))
    
    contour_1 = ax[1].contourf(xs[1], ys[1], z_resampled[1], cmap=cmap, 
        levels = np.linspace(-1.001, 1, contour_step_size[1]))

    for i in range(2):
        ax[i].scatter(x[i], y[i], color='k', s=5, marker='o')
        ax[i].scatter(x[i], y[i], color='k', s=5, marker='o')
        ax[i].set_xlim(param_space[0], param_space[1])
        ax[i].set_ylim(param_space[2], param_space[3])
        ax[i].scatter(x[0], y[0], color='k', s=5, marker='o')
        ax[i].set_xlabel(x_param)
        ax[i].set_ylabel(y_param)
        ax[i].set_aspect(aspect_scaling)

    if sel_params_0 is not None:

        if rescale_x[0] == True:
            y_sel_plot = sel_params_0[y_param]*100
        else:   
            x_sel_plot = sel_params_0[x_param]

        if rescale_y[0] == True:
            y_sel_plot = sel_params_0[y_param]*100
        else:
            y_sel_plot = sel_params_0[y_param]

        ax[0].scatter(x_sel_plot, y_sel_plot, 
            color='white', s=100,marker='X',  edgecolors='k')


    elif sel_params_1 is not None:
        
        if rescale_x[1] == True:
            y_sel_plot = sel_params_0[y_param]*100
        else:   
            x_sel_plot = sel_params_0[x_param]

        if rescale_y[1] == True:
            y_sel_plot = sel_params_0[y_param]*100
        else:
            y_sel_plot = sel_params_0[y_param]

        ax[1].scatter(x_sel_plot, y_sel_plot, 
            color='white', s=100, marker = 'X',  edgecolors='k')


    ax[0].set_title('DBOpt, Method: eom')
    ax[1].set_title('DBOpt, Method: leaf') 
    cbar_0 = fig.colorbar(contour_0, shrink=0.7, label='DBCV score')
    cbar_0.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    cbar_1 = fig.colorbar(contour_1, shrink=0.7, label='DBCV score')
    cbar_1.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])


    if rescale_x[0] == True:
        x_ticks_0 = ax[0].get_xticks()
        ax[0].set_xticks(x_ticks_0, labels = x_ticks_0/100)

    if rescale_x[1] == True:
        x_ticks_1 = ax[1].get_xticks()
        ax[1].set_xticks(x_ticks_1, labels = x_ticks_1/100)

    if rescale_y[0] == True:
        y_ticks_0 = ax[0].get_yticks()
        ax[0].set_yticks(y_ticks_0, labels = y_ticks_0/100)

    if rescale_y[1] == True:
        y_ticks_1 = ax[0].get_yticks()
        ax[1].set_yticks(y_ticks_1, labels = y_ticks_1/100)


    plt.tight_layout()
    plt.show()
    
    return fig



def data_format_contour(
    sweep_arr: npt.NDArray[Any], 
    method: str = 'single'
) -> Tuple[
    List[npt.NDArray[np.float_]], 
    List[npt.NDArray[np.float_]], 
    List[npt.NDArray[np.float_]], 
    List[npt.NDArray[np.float_]], 
    List[npt.NDArray[np.float_]], 
    List[int]
]:

    """
    Utility function for formatting optimization data to be plotted 
    in contour plots.

    Args:
        sweep_arr ():
        method (): 


    Returns:
        
    
    """

    if method == 'single':
        sc_arr = sweep_arr[1::]
        x = sc_arr[:,0]
        y = sc_arr[:,1]
        z = sc_arr[:,2]

        rescale_x = False
        rescale_y = False    
        if max(x)*100 < max(y):
            x = x*100
            rescale_x = True
        if max(y)*100 < max(x):
            y = x*100
            rescale_y = True


        total_pts = len(x)
        total_area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))
        contour_step_size = int(np.round(total_area / (total_area / total_pts)))
        param_space = (np.min(x), np.max(x), np.min(y), np.max(y))

        xs,ys = np.mgrid[param_space[0]:param_space[1], 
                            param_space[2]:param_space[3]] 


        z_resampled = griddata((x, y), z, (xs, ys)) 

        return x, y, xs, ys, z_resampled, contour_step_size, rescale_x, rescale_y  

    elif method == 'dual':
        index_method = np.where(sweep_arr[0] == 'cluster_selection_method')[0][0]

        score_sweep_eom =  np.vstack((sweep_arr[0], 
            sweep_arr[1:][sweep_arr[1:][:,index_method] == 'eom']))

        score_arr_eom = np.delete(score_sweep_eom, index_method, 1)

        score_sweep_leaf =  np.vstack((sweep_arr[0], 
            sweep_arr[1:][sweep_arr[1:][:,index_method] == 'leaf']))

        score_arr_leaf = np.delete(score_sweep_leaf, index_method, 1)

        sc_arr = score_arr_eom[1::]
        sc_arr_2 = score_arr_leaf[1::]
            
        x = sc_arr[:,0]
        y = sc_arr[:,1]
        z = sc_arr[:,2]

        rescale_x = False
        rescale_y = False    
        if max(x)*100 < max(y):
            x = x*100
            rescale_x = True
        if max(y)*100 < max(x):
            y = x*100
            rescale_y = True
        
        x_2 = sc_arr_2[:,0]
        y_2 = sc_arr_2[:,1]
        z_2 = sc_arr_2[:,2]

        rescale_x_2 = False
        rescale_y_2 = False    
        if max(x_2)*100 < max(y_2):
            x_2 = x_2*100
            rescale_x_2 = True
        if max(y_2)*100 < max(x_2):
            y_2 = x_2*100
            rescale_y_2 = True
                                  
        total_pts = len(x)
        total_area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))
        contour_step_size = int(np.round(total_area / (total_area / total_pts)))
        xs,ys = np.mgrid[np.min(x):np.max(x), np.min(y):np.max(y)] 
        z_resampled = griddata((x, y), z, (xs, ys)) 
        
        total_pts_2 = len(x_2)
        total_area_2 = (np.max(x_2) - np.min(x_2)) * (np.max(y_2) - np.min(y_2))
        contour_step_size_2 = int(np.round(total_area_2 / (total_area_2 / total_pts_2)))
        xs_2,ys_2 = np.mgrid[np.min(x_2):np.max(x_2), np.min(y_2):np.max(y_2)]
        z_resampled_2 = griddata((x_2, y_2), z_2, (xs_2, ys_2))


        return tuple((
            [x, x_2], 
            [y, y_2], 
            [xs, xs_2], 
            [ys, ys_2], 
            [z_resampled, z_resampled_2], 
            [contour_step_size, contour_step_size_2], 
            [rescale_x, rescale_x_2],
            [rescale_y,rescale_y_2]
            )) 

