#parallel_plots

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import numpy as np

from typing import Any, List
import numpy.typing as npt
import matplotlib.figure
def parallel_plot(
    sweep_arr: npt.NDArray[Any], 
    sel_params 
) -> matplotlib.figure:

    """
    Creates a parallel dimension plot with a separate y-axis for each 
    optimized parameter and a final y-axis corresponding to the DBCV
    score. Lines are colored to the DBCV score with white triangles
    indicating DBOpt selected parameters.

    Args:
        sweep_arr ():

        sel_params ():

    Returns:
        Matplotlib figure of the parallel plot.
    
    """

    ynames = sweep_arr[0]
    if 'cluster_selection_method' in ynames:
        method_index = np.where(ynames == 'cluster_selection_method')[0]
        sweep_arr[sweep_arr == 'eom'] = 0
        sweep_arr[sweep_arr == 'leaf'] = 1
    else:
        method_index = None
        
    yvals = sweep_arr[1:] 
    sel_params['score'] = np.max(yvals[:,-1])
    
    xvalues = []
    x = []
    for i in range(len(ynames)):
        x.append(i)
        xvalues.append(np.full(len(yvals), i))

    fig, ax = plt.subplots(1, len(ynames) - 1, sharey=False, figsize=(10,5))
    color_map = matplotlib.cm.rainbow
    cm_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    pe_linestyle = [mpe.Stroke(linewidth=3, foreground='black'), 
                    mpe.Stroke(foreground ='white', alpha=1)]
    
    sel_yvals = []
    for i,n in enumerate(sel_params):
        if n == 'cluster_selection_method':
            if sel_params[n] == 'eom':
                sel_yvals.append(0)
            elif sel_params[n] == 'leaf':
                sel_yvals.append(1)
        else:
            sel_yvals.append(sel_params[n]) 

    yvals_ = np.vstack((yvals, np.array(sel_yvals)))
    scaled_ys  = scale_axes(yvals_, method_index)
    
    for i in range(len(ynames)-1):
        for j in range(len(yvals_[:,i])):

            if j == len(yvals_[:,i])-1:
                ax[i].plot([x[i],x[i+1]], [yvals_[:,i][j], scaled_ys[i][j]], 
                    lw=2, color='white', linestyle='--', path_effects=pe_linestyle)

            else:
                if yvals_[:,-1][j] == -1:
                    ax[i].plot(
                        [x[i],x[i+1]], [yvals_[:,i][j], scaled_ys[i][j]],
                        color='grey', alpha=0.3
                        )

                else:
                    ax[i].plot(
                        [x[i],x[i+1]], [yvals_[:,i][j], scaled_ys[i][j]], 
                        color=color_map(cm_norm(yvals_[:,-1][j]))
                        )

        if i == method_index:
            ax[i].set_ylim(-0.1, 1.1)
            ax[i].set_yticks([0, 1])
            ax[i].set_yticklabels(['eom', 'leaf'])
        else:
            ax[i].set_ylim(min(yvals[:,i]), max(yvals[:,i]))

        ax[i].set_xlim(x[i],x[i + 1])
        ax[i].set_xticks([i])

        if i == method_index:
            ax[i].set_xticklabels(['method'])  
        else:
            ax[i].set_xticklabels([ynames[i]])
        
        ax[i].spines[['top', 'bottom']].set_visible(False)
        if i == len(ynames) - 2:
            axfin = ax[i].twinx()
            axfin.set_ylim(-1, 1)
            axfin.set_yticks([-1, -0.5, 0, 0.5, 1])
            axfin.set_xticks([x[i], x[i + 1]])
            axfin.spines[['top', 'bottom']].set_visible(False)

            if i == method_index:
                axfin.set_xticklabels(['method','DBCV Score'])
            else:
                axfin.set_xticklabels([ynames[-2],'DBCV Score'])

            axfin.scatter(i + 0.98, sel_yvals[i + 1], 
                s=50, marker='>', color ='white', edgecolor='k', zorder=2)

        ax[i].scatter(i + 0.02, sel_yvals[i], 
            s=50, marker='<', color='white', edgecolor='k', zorder=2)
    
    cbar_ax = fig.add_axes([0.12, -0.02, 0.785, 0.05])
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=cm_norm, 
        cmap=color_map), cax=cbar_ax, label='DBCV score', location='bottom',
        orientation='horizontal')

    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    fig.subplots_adjust(wspace=0)
    fig.suptitle('DBOpt', fontsize=22)
    plt.show()

    return fig

def scale_axes(
    ydata, 
    method_index
) -> List[npt.NDArray[np.float_]]:

    """
    Rescales each set of y data to correspond to the next y-axis,
    ensuring proper depiction of the parallel plot.

    Args:
        ydata ():

        method_index():


    Returns:
        List of numpy arrays 
    
    """

    scaled_ys = []
    for i in range(ydata.shape[1] - 1):
        if i == method_index:
            ymin = -0.1
            ymax = 1.1
        else:
            ymin = np.min(ydata[:,i])
            ymax = np.max(ydata[:,i])

        if i + 1 == ydata.shape[1] - 1:
            ymin2 = -1
            ymax2 = 1
        elif i + 1 == method_index:
            ymin2 = -0.1
            ymax2 = 1.1   
        else:
            ymin2 = np.min(ydata[:,i + 1])
            ymax2 = np.max(ydata[:,i + 1])

        out_scale = ymin + ((ydata[:,i + 1] - ymin2) * (ymax - ymin)) / (ymax2 - ymin2)

        scaled_ys.append(out_scale)
    
    return scaled_ys


