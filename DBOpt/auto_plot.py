# auto_plot 
from .parallel_plots import parallel_plot
from .countour_plots import contour_plot, dual_contour_plot
from .single_dim_plots import single_dim_plot, dual_single_dim_plot

from typing import Dict, Any
import numpy.typing as npt
import matplotlib.figure

def auto_plot_params(
    sweep_arr: npt.NDArray[Any], 
    sel_params: Dict[str, float]
) -> matplotlib.figure:
    """
    Determines which plotting function is most appropriate for 
    generating parameter sweep plots. Binary parameters (i.e. HDBSCAN 
    cluster_selection_method) automatically plots with parallel plot 
    along with either two dimension (contour) of one dimension 
    (single_dim) plots.

    Args:
        sweep_arr (npt.NDArray[]):
            Array with optimized parameter values and corresponding 
            DBCV scores.
        chosen_parameters (typing.Dict[str,float]):
            Dictionary with parameter names (keys) and values 
            corresponding to the selected optimal parameters.

    Returns:
        plot ():
            Matplotlib figure depicting the optimization sweep.

    """
    if 'cluster_selection_method' in sweep_arr[0]:
        plot = parallel_plot(sweep_arr.copy(), sel_params.copy())
        if len(sweep_arr[0]) == 4:
            plot2 = dual_contour_plot(sweep_arr.copy(), sel_params.copy())
            return plot, plot2   
        elif len(sweep_arr[0]) == 3:
            plot2 = dual_single_dim_plot(sweep_arr.copy(), sel_params.copy())
            return plot, plot2   
    
    elif len(sweep_arr[0]) == 3:
        plot = contour_plot(sweep_arr.copy(), sel_params.copy())
        
    elif len(sweep_arr[0]) == 2:
        plot = single_dim_plot(sweep_arr.copy(), sel_params.copy())
        
    return plot