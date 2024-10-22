#utility
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict, Optional, Any, Union, Optional

def parameter_selector(
    parameter_sweep: npt.NDArray[Any], 
    med_ind_clust_scores: npt.NDArray[np.float_]
) -> Tuple[str, Dict[str, float]]:
    """
    Selects optimial performing parameters from the parameter sweep.

    Args:
        parameter_sweep:
            Array containing the names of parameters with 
            corresponding values from the optimiaztion sweep.
        med_ind_clust_scores:
            Array containing the median individual cluster scores for 
            each corresponding index in the parameter sweep.

    Returns:
        -String specifiying the optimal parameters.
        -Dictionary with keys corresponding to parameter names and
         values corresponding to parameter values. 

    """

    parameter_names = parameter_sweep[0][:-1]
    param_sweep = parameter_sweep[1:]
    param_sweep_scores = param_sweep[:,-1]
    Threshold = np.max(param_sweep_scores)
    Top_params_index = np.where(param_sweep_scores == Threshold)[0]
    selected_medians = med_ind_clust_scores[Top_params_index]
    selected_param_index = Top_params_index[np.argmax(selected_medians)]
    parameter_vals = param_sweep[selected_param_index][:-1]
    
    fit_params = {}
    parameter_output = ''
    for i in range(len(parameter_names)):
        try:
            fit_params[parameter_names[i]] = parameter_vals[i]
        except ValueError:
            fit_params[parameter_names[i]] = parameter_vals[i]
        if i == 0:
            parameter_output += parameter_names[i] + ': ' + str(parameter_vals[i])
        else: 
            parameter_output += ', ' + parameter_names[i] + ': ' + str(parameter_vals[i])
            
    return parameter_output, fit_params

def constant_log(
    algorithm: str,
    eps: Optional[Union[float, Tuple[float,float]]], 
    min_samples: Optional[Union[int, Tuple[int,int]]], 
    min_cluster_size: Optional[Union[int, Tuple[int,int]]], 
    cluster_selection_method: Optional[Union[str, Tuple[str,str]]], 
    cluster_selection_epsilon: Optional[Union[float, Tuple[float,float]]], 
    alpha: Optional[Union[float, Tuple[float,float]]], 
    xi: Optional[Union[float, Tuple[float,float]]]
) -> Dict[str, Union[float,int]]:

    """
    Determines which parameters are constants and stores them in a 
    dictionary.

    Args:
        
    Returns:
        Dictionary with keys corresponding to only parameter names and 
        values that correspond to user defined values only for 
        parameter values set as constants. 
        
    """
    fit_params = {}

    if algorithm == 'DBSCAN':

        if type(eps) != list:
            fit_params['eps'] = eps

        if type(min_samples) != list:
            fit_params['min_samples'] = min_samples


    elif algorithm == 'HDBSCAN':

        if type(min_cluster_size) != list:
            fit_params['min_cluster_size'] = min_cluster_size

        if min_samples is not None:
            if type(min_samples) != list:
                fit_params['min_samples'] = min_samples
        else:
            if type(min_cluster_size) != list:
                fit_params['min_samples'] = min_cluster_size
            else:
                fit_params['min_samples'] = -1

        if cluster_selection_method is not None:
            if type(cluster_selection_method) != list:
                fit_params['cluster_selection_method'] = cluster_selection_method
        else:
            fit_params['cluster_selection_method'] = 'eom'

        if cluster_selection_epsilon is not None:
            if type(cluster_selection_epsilon) != list:
                fit_params['cluster_selection_epsilon'] = cluster_selection_method
        else:
            fit_params['cluster_selection_epsilon'] = 0.0

        if alpha is not None:
            if type(alpha) != list:
                fit_params['alpha'] = alpha
        else:
            fit_params['alpha'] = 1


    elif algorithm == 'OPTICS':

        if type(xi) != list:
            fit_params['xi'] = xi

        if type(min_samples) != list:
            fit_params['min_samples'] = min_samples


    return fit_params