import numpy as np
from typing import Dict, Any, Tuple, Union, Optional

def set_pbounds(
    algorithm: str,
    scale_params: bool,
    alg_params: Dict[str,Any]  
) -> Tuple[
    Dict[str, Union[int,float]], 
    Dict[str, Union[int,float]], 
    Optional[Dict[str, Tuple[float,float]]]
]:

    """
    Determines whether a parameter is set to a range or is a constant and
    assignes these values to the appropriate dictionary.

    Args:
        algorithm (str):
            Selects which algorithm is being implemented. Currently supports 
            'DBSCAN', 'HDBSCAN', or 'OPTICS'.

        scale_params (bool):
            Default set to False, setting to True scales all parameter spaces
            to be equal in size.  

        alg_params (Dict):

    Return:

    """

    pbounds = {}
    constant_params = {}
    scale_parameter = {}


    if algorithm == 'DBSCAN':

        if type(alg_params['eps'])==float or type(alg_params['eps'])== int:
            constant_params['eps'] = alg_params['eps'].astype('float')
        elif type(alg_params['eps']) == list:
            if scale_params:
                scale_parameter['eps'] = parameter_scaling(alg_params['eps'])
                pbounds['eps'] = [0,1]
            else:
                pbounds['eps'] = alg_params['eps']

        if type(alg_params['min_samples']) == float:
            constant_params['min_samples'] = int(np.round(alg_params['min_samples']))
        elif type(alg_params['min_samples']) == int:
            constant_params['min_samples'] = alg_params['min_samples']
        elif type(alg_params['min_samples']) == list:
            if scale_params:
                scale_parameter['min_samples'] = parameter_scaling(alg_params['min_samples'])
                pbounds['min_samples'] = [0,1]
            else:
                pbounds['min_samples'] = alg_params['min_samples']

    elif algorithm == 'HDBSCAN':

        if type(alg_params['min_cluster_size'])==int:
            constant_params['min_cluster_size'] = alg_params['min_cluster_size']
        elif type(alg_params['min_cluster_size'])==float:
            constant_params['min_cluster_size'] = int(np.round(alg_params['min_cluster_size']))
        elif type(alg_params['min_cluster_size'])==list:
            if scale_params:
                scale_parameter['min_cluster_size'] = parameter_scaling(alg_params['min_cluster_size'])
                pbounds['min_cluster_size'] = [0,1]
            else:
                pbounds['min_cluster_size'] = alg_params['min_cluster_size']
 
        if type(alg_params['min_samples']) == float:
            constant_params['min_samples'] = int(np.round(alg_params['min_samples']))
        elif type(alg_params['min_samples']) == int:
            constant_params['min_samples'] = alg_params['min_samples']
        elif type(alg_params['min_samples']) == list:
            if scale_params:
                scale_parameter['min_samples'] = parameter_scaling(alg_params['min_samples'])
                pbounds['min_samples'] = [0,1]
            else:
                pbounds['min_samples'] = alg_params['min_samples']


        if type(alg_params.get('cluster_selection_method'))==str:
            if alg_params['cluster_selection_method'] == 'eom':
                constant_params['cluster_selection_method'] = 'eom'
            elif alg_params['cluster_selection_method'] == 'leaf':
                constant_params['cluster_selection_method'] = 'leaf'
        elif type(alg_params.get('cluster_selection_method'))==list:
            pbounds['cluster_selection_method'] = [0,1]  
        else:
            constant_params['cluster_selection_method'] = 'eom'



        if type(alg_params.get('cluster_selection_epsilon'))==float:
            constant_params['cluster_selection_epsilon'] = alg_params['cluster_selection_epsilon']
        elif type(alg_params.get('cluster_selection_epsilon'))==int:
            constant_params['cluster_selection_epsilon'] = alg_params['cluster_selection_epsilon'].astype('float64')
        elif type(alg_params.get('cluster_selection_epsilon'))==list:
            if scale_params:
                scale_parameter['cluster_selection_epsilon'] = parameter_scaling(alg_params['cluster_selection_epsilon'])
                pbounds['cluster_selection_epsilon'] = [0,1]
            else:
                pbounds['cluster_selection_epsilon'] = alg_params['cluster_selection_epsilon']
        else:
            constant_params['cluster_selection_epsilon'] = 0.0

        
        if type(alg_params.get('alpha')) == float:
            constant_params['alpha'] = alg_params['alpha']
        if type(alg_params.get('alpha')) == int:
            constant_params['alpha'] = alg_params['alpha'].astype('float64')   
        elif type(alg_params.get('alpha')) == list:
            if scale_params:
                scale_parameter['alpha'] = parameter_scaling(alg_params['alpha'])
                pbounds['alpha'] = [0,1]
            else:
                pbounds['alpha'] = alg_params['alpha']
        else:
            constant_params['alpha'] = 1.0


    elif algorithm == 'OPTICS':

        if type(alg_params['min_samples']) == float:
            constant_params['min_samples'] = int(np.round(alg_params['min_samples']))
        elif type(alg_params['min_samples']) == int:
            constant_params['min_samples'] = alg_params['min_samples']
        elif type(alg_params['min_samples']) == list:
            if scale_params:
                scale_parameter['min_samples'] = parameter_scaling(alg_params['min_samples'])
                pbounds['min_samples'] = [0,1]
            else:
                pbounds['min_samples'] = alg_params['min_samples']

        if type(alg_params['xi'])==float: 
            constant_params['xi'] = alg_params['xi'].astype('float64')
        elif type(alg_params['xi'])== int:
            constant_params['xi'] = alg_params['xi'].astype('float64')
        elif type(alg_params['xi']) == list:
            if scale_params:
                scale_parameter['xi'] = parameter_scaling(alg_params['xi'])
                pbounds['xi'] = [0,1]
            else:
                pbounds['xi'] = alg_params['xi']
        else:
            pbounds['xi'] = alg_params['xi']
            
    else:
        print('Algorthm not defined or unsupported. Select DBSCAN, HDBSCAN, or OPTICS.')


    return pbounds, constant_params, scale_parameter


def parameter_scaling(
    param: Tuple[float,float]
) -> Tuple[float, float]:
    """
    Gets information about the parameter ranges for later scaling
    during and after optimization. Only runs when DBOpt scale_params is 
    set to True.

    Args:
        param (Tuple[float,float]):
            Parameter range that was input into DBOpt. 

    Returns:
        dx (float):
            Size of the parameter range.

        min_param (float):
            Lower bound of the parameter range.


    """

    dx = param[1] - param[0]
    min_param = param[0]
    return dx, min_param


