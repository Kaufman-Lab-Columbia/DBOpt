import time
import numpy as np

from bayes_opt import BayesianOptimization
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
import DBCV 

import numpy.typing as npt
from typing import Tuple, Dict, Optional, Any, Union, List

def optimize_algorithm(
    X: npt.NDArray[np.float_], 
    algorithm: str, 
    pbounds, 
    constant_params_input, 
    runs: int, 
    rand_n: int, 
    scale_parameter_input, 
    mem_cutoff_value: float
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float_]]:

    """
    
    Args:

    Return:
    
    """

    global median_ind_clust_scores 
    global X_coords_data   
    global constant_params
    global scale_parameter
    global mem_threshold

    median_ind_clust_scores= []
    X_coords_data = X 
    constant_params = constant_params_input
    scale_parameter = scale_parameter_input
    mem_threshold = mem_cutoff_value


    if algorithm == 'DBSCAN':
        optimizer = BayesianOptimization(f=black_box_DBSCAN, 
                                    pbounds=pbounds, verbose=2, 
                                    random_state=int(time.time()),
                                    allow_duplicate_points=True)

    elif algorithm == 'HDBSCAN':
        optimizer = BayesianOptimization(f=black_box_HDBSCAN, 
                                    pbounds=pbounds, verbose=2, 
                                    random_state=int(time.time()),
                                    allow_duplicate_points=True)

    elif algorithm == 'OPTICS':
        optimizer = BayesianOptimization(f=black_box_OPTICS, 
                                    pbounds=pbounds, verbose=2, 
                                    random_state=int(time.time()),
                                    allow_duplicate_points=True)

    optimizer.maximize(init_points=runs, n_iter=rand_n)

    return optimizer, np.array(median_ind_clust_scores)   


def black_box_DBSCAN(
    eps: Optional[float] = None, 
    min_samples: Optional[float] = None
    ) -> float:

    """
    
    Args:

    Return:
    
    """

    global median_ind_clust_scores

    X, params, mem_threshold = fetch_black_box_info(
        alg='DBSCAN',
        eps=eps, 
        min_samples=min_samples, 
        )


    model = DBSCAN(eps=params[0], min_samples=params[1])
    model.fit(X)
    labels = model.labels_

    DBCV_score = DBCV.DBCV_score(X, labels, ind_clust_scores=True, mem_cutoff=mem_threshold, batch_mode = True)

    if DBCV_score[0] == -1:
        median_ind_clust_scores.append(-1)
        return -1

    else:
        median_ind_clust_scores.append(np.median(DBCV_score[1]))
        return np.around(DBCV_score[0],2)
    


def black_box_HDBSCAN(
    min_cluster_size: Optional[float] = None, 
    min_samples: Optional[float] = None, 
    cluster_selection_method: Optional[float] = None, 
    cluster_selection_epsilon: Optional[float] = None, 
    alpha: Optional[float] = None
) -> float:  

    """
    
    Args:

    Return:
    
    """

    global median_ind_clust_scores
   
    X, params, mem_threshold = fetch_black_box_info(
        alg='HDBSCAN',
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples, 
        cluster_selection_method=cluster_selection_method, 
        cluster_selection_epsilon=cluster_selection_epsilon, 
        alpha=alpha, 
        )

    if params[1] > params[0]:
        median_ind_clust_scores.append(-1)
        return -1    
    else:
        model = HDBSCAN(
            min_cluster_size=params[0], 
            min_samples=params[1], 
            cluster_selection_method=params[2], 
            cluster_selection_epsilon=params[3], 
            alpha=params[4] 
            )

        model.fit(X)
        labels = model.labels_
        
        DBCV_score = DBCV.DBCV_score(X,labels, ind_clust_scores=True, mem_cutoff=mem_threshold, batch_mode = True)

        if DBCV_score[0] == -1:
            median_ind_clust_scores.append(-1)
            return -1
        else:
            median_ind_clust_scores.append(np.median(DBCV_score[1]))
            return np.around(DBCV_score[0],2)


def black_box_OPTICS(
    min_samples: Optional[float] = None, 
    xi: Optional[float] = None
) -> float:

    """
    
    Args:

    Return:
    
    """

    global median_ind_clust_scores

    X, params, mem_threshold = fetch_black_box_info(
        alg='OPTICS',
        xi=xi, 
        min_samples=min_samples 
        )

    model = OPTICS(xi=params[0], min_samples=params[1])
    model.fit(X_coords_data)
    labels = model.labels_

    DBCV_score = DBCV.DBCV_score(X_coords_data,labels, ind_clust_scores=True, mem_cutoff=mem_threshold, batch_mode = True)

    if DBCV_score[0] == -1:
        median_ind_clust_scores.append(-1)
        return -1

    else:
        median_ind_clust_scores.append(np.median(DBCV_score[1]))
        return np.around(DBCV_score[0],2)



def fetch_black_box_info(
    alg = None, 
    eps = None, 
    min_samples = None, 
    min_cluster_size = None, 
    cluster_selection_method = None, 
    cluster_selection_epsilon = None, 
    alpha = None, 
    xi = None,
) -> Tuple[npt.NDArray[np.float_],List[Any],float]:

    """
    
    Args:

    Return:
    
    """

    global X_coords_data 
    global constant_params
    global scale_parameter
    global mem_threshold

    if alg == 'DBSCAN':

        if eps is not None:
            if 'eps' in scale_parameter:
                eps_out = rescale_param(eps, scale_parameter['eps'])
            else:
                eps_out = eps
        else:
            if type(constant_params.get('eps')) == float:
                eps_out = constant_params.get('eps')
            elif type(constant_params.get('eps')) == int:
                eps_out = constant_params.get('eps').astype('float64') 

        if min_samples is not None:
            if 'min_samples' in scale_parameter:
                min_samples_rescaled = rescale_param(min_samples, scale_parameter['min_samples'])
                min_samples_out = int(np.round(min_samples_rescaled))
            else:
                min_samples_out = int(np.round(min_samples))
        elif type(constant_params.get('min_samples')) == int:
            min_samples_out = constant_params.get('min_samples')
        elif type(constant_params.get('min_samples')) == int:
            min_samples_out = int(np.round(constant_params.get('min_samples')))

        params = [eps_out, min_samples_out]

    elif alg == 'HDBSCAN':

        if min_cluster_size is not None:
            if 'min_cluster_size' in scale_parameter:
                min_cluster_size_rescaled = rescale_param(min_cluster_size, scale_parameter['min_cluster_size'])
                min_cluster_size_out = int(np.round(min_cluster_size_rescaled))
            else:
                min_cluster_size_out = int(np.round(min_cluster_size))   

        elif type(constant_params.get('min_cluster_size')) == int:
            min_cluster_size_out = constant_params.get('min_cluster_size')
        elif type(constant_params.get('min_cluster_size')) == float:
            min_cluster_size_out = int(np.round(constant_params.get('min_cluster_size')))

        if min_samples is not None:
            if 'min_samples' in scale_parameter:
                min_samples_rescaled = rescale_param(min_samples, scale_parameter['min_samples'])
                min_samples_out = int(np.round(min_samples_rescaled))
            else:
                min_samples_out = int(np.round(min_samples))
        elif type(constant_params.get('min_samples')) == int:
            min_samples_out = constant_params.get('min_samples')
        elif type(constant_params.get('min_samples')) == float: 
            min_samples_out = int(np.round(constant_params.get('min_samples'))) 
        else:
            if min_cluster_size is not None:
                min_samples_out = min_cluster_size_out

        if cluster_selection_method is not None:
            cluster_sel_num = int(np.round(cluster_selection_method))
            if cluster_sel_num == 0:
                cluster_selection_method_out = 'eom'
            else:
                cluster_selection_method_out = 'leaf'
        elif type(constant_params.get('cluster_selection_method')) == str:
            cluster_selection_method_out = constant_params.get('cluster_selection_method')
        else:
            cluster_selection_method_out = 'eom'

        if cluster_selection_epsilon is not None:
            if 'cluster_selection_epsilon' in scale_parameter:
                cluster_selection_epsilon_out = rescale_param(cluster_selection_epsilon, scale_parameter['cluster_selection_epsilon'])
            else:
                cluster_selection_epsilon_out = cluster_selection_epsilon
        elif type(constant_params.get('cluster_selection_epsilon')) == float:
            cluster_selection_epsilon_out = constant_params.get('cluster_selection_epsilon')
        elif type(constant_params.get('cluster_selection_epsilon')) == int:
            cluster_selection_epsilon_out = constant_params.get('cluster_selection_epsilon').astype('float64')
        else:
            cluster_selection_epsilon_out = 0.0


        if alpha is not None:
            if 'alpha' in scale_parameter:
                alpha_out = rescale_param(alpha, scale_parameter['alpha'])
            else:
                alpha_out = alpha
        elif type(constant_params.get('alpha')) == float:
            alpha_out = constant_params.get('alpha')
        elif type(constant_params.get('alpha')) == int:
            alpha_out = constant_params.get('alpha').astype('float64')
        else:
            alpha_out = 1.0


        params = [min_cluster_size_out, min_samples_out, cluster_selection_method_out, cluster_selection_epsilon_out, alpha_out]

    elif alg == 'OPTICS':

        if min_samples is not None:
            if 'min_samples' in scale_parameter:
                min_samples_rescaled = rescale_param(min_samples, scale_parameter['min_samples'])
                min_samples_out = int(np.round(min_samples_rescaled))
            else:
                min_samples_out = int(np.round(min_samples))
        elif type(constant_params.get('min_samples')) == int:
            min_samples_out = constant_params.get('min_samples')
        elif type(constant_params.get('min_samples')) == float: 
            min_samples_out = int(np.round(constant_params.get('min_samples')))   

        if xi is not None:
            if 'xi' in scale_parameter:
                xi_out = rescale_param(xi, scale_parameter['xi'])
            else:
                xi_out = xi
        elif type(constant_params.get('xi')) == float:
            xi_out = constant_params.get('xi')
        elif type(constant_params.get('xi')) == int:
            xi_out = constant_params.get('xi').astype('float64')

        params = [xi_out, min_samples_out]

    return X_coords_data, params, mem_threshold


def rescale_param(
    parameter, 
    scale_parameters
) -> float:

    """
    
    Args:

    Return:
    
    """

    dx = scale_parameters[0]
    min_x = scale_parameters[1]
    rescaled_parameter = (parameter * dx) + min_x

    return rescaled_parameter