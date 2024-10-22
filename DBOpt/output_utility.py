import numpy as np
import numpy.typing as npt
from typing import Any

def output_data_sorter(
    optimizer, 
    pbounds, 
    scale_parameter
    ) -> npt.NDArray[Any]:
    """

    """

    column_labels = []
    output_list = []

    if 'eps' in pbounds:
        eps_arr = np.array([[res["params"]["eps"]] for res in optimizer.res]).flatten()
        column_labels.append('eps')
        if 'eps' in scale_parameter:
            output_list.append(revert_scaling(eps_arr, scale_parameter['eps']))
        else:
            output_list.append(eps_arr)

    if 'min_cluster_size' in pbounds:
        clust_size_arr = np.array([[res["params"]["min_cluster_size"]] for res in optimizer.res]).flatten()
        column_labels.append('min_cluster_size')
        if 'min_cluster_size' in scale_parameter:
            output_list.append(np.round(revert_scaling(clust_size_arr, scale_parameter['min_cluster_size'])).astype(int))
        else:
            output_list.append(np.round(clust_size_arr).astype(int))

    if 'xi' in pbounds:
        xi_arr = np.array([[res["params"]["xi"]] for res in optimizer.res]).flatten()
        column_labels.append('xi')
        if 'xi' in scale_parameter:
            output_list.append(revert_scaling(xi_arr, scale_parameter['xi']))
        else:
            output_list.append(xi_arr)

    if 'min_samples' in pbounds:
        samp_arr = np.array([[res["params"]["min_samples"]] for res in optimizer.res]).flatten()
        column_labels.append('min_samples')
        if 'min_samples' in scale_parameter:
            output_list.append(np.round(revert_scaling(samp_arr, scale_parameter['min_samples'])).astype(int))
        else:
            output_list.append(np.round(samp_arr).astype(int))


    if 'cluster_selection_method' in pbounds:
        m_arr = np.round(
            [[res["params"]["cluster_selection_method"]] for res in optimizer.res]
            ).flatten().astype(int).astype(str)

        column_labels.append('cluster_selection_method')
        m_arr[m_arr == '0'] = 'eom'
        m_arr[m_arr == '1']  = 'leaf'
        output_list.append(m_arr)

    if 'cluster_selection_epsilon' in pbounds:
        cluster_selection_epsilon_arr = np.array([[res["params"]["cluster_selection_epsilon"]] for res in optimizer.res]).flatten()
        column_labels.append('cluster_selection_epsilon')
        if 'cluster_selection_epsilon' in scale_parameter:
            output_list.append(revert_scaling(cluster_selection_epsilon_arr, scale_parameter['cluster_selection_epsilon']))
        else:
            output_list.append(cluster_selection_epsilon_arr)

    if 'alpha' in pbounds:
        alpha_arr = np.array([[res["params"]["alpha"]] for res in optimizer.res]).flatten()
        column_labels.append('alpha')
        if 'alpha' in scale_parameter:
            output_list.append(revert_scaling(alpha_arr, scale_parameter['alpha']))
        else:
            output_list.append(alpha_arr)
        

    score_arr = np.array([[res["target"] for res in optimizer.res]]).flatten()
    output_list.append(score_arr)
    column_labels.append('score')
    output_arr = np.vstack(output_list, dtype = object).T
    rescaled_optimized_arr = np.vstack((column_labels, output_arr), dtype = object)

    return rescaled_optimized_arr

def revert_scaling(
    parameter_arr, 
    scale_parameters
) -> float:

    """
    
    """
    dx = scale_parameters[0]
    min_x = scale_parameters[1]
    rescaled_parameter = (parameter_arr * dx) + min_x

    return rescaled_parameter