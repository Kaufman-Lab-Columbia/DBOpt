import numpy as np

def output_data_sorter(optimizer, pbounds, scale_parameter2):

    column_labels = []
    output_arr = []

    if scale_parameter2 == False:
        scale_parameter2  = 1

    if pbounds.get('eps') != None:
        eps_arr = np.array([[res["params"]["eps"]] for res in optimizer.res]).flatten()
        output_arr.append(eps_arr/scale_parameter2)
        column_labels.append('epsilon')

    if pbounds.get('min_samples') != None:
        samp_arr = np.round([[res["params"]["min_samples"]] for res in optimizer.res]).flatten()
        output_arr.append(samp_arr)
        column_labels.append('min_samples')

    if pbounds.get('min_cluster_size') != None:
        clust_size_arr = np.round([[res["params"]["min_cluster_size"]] for res in optimizer.res]).flatten()
        output_arr.append(clust_size_arr)
        column_labels.append('min_cluster_size')

    if pbounds.get('cluster_selection_method') != None:
        m_arr = np.round([[res["params"]["cluster_selection_method"]] for res in optimizer.res]).flatten()
        output_arr.append(m_arr)
        column_labels.append('cluster_selection_method')

    if pbounds.get('alpha') != None:
        alpha_arr = np.array([[res["params"]["alpha"]] for res in optimizer.res]).flatten()
        output_arr.append(alpha_arr)
        column_labels.append('alpha')

    if pbounds.get('xi') != None:
        xi_arr = np.array([[res["params"]["xi"]] for res in optimizer.res]).flatten()
        output_arr.append(xi_arr/scale_parameter2)
        column_labels.append('xi')

    score_arr = np.array([[res["target"] for res in optimizer.res]]).flatten()
    output_arr.append(score_arr)
    column_labels.append('score')
    output_arr = np.vstack(output_arr).T
    output = np.vstack((column_labels, output_arr))

    return output