from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
import time
import numpy as np
from bayes_opt import BayesianOptimization
import DBCV_working_sparsefix as DBCV 

def optimize_algorithm(X, algorithm,pbounds,constant_params,runs,rand_n, scale_parameter2):
    global median_ind_clust_scores 
    global X_coords_data   
    global constant_params
    global scale_parameter2 

    X_coords_data = X 
    median_ind_clust_scores= []

    if algorithm == 'DBSCAN':
        optimizer = BayesianOptimization(f = black_box_DBSCAN, 
                                    pbounds = pbounds, verbose = 2, 
                                    random_state = int(time.time()),
                                    allow_duplicate_points = True)

    elif algorithm == 'HDBSCAN':
        optimizer = BayesianOptimization(f = black_box_HDBSCAN, 
                                    pbounds = pbounds, verbose = 2, 
                                    random_state = int(time.time()),
                                    allow_duplicate_points = True)

    elif algorithm == 'OPTICS':
        optimizer = BayesianOptimization(f = black_box_OPTICS, 
                                    pbounds = pbounds, verbose = 2, 
                                    random_state = int(time.time()),
                                    allow_duplicate_points = True)

    optimizer.maximize(init_points = runs, n_iter = rand_n)
    return optimizer, median_ind_clust_scores   



def black_box_DBSCAN(eps = False, min_samples = False):
    global X_coords_data 
    global constant_params
    global median_ind_clust_scores
    global scale_parameter2 
    if type(constant_params.get('eps')) == float:
        eps= constant_params.get('eps')
        
    if type(constant_params.get('min_samples')) == int:
        min_samples = constant_params.get('min_samples')
    
    else: min_samples = int(np.round(min_samples))
    
    if scale_parameter2 == False:
        model = DBSCAN(eps = eps, min_samples = min_samples)
    else:
        model = DBSCAN(eps = eps/scale_parameter2, min_samples = min_samples)

    model.fit(X_coords_data)
    labels = model.labels_

    if len(np.unique(labels))>2:
        DBCV_score = DBCV.DBCV(X_coords_data,labels, ind_clust_scores = True, mem_cutoff = 15000)
        if DBCV_score[0]==-1:
            median_ind_clust_scores.append(-1)
            return -1

        else:
            median_ind_clust_scores.append(np.median(DBCV_score[1]))
            return np.around(DBCV_score[0],2)
    else:
        median_ind_clust_scores.append(-1)
        return -1


def black_box_HDBSCAN(min_cluster_size = False, min_samples = False, cluster_selection_method = False, cluster_selection_epsilon = False, alpha = False):
    global X_coords_data
    global constant_params  
    global median_ind_clust_scores

    if type(constant_params.get('min_cluster_size')) == int:
        min_cluster_size = constant_params.get('min_cluster_size')
    else:
        min_cluster_size = int(np.round(min_cluster_size))

    if type(constant_params.get('min_samples')) == int:
        min_samples = constant_params.get('min_samples')
    else: 
        min_samples = int(np.round(min_samples)) 

    if type(constant_params.get('cluster_selection_method')) == str:
        cluster_selection_method = constant_params.get('cluster_selection_method')
    else: 
        cluster_selection_method = int(np.round(cluster_selection_method))
        if cluster_selection_method == 0:
            cluster_selection_method = 'eom'
            
        else: 
            cluster_selection_method = 'leaf'  
            
    if type(constant_params.get('cluster_selection_epsilon ')) == float:
        cluster_selection_epsilon  = constant_params.get('cluster_selection_epsilon ')
    # else:
    #     cluster_selection_epsilon  = cluster_selection_epsilon.item()

    if type(constant_params.get('alpha')) == float:
        alpha = constant_params.get('alpha')
    # else:
    #     alpha = alpha.item()

    if min_samples>min_cluster_size:
        median_ind_clust_scores.append(-1)
        return -1
        
    else:
        model = HDBSCAN(min_cluster_size = min_cluster_size , min_samples=min_samples, cluster_selection_method = cluster_selection_method, cluster_selection_epsilon = cluster_selection_epsilon, alpha = alpha)
        model.fit(X_coords_data)
        labels = model.labels_
        
        if len(np.unique(labels))>2:
            DBCV_score = DBCV.DBCV(X_coords_data,labels, ind_clust_scores = True, mem_cutoff = 15000)
            if DBCV_score[0]==-1:
                median_ind_clust_scores.append(-1)
                return -1

            else:
                median_ind_clust_scores.append(np.median(DBCV_score[1]))
                return np.around(DBCV_score[0],2)
        else:
            median_ind_clust_scores.append(-1)
            return -1

def black_box_OPTICS(min_samples = False, xi = False):
    global X_coords_data
    global constant_params
    global median_ind_clust_scores
    global scale_parameter2

    if type(constant_params.get('min_samples')) == float:
        min_samples = constant_params.get('min_samples')
        min_samples = int(np.round(min_samples))
    elif type(constant_params.get('min_samples')) == int:
        min_samples = constant_params.get('min_samples')
    else:
        min_samples = int(np.round(min_samples)) 

    if type(constant_params.get('xi')) == float:
        xi = constant_params.get('xi')

    if scale_parameter2 == False:
        model = OPTICS(min_samples = min_samples, xi = xi)
    else: 
        model = OPTICS(min_samples = min_samples, xi = xi/scale_parameter2)

    model.fit(X_coords_data)
    labels = model.labels_


    if len(np.unique(labels))>2:


        DBCV_score = DBCV.DBCV(X_coords_data,labels, ind_clust_scores = True, mem_cutoff = 15000)

        if DBCV_score[0]==-1:
            median_ind_clust_scores.append(-1)
            return -1

        else:
            median_ind_clust_scores.append(np.median(DBCV_score[1]))
            return np.around(DBCV_score[0],2)
    else:
        median_ind_clust_scores.append(-1)
        return -1