import allothersubfiles

def DB_Optimization(X,runs,rand_n,algorithm, scale_params = False, **kwargs):  

    pbounds, constant_params,scale_parameter2 = set_pbounds(algorithm,scale_params, kwargs)
    optimizer, median_ind_clust_scores = optimize_algorithm(X, algorithm,pbounds, constant_params, runs, rand_n, scale_parameter2)
    scoresweeparray = output_data_sorter(optimizer, pbounds, scale_parameter2)

    return scoresweeparray, median_ind_clust_scores