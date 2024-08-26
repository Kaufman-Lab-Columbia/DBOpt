def set_pbounds(algorithm,scale_params, alg_params):

    pbounds = {}
    constant_params = {}
    if algorithm == 'DBSCAN':

        if type(alg_params['eps'])==float or type(alg_params['eps'])== int:
            constant_params["eps"] = alg_params['eps']
        elif type(alg_params['eps']) == list:
            if type(alg_params['min_samples']) == list:
                if scale_params == True:
                    param2_out, scale_parameter2 = transform_parameter_space(alg_params['min_samples'], alg_params['eps'])
                    pbounds["eps"] = param2_out
                else:
                    pbounds["eps"] = alg_params['eps']
                    scale_parameter2 = False
            else:
                pbounds["eps"] = alg_params['eps']
                scale_parameter2 = False

        if type(alg_params['min_samples'])==float or type(alg_params['min_samples'])== int:
            constant_params["min_samples"] = alg_params['min_samples']
        elif type(alg_params['min_samples']) == list:
            pbounds["min_samples"] = alg_params['min_samples']

    elif algorithm == 'HDBSCAN':
        if type(alg_params['min_cluster_size'])==int:
            constant_params['min_cluster_size'] = alg_params['min_cluster_size']
        elif type(alg_params['min_cluster_size'])==list:
            pbounds['min_cluster_size'] = alg_params['min_cluster_size'] 

        if type(alg_params['min_samples'])==float or type(alg_params['min_samples'])== int:
            constant_params['min_samples'] = alg_params['min_samples']
        elif type(alg_params['min_samples']) == list:
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
        elif type(alg_params.get('cluster_selection_epsilon'))==list:
            pbounds['cluster_selection_epsilon'] = alg_params['cluster_selection_epsilon']
        else:
            constant_params['cluster_selection_epsilon'] = 0.0
        
        if type(alg_params.get('alpha'))==float:
            constant_params["alpha"] = alg_params['alpha']
        elif type(alg_params.get('alpha'))==list:
            pbounds["alpha"] = alg_params['alpha']
        else:
            constant_params["alpha"] = 1.0

        scale_parameter2 = False

    elif algorithm == 'OPTICS':

        if type(alg_params['min_samples'])==float or type(alg_params['min_samples'])== int:
            constant_params['min_samples'] = alg_params['min_samples']
        elif type(alg_params['min_samples']) == list:
            pbounds['min_samples'] = alg_params['min_samples']

        if type(alg_params['xi'])==float or type(alg_params['xi'])== int:
            constant_params["xi"] = alg_params['xi']
        elif type(alg_params['xi']) == list:
            if type(alg_params['min_samples']) == list:
                if scale_params == True:
                    param2_out, scale_parameter2 = transform_parameter_space(alg_params['min_samples'], alg_params['xi'])
                    pbounds["xi"] = param2_out
                else:
                    pbounds["xi"] = alg_params['xi']
                    scale_parameter2 = False
            else:
                pbounds["xi"] = alg_params['xi']
                scale_parameter2 = False
    else:
        print('Algorthm not defined or unsupported. Select DBSCAN, HDBSCAN, or OPTICS.')

    return pbounds, constant_params, scale_parameter2

def transform_parameter_space(min_samples, param2):
    min_samples_diff = min_samples[1]-min_samples[0]
    param2_diff = param2[1]-param2[0]
    scaling = min_samples_diff/param2_diff    
    param2_out = [scaling*param2[0], scaling*param2[1]]
    return param2_out, scaling