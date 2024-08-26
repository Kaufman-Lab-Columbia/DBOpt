# Draft-DBOpt

DBOpt is a python program for efficient parameter selection tool for density based clusterering algorithms in order to indentify robust and reproducible clusters. DBOpt is currently compatible with the density based clustering algorithms: DBSCAN, HDBSCAN, and OPTICS

## Getting Started
### Dependencies
- DBCV*
- bayes-opt
- sci-kit learn
- numpy
### Installation
DBOpt* can be installed via pip:
```
pip install ....?
```
or
```
pip install ....?
```

## Usage
The DBOptimization function takes one input, X, which is the coordinates of points in d dimensional space. The algorithm along with the relevant hyperparameters must also be defined. These parameters include the bounds of the algorithms parameters to explore, the number of iterations to optimize the parameter space, and the number of initial parameter combinations to seed the parameter space before optimizing. 
### DBOpt-DBSCAN 
For DBSCAN, the bounds for the parameters epsilon and min_samples must be defined. 
```
DB_Optimization(X, algorithm = 'DBSCAN', eps = [4,200], min_samples = [4,200], runs = 200, rand_n = 40)
```
### DBOpt-HDBSCAN
HDBSCAN has two primary parameters, min_cluster_size and min_samples.
```
DB_Optimization(X, algorithm = 'HDBSCAN', min_cluster_size = [4,200], min_samples = [4,200], runs = 200, rand_n = 40)
```
DBOpt is capable of optimizing addition parameters for HDBSCAN including epsilon, the cluster selection method, and alpha.
```
DB_Optimization(X, algorithm = 'HDBSCAN', min_cluster_size = [4,200], min_samples = [4,200], eps = [0,200], method = [0,1], alpha = [0,1], runs = 200, rand_n = 40)
```
### DBOpt-OPTICS
OPTICS can be optimized with the xi method.
```
DB_Optimization(X, algorithm = 'OPTICS', xi = [0.05,0.5], min_samples = [4,200], runs = 200, rand_n = 40)
```
### Scaling parameters
During each optimization itereation, the Bayesian optimization implementation relies on a gaussian prior function. When optimizing parameters that have large differences in space (ex: xi and min_samples for OPTICS), scaling the parameters during optimization may help find optimal parameters. When scale_parameters=True, the xi or eps parameter will scale to the min_samples parameter for OPTICS or DBSCAN, respectively. By default, scale_parameters is set to False. 
Note: The printed output from the optimization will display the scaled parameters, not the parameters being evaluated. Evaluated parameters will be returned in the output array.
```
DB_Optimization(X, algorithm = 'OPTICS', xi = [0.05,0.5], min_samples = [4,200], runs = 200, rand_n = 40, scale_parameters = True)
```
## License
ClustSim is licensed with an MIT license. See LICENSE file for more information.
## Referencing

## Contact 

