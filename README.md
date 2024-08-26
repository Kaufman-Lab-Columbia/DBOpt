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
DBOptimization(X, algorithm = 'DBSCAN', eps = [4,200], min_samples = [4,200], runs = 200, rand_n = 40)
```
### DBOpt-HDBSCAN
HDBSCAN has two primary parameters, min_cluster_size and min_samples.
```
DBOptimization(X, algorithm = 'HDBSCAN', min_cluster_size = [4,200], min_samples = [4,200], runs = 200, rand_n = 40)
```
DBOpt is capable of optimizing addition parameters for HDBSCAN including epsilon, the cluster selection method, and alpha.
```
DBOptimization(X, algorithm = 'HDBSCAN', min_cluster_size = [4,200], min_samples = [4,200], epsilon = [0,200], method = [0,1], alpha = [0,1], runs = 200, rand_n = 40)
```
### DBOpt-OPTICS
OPTICS can be optimized with the xi method.
```
DBOptimization(X, algorithm = 'OPTICS', xi = [0,0.5], min_samples = [4,200], runs = 200, rand_n = 40)
```
## License
ClustSim is licensed with an MIT license. See LICENSE file for more information.
## Referencing

## Contact 

