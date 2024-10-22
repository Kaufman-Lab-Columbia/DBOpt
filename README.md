# DBOpt

DBOpt is a python program for efficient, reproducible, and robust parameter selection for density based clusterering algorithms. DBOpt is currently compatible with the density based clustering algorithms: DBSCAN, HDBSCAN, and OPTICS.

## Getting Started
### Dependencies
- k-DBCV
- bayes-opt
- sci-kit learn
- numpy
### Installation
DBOpt can be installed via pip:
```
pip install ....?
```
or
```
pip install ....?
```

## Usage
DBOpt class can be initialized by setting hyperparameters for the optimization. These include the algorithm to be optimized, the number of optimization iterations (runs), the number of initial parameter combinations to probe (rand_n), and the parameter space that is to be optimized. Each algorithm has its own set of parameters that can be optimized. More information about these parameters can be found in the corresponding scikit-learn documentation.

#### DBOpt-DBSCAN 
For DBSCAN, the relevant parameters are eps and min_samples. Bounds for one or both of these parameters must be set. 
```
model = DBOpt.DBOpt(X, algorithm = 'DBSCAN', runs = 200, rand_n = 40, eps = [3,200], min_samples = [3,200])
```
Parameters can be held constant:
```
model = DBOpt.DBOpt(X, algorithm = 'DBSCAN', runs = 200, rand_n = 40, eps = [4,200], min_samples = 6)
```
#### DBOpt-HDBSCAN
HDBSCAN has two primary parameters, min_cluster_size and min_samples.
```
model = DBOpt.DBOpt(X, algorithm = 'HDBSCAN', min_cluster_size = [4,200], min_samples = [4,200], runs = 200, rand_n = 40)
```
DBOpt is capable of optimizing addition parameters for HDBSCAN including cluster_selection_epsilon, cluster_selection_method, and alpha.
In cases like these when parameter spaces are vastly different in size, it can be helpful to scale all parameters the same by setting scale_params = True. scale_params is set to False by default.
```
model = DBOpt.DBOpt(X, algorithm = 'HDBSCAN',  runs = 200, rand_n = 40, min_cluster_size = [4,200], min_samples = [4,200], eps = [0,200], method = [0,1], alpha = [0,1], scale_params = True)
```
#### DBOpt-OPTICS
OPTICS can currently be optimized with the xi method.
```
model = DBOpt.DBOpt(X, algorithm = 'OPTICS', xi = [0.05,0.5], min_samples = [4,200], runs = 200, rand_n = 40)
```
### Optimizing the parameters
#### Importing Data
The data can be multidimensional coordinates. Here we use the C01 simulation from the data folder.
<p align="center">
  <img src=![scatter_plot](https://github.com/user-attachments/assets/c97e81a7-3b86-42c3-8e87-4c35ceb96874)
</p>

We create an array X which is a 2D array with x positions in column 0 and y positions in column 1.
#### Optimizing parameters for the data
Once hyperparameters have beeen set, the algorithm can be optimized for the data. 
```
model.optimize(X)
```
Information about the chosen parameters and the full parameter sweep can be extracted after optimizing.
```
parameter_sweep_arr = model.parameter_sweep_
DBOpt_selected_parameters = model.parameters_
```
The optimization can be plotted:
```
parameter_sweep_plot = model.plot_optimization()
```
<p align="center">
  <img width="500" height="300" src=![image](https://github.com/user-attachments/assets/4a3fe62c-e059-4bb7-90cc-d20f7d294179)
</p>
  
### Clustering
The data is clustered via the fit function.
```
model.fit(X)
```
The optimization step and fit step can be performed together:
```
model.optimize_fit(X)
```
After fitting the labels and DBCV score can be stored:
```
labels = model.labels_
DBCV_score = model.DBCV_score_
```
The clusters can be plotted where show_noise will determine if the noise is shown or not (Default = True) and setting ind_cluster_scores = True will plot clusters colormapped to the individual cluster scores instead of colored randomly (Default = False) :
```
cluster_plot = model.plot_clusters()
```
<p align="center">
  <img width="500" height="300" src=![image](https://github.com/user-attachments/assets/e7217963-6de8-4155-a87a-e5f3fef62f13)
</p>
```
cluster_plot_modified = model.plot_clusters(show_noise = True, ind_cluster_scores = True)
```
<p align="center">
  <img width="500" height="300" src=![image](https://github.com/user-attachments/assets/7cd0f56c-e724-4e92-a9c5-68c40413435b)
</p>

## License
DBOpt is licensed with an MIT license. See LICENSE file for more information.
## Referencing

## Contact 

