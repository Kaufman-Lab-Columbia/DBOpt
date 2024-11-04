# DBOpt

DBOpt is a python program enabling reproducible and robust parameter selection for density based clusterering algorithms. The method combines and efficient implementaion of density based cluster validation (DBCV) with Bayesian optimization to find optimal clustering algorithm parameters that maximize the DBCV score. DBOpt is currently compatible with the density based clustering algorithms: DBSCAN, HDBSCAN, and OPTICS. For more information about the DBOpt method read Hammer et al. Preprint at https://www.biorxiv.org/content/10.1101/2024.11.01.621498v1 (2024).

## Getting Started
### Dependencies
- k-DBCV
- BayesianOptimization
- sci-kit learn
- NumPy
  
### Installation
DBOpt can be installed via pip:
```
pip install DBOpt
```


## Usage
DBOpt class can be initialized by setting hyperparameters for the optimization. These include the algorithm to be optimized, the number of optimization iterations (runs), the number of initial parameter combinations to probe (rand_n), and the parameter space that is to be optimized. Each algorithm has its own set of parameters that can be optimized. More information about these parameters can be found in the corresponding scikit-learn documentation.

#### DBOpt-DBSCAN 
For DBSCAN, the relevant parameters are eps and min_samples. Bounds for one or both of these parameters must be set. 
```
model = DBOpt.DBOpt(X, algorithm = 'DBSCAN', runs = 200, rand_n = 40,
                    eps = [3,200], min_samples = [3,200])
```
Parameters can be held constant:
```
model = DBOpt.DBOpt(X, algorithm = 'DBSCAN', runs = 200, rand_n = 40,
                    eps = [4,200], min_samples = 6)
```
#### DBOpt-HDBSCAN
HDBSCAN has two primary parameters, min_cluster_size and min_samples.
```
model = DBOpt.DBOpt(X, algorithm = 'HDBSCAN', runs = 200, rand_n = 40,
                    min_cluster_size = [4,200], min_samples = [4,200])
```
DBOpt is capable of optimizing addition parameters for HDBSCAN including cluster_selection_epsilon, cluster_selection_method, and alpha.
In cases like these when parameter spaces are vastly different in size, it can be helpful to scale all parameters the same by setting scale_params = True. scale_params is set to False by default.
```
model = DBOpt.DBOpt(X, algorithm = 'HDBSCAN',  runs = 200, rand_n = 40,
                    min_cluster_size = [4,200], min_samples = [4,200], eps = [0,200], method = [0,1], alpha = [0,1],
                    scale_params = True)
```
#### DBOpt-OPTICS
OPTICS can currently be optimized with the xi method.
```
model = DBOpt.DBOpt(X, algorithm = 'OPTICS', runs = 200, rand_n = 40,
                    xi = [0.05,0.5], min_samples = [4,200])
```
### Optimizing the parameters
#### Importing Data
The data can be multidimensional coordinates. Here we use the C01 simulation from the data folder.

<p align="center">
    <img width=45% height=45% src="https://github.com/user-attachments/assets/e72dfc14-34ab-484f-816d-bf8d8e46da21">
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
    <img width=60% height=60% src="https://github.com/user-attachments/assets/1487a4c1-44cf-4d0f-9913-a00ae383d1a1">
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
    <img width=40% height=40% src="https://github.com/user-attachments/assets/fbee5fe3-5f78-450e-a79b-11631b96543c">
</p>

```
cluster_plot_modified = model.plot_clusters(show_noise = True, ind_cluster_scores = True)
```

<p align="center">
    <img width=50% height=50% src="https://github.com/user-attachments/assets/46e5a5bd-f0ab-42ee-b228-ed1906ca6e10">
</p>


## License
DBOpt is licensed with an MIT license. See LICENSE file for more information.

## Referencing
If you use DBOpt for your work, cite with the following (currently in preprint):

Hammer, J. L., Devanny, A. J. & Kaufman, L. J. Density-based optimization for unbiased, reproducible clustering applied to single molecule localization microscopy. Preprint at https://www.biorxiv.org/content/10.1101/2024.11.01.621498v1 (2024)

## Contact 
kaufmangroup.rubylab@gmail.com
