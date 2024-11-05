# DBOpt Script
#Joseph Hammer 2024

from .DBOpt_utility import constant_log, parameter_selector
from .DB_optimize import DB_Optimization
from .auto_plot import auto_plot_params
from .cluster_plots import cluster_plot

#external libraries
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
import kDBCV
import numpy as np

import numpy.typing as npt
from typing import Optional, Union, List, Any, Tuple

class DBOpt:
    def __init__(
        self, 
        algorithm: str, 
        runs: int, 
        rand_n: int, 
        eps: Optional[Union[tuple, float]] = None, 
        min_samples: Optional[Union[tuple,int]] = None, 
        xi: Optional[Union[tuple,float]] = None, 
        min_cluster_size: Optional[Union[tuple,int]] = None, 
        cluster_selection_epsilon: Optional[Union[tuple,float]] = None, 
        cluster_selection_method: Optional[Union[tuple, str]] = None, 
        alpha: Optional[Union[tuple,float]] = None,
        scale_params: bool = False,
        mem_cutoff: float = 25.0
    ) -> None:

        """
        DBOpt initializes the model by setting the algorithm and
        hyperparameters for the DBOpt method. Each algorithm has
        corresponding parameters that can be set as a tuple to define a
        range or a single value to keep that parameter constant. 
        The parameter ranges can be equally scaled via scale_params and 
        the mem_cutoff enables the user to ensure the method does not
        exceed the capabilities of the computer being used.
        
        Args:
            algorithm (str):
                Currently capable of being set to 'DBSCAN', 'HDBSCAN', 
                or 'OPTICS'. See the sklearn clusering documentation
                for more information on each of these algorithms.

            runs (int):
                Sets the number of iterations to optimize the parameter 
                space being explored.

            rand_n (int):
                Sets the number of parameter combinations to initially 
                probe prior to optimization.

            eps (Optional[Union[tuple,float]]):
                Set as a tuple (range that will be optimized) or a 
                float (constant). eps corresponds to the eps parameter 
                for the 'DBSCAN' algorithm.

            min_samples (Optional[Union[tuple,int]]):
                Set as a tuple (range that will be optimized) or an 
                integer (constant). min_samples corresponds to the 
                min_samples parameterfor the 'DBSCAN', 'HDBSCAN', or 
                'OPTICS' depending on which algorithm DBOpt is set to. 

            xi (Optional[Union[tuple,float]]):
                Set as a tuple (range that will be optimized) or float 
                (constant). xi corresponds to the xi parameter for the 
                'OPTICS' algorithm.

            min_cluster_size (Optional[Union[tuple,int]]):
                Set as a tuple (range that will be optimized) or an 
                integer (constant). min_cluster_size  corresponds to 
                the min_cluster_sizeparameter for the 'HDBSCAN' 
                algorithm.                

            cluster_selection_method (Optional[Union[tuple,str]]):
                Set as a tuple (['eom','leaf'] to be a range that will 
                be optimized) or a string ('eom' or 'leaf' to be 
                constant). cluster_selection_method corresponds to the 
                cluster_selection_method parameter for the 'HDBSCAN'
                algorithm.

            alpha (Optional[Union[tuple,float]]):
                Set as a tuple (range that will be optimized) or float 
                (constant). alpha corresponds to the alpha parameter
                for the 'HDBSCAN' algorithm.  

            scale_params (bool):
                Set to True will scale all parameter ranges equally 
                when optimizing. Default is set to False.

            mem_cutoff (float):
                The maximum (approx.) memory allocation the method will 
                allow. When the scoring method is expected to exceed 
                the cutoff,  computation of the DBCV score will be 
                avoided and a -1 score will be output for that set of 
                parameters. Recommended ~80% of total memory available. 
                Default is 25.0 set for a system with 32 GB of memory.

        Returns:
            None

        """

        self.__algorithm = algorithm
        self.__runs = runs
        self.__rand_n = rand_n
        self.__eps = eps
        self.__min_samples = min_samples
        self.__xi = xi
        self.__min_cluster_size = min_cluster_size
        self.__cluster_selection_epsilon = cluster_selection_epsilon
        self.__cluster_selection_method = cluster_selection_method
        self.__alpha = alpha
        self.__mem_cutoff = mem_cutoff
        self.__scale_params = scale_params
        self.__fit_params = constant_log(algorithm, eps, min_samples, min_cluster_size, 
                 cluster_selection_method, cluster_selection_epsilon, alpha, xi)
        
    def optimize(
        self, 
        X: npt.NDArray[np.float_], 
    ) -> None:

        """
        The optimize function performs the bayesian optimization on the
        parameter space defined by the model. DBOpt must be initialized
        first.

        Args:
            X (npt.NDArray[np.float_])
                Array of data to be clustered with each column
                corresponding to a separate dimension. 

        Returns:
            None

        """
        self.parameter_sweep_, self.med_ind_clust_scores_ = DB_Optimization(
                                X, 
                                self.__runs, 
                                self.__rand_n,
                                self.__algorithm, 
                                mem_cutoff = self.__mem_cutoff,
                                scale_params = self.__scale_params,
                                eps = self.__eps, 
                                min_samples = self.__min_samples, 
                                xi = self.__xi, 
                                min_cluster_size = self.__min_cluster_size, 
                                cluster_selection_epsilon = self.__cluster_selection_epsilon,
                                cluster_selection_method = self.__cluster_selection_method, 
                                alpha =  self.__alpha,  
                                )
        
        self.parameters_, self.__fit_params_optimized = parameter_selector(self.parameter_sweep_, 
                                                                           self.med_ind_clust_scores_)
        
        self.__fit_params.update(self.__fit_params_optimized)
        
        
    def fit(
        self, 
        X: npt.NDArray[np.float_]
    ) -> None :  

        """
        The fit function applies the selected parameters from
        optimization and identifies clusters in the data. 

        Args:
            X (npt.NDArray[np.float_])
                Array of data to be clustered with each column
                corresponding to a separate dimension. 

        Returns:
            None

        """

        self.__Xdata = X
        if self.__algorithm == 'DBSCAN':
            clusterer = DBSCAN(eps = self.__fit_params['eps'], 
                               min_samples = int(self.__fit_params['min_samples']))
            
            clusterer.fit(X)
            
            self.labels_ = clusterer.labels_
        
        elif self.__algorithm == 'HDBSCAN':
            if int(self.__fit_params['min_samples']) == -1:
                min_samples = int(self.__fit_params['min_cluster_size'])
            else:
                min_samples = int(self.__fit_params['min_samples'])

            clusterer = HDBSCAN(min_cluster_size = int(self.__fit_params['min_cluster_size']), 
                                min_samples = min_samples, 
                                cluster_selection_method = self.__fit_params['cluster_selection_method'], 
                                cluster_selection_epsilon = self.__fit_params['cluster_selection_epsilon'], 
                                alpha = self.__fit_params['alpha'])
            
            clusterer.fit(X)
            
            self.labels_ = clusterer.labels_
        
        elif self.__algorithm == 'OPTICS':
            clusterer = OPTICS(min_samples = int(self.__fit_params['min_samples']), 
                               xi = self.__fit_params['xi'])
            
            clusterer.fit(X)
            
            self.labels_ = clusterer.labels_
        
        self.DBCV_score_, self.ind_clust_scores_ = kDBCV.DBCV_score(X,self.labels_, ind_clust_scores = True)
        
        
    def optimize_fit(
        self, 
        X: npt.NDArray[np.float_]
    ) -> None:

        """
        Function that runs the optimization and automatically fits the data at the end. 

        Args:
            X (npt.NDArray[np.float_])
                Array of data to be clustered with each column
                corresponding to a separate dimension. 

        Return:
            None

        """

        self.optimize(X)
        self.fit(X)
    
    def plot_optimization(
        self
    ) -> None:

        """
        Automatically plots the parameter sweep of the Bayesian 
        optimization and indicates on the plot which parameters the
        model selects for clustering. Plots types will vary depending
        on the parameters that were optimized.

        Args:
            None

        Return:
            None
        
        """

        self.optimization_plot_ = auto_plot_params(self.parameter_sweep_.copy(), self.__fit_params_optimized.copy())
        
    def plot_clusters(
        self, 
        show_noise: bool = True, 
        ind_cluster_scores: bool = False
    ) -> None:

        """
        Plots the clustered data after fitting with the chosen parameters from Bayesian 
        optimization. 

        Args:
            show_noise (bool):
                Setting to False will remove points assigned as noise 
                from the plot. Default is set to True.

            ind_cluster_scores (bool):
                If set to True, clusters will be coloredbased on 
                corresponding individual cluster scores.Setting to 
                False will randomly color clusters. Default is set to 
                False. 

        Return:
            None
        
        """

        if ind_cluster_scores == False:
            cluster_plot(self.__Xdata.copy(), 
                self.labels_.copy(), show_noise)
        else:
            cluster_plot(self.__Xdata.copy(), 
                self.labels_.copy(), show_noise, self.ind_clust_scores_.copy())