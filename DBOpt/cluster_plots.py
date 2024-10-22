#cluster plotter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typing import Optional
import numpy.typing as npt
import matplotlib.figure
def cluster_plot(
    X: npt.NDArray[np.float_], 
    labels: npt.NDArray[np.float_], 
    show_noise: bool = True, 
    ind_cluster_scores: Optional[npt.NDArray[np.float_]] = None
) -> matplotlib.figure:
    """
    Plots resulting clusters (2D) with options to show noise and color 
    map to individual cluster scores. 

    Args:
        X (npt.NDArray[np.float_]: 
            2D Array of coordinates.
        labels (npt.NDArray[np.float_]: 
            1D Array of cluster assignments cooresponding to X.
        show_noise (bool):
            Sets whether or not to show noise points, which will appear in grey.
        ind_cluster_scores (Optional[npt.NDArray[np.float_]]):
            Changes the color assignment of clusters to correspond to 
            the individualcluster scores. None will randomly assign 
            colors to clusters.

    Returns:

    """
    X_lab = np.column_stack((X, labels))
    sorted_X_lab = X_lab[X_lab[:,-1].argsort()]
    cluster_ID_split = np.where(np.diff(sorted_X_lab[:,-1]))[0] 
    cluster_groups = np.split(sorted_X_lab[...,:-1], cluster_ID_split+1)
    if ind_cluster_scores:
        color_map = matplotlib.cm.rainbow
        cm_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        for i in range(len(cluster_groups)):
            color_i = color_map(cm_norm(ind_cluster_scores[i-1]))
            if i == 0:
                if show_noise == True:
                    plt.scatter(*cluster_groups[i].T, color='grey', 
                        alpha = 0.5, s = 1)  
                else:
                    continue
            else:
                plt.scatter(*cluster_groups[i].T, color=color_i, s=1) 
        
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=cm_norm, cmap=color_map), 
            ax = plt.gca(), label = 'Individual Cluster Score'
            )

        cbar.set_ticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
        plt.gca().set_aspect('equal')
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.title('DBOpt Clusters', fontsize=18)
        plt.show()
    else:
        for i in range(len(cluster_groups)):
            color_i = generate_non_grey_color()
            if i == 0:
                if show_noise == True:
                    plt.scatter(*cluster_groups[i].T, 
                        color = 'grey', alpha=0.5, s=1
                        )  
                else:
                    continue
            else:
                plt.scatter(*cluster_groups[i].T, color=color_i, s=1)
                
        plt.gca().set_aspect('equal')
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.title('DBOpt Clusters', fontsize=18)
        plt.show()
        
def generate_non_grey_color(
    tol: float = 0.1, 
    scale: float = 1.0
) -> npt.NDArray[np.float_]:
    """
    Generates a random color while allowing avoidance of grey. Used 
    for assigning color to clusters.

    Args:
        tol: 
            Value used to avoid greyscale with a 

        scale:
            

    Returns:
        Array of RGB values

    """

    while True:
        color = np.random.random(3)
        # Check if the color is not greyscale using absolute tolerance
        if not (np.abs(color[0] - color[1]) <= tol and np.abs(color[1] - color[2]) <= tol):
            return color*scale