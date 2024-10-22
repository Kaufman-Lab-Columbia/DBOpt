import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
import numpy.typing as npt
from typing import List, Tuple, Dict, Optional


# Flags indicating possible scoring outcomes
_NOT_ENOUGH_CLUSTERS = -2
_ALL_NOISE = -1
_SUCCESS = 0


def DBCV_score(
    X: npt.NDArray[np.float_],
    labels: npt.NDArray[np.int_],
    ind_clust_scores: bool = False, 
    mem_cutoff: float = 25.0,
    batch_mode = False
) -> Tuple[float, Optional[List[float]]]:
    """
    Main function that returns the aggregate and (optionally) individual
    DBCV cluster scores based on input coordinate data (clustered + noise).

    Args:
        X (npt.NDArray[np.float_]):
            An array of float coordinates with shape (N, d), where N is the total
            number of points (clustered + noise) and d is the dimensionality of the
            data.
        labels (npt.NDArray[np.int_]):
            An array of integer labels with shape (N,), where N is the total number
            of points (clustered + noise). The labels map back to the points in X.
            Cluster labels are consecutive integers in the range
            [0, num_clusters - 1], while noise points are assigned a label of -1.
        ind_clust_scores (bool):
            Indicates whether to return the aggregate DBCV score and individual
            cluster scores (True) or only the aggregate score (False).
        mem_cutoff (float):
            A cutoff value for the maximum amount of memory to use when calculating
            intracluster properties, in GB. The value is estimated by
            predict_memory_allocation() below.
        batch_mode (bool):
            Controls whether error message printing should occur (True) or be
            suppressed (False).

    Returns:
        Tuple containing:
            - float: The aggregate DBCV score for the input data.
            - Optional[List[float]]: The list of individual DBCV scores for each
              cluster. Only returned if ind_clust_scores is set to True, otherwise
              returned as None.
    """

    # Format the data for calculation efficiency
    (
        status, 
        cluster_sort, 
        cluster_groups, 
        cluster_ind, 
        n_samp, 
        d, 
        N_clust
    ) = format_data(
        X, labels
    )

    # Early exits where scoring can not be performed
    if status != 0:
        if not batch_mode:
            if status == _ALL_NOISE:
                print('All points assigned to noise')
            elif status == _NOT_ENOUGH_CLUSTERS:
                print('Not enough clusters: must have at least two.')
        
        return (-1, -1) if ind_clust_scores else (-1, None)

    # Early exits due to exceeding memory cutoff
    pred_mem_alloc = predict_memory_allocation(cluster_sort[...,-1])
    if pred_mem_alloc > mem_cutoff:
        if not batch_mode:
            print('memory cutoff reached')

        return (-1, -1) if ind_clust_scores else (-1, None)

    # Sparseness calculation and find core points
    sparseness, core_dists_arr, core_dists_dict, core_pts = intracluster_analysis(
        N_clust, cluster_groups, d, 
    )
    
    # Format core points for intercluster analysis        
    core_X, core_cluster_groups, core_X_ind = core_points_analysis(
        cluster_sort, cluster_ind, core_pts
    )

    # Separation calculation
    separation = intercluster_analysis(
        core_X, core_cluster_groups, core_X_ind, core_dists_arr, core_dists_dict
    )

    # Compute individual and aggregate DBCV scores
    DBCV_val_agg, DBCV_val_ind = weighted_score(
        sparseness, separation, N_clust, cluster_groups, n_samp, ind_clust_scores
    )
    
    return DBCV_val_agg, DBCV_val_ind


def format_data(
    X: npt.NDArray[np.float_],
    labels: npt.NDArray[np.int_]
) -> Tuple[
    int,
    Optional[npt.NDArray[np.float_]],
    Optional[List[npt.NDArray[np.float_]]],
    Optional[npt.NDArray[np.int_]],
    int,
    int,
    int
]:
    """
    Formats coordinates of clustered and noise points for DBCV scoring based
    on input labels.

    Args:
        X (npt.NDArray[np.float_]):
            See DBCV_score() args.
        
        labels (npt.NDArray[np.int_]):
            See DBCV_score() args.

    Returns:
        Tuple containing:
            - An integer status code indicating whether the data can be scored:
              - _ALL_NOISE (int): Scoring is not possible because all points are
                assigned to noise.
              - _NOT_ENOUGH_CLUSTERS (int): Scoring is not possible because not enough
                clusters were found.
              - _SUCCESS (int): The data can be scored.
            - Optional[npt.NDArray[np.float_]]: A master array of float coordinates
              with shape (N, d + 1), where N is the number of clustered points and d
              is the dimensionality of the data. Contains all clustered points and
              associated cluster labels. The clustered points are contained in the
              first d columns, followed by the labels in the last column. The array is
              sorted in ascending order by the label column. Returns as None if the
              data cannot be scored.
            - Optional[List[npt.NDArray[np.float_]]]: A list of arrays with
              len(List) = num_clusters. Each array contains the coordinates
              corresponding to a specific cluster label, stored in ascending order of
              labels. For example, List[0] contains the coordinates belonging to
              cluster 0, List[1] contains cluster 1, etc. Arrays contain floats and
              have shape (N, d + 1), where N is the number of points belonging to the
              current cluster and d is the dimensionality. The first d columns contain
              the coordinates, while the last column contains the label for the
              current cluster. Returns as None if the data cannot be scored.
            - Optional[npt.NDArray[np.int_]]: An array of integer indices for quick
              lookup of clustered points in the sorted master array. The indices are
              stored as follows:
              [0, start_index_1, start_index_2, ...start_index_last, end_index_last].
              The indices are structured such that master_arr[0:start_index_1] gives
              the coordinates for cluster 0, master_arr[start_index_1:start_index_2]
              gives the coordinates for cluster 2, etc. The first element in the list
              is always 0, while the last element always defines the end index of the
              final cluster label (num_cluster - 1). Returns as None if the data
              cannot be scored.
            - int: An integer representing the total number of coordinates (clustered
              + noise). Returns 0 if data cannot be scored.
            - int: An integer indicating the dimensionality of the coordinates.
              Returns 0 if data cannot be scored.
            - int: An integer indicating the number of clusters. Returns 0 if data
              cannot be scored.
    """

    if sum(np.unique(labels) != -1) == 1:
        return _NOT_ENOUGH_CLUSTERS, None, None, None, 0, 0, 0 

    n_samp = X.shape[0]

    # Initial check if all data is noise
    if np.sum(labels) == -n_samp:
        return _ALL_NOISE, None, None, None, 0, 0, 0

       
    d = X.shape[1] 

    # Stack labels with X and sort
    relist = [i for i in X.T]
    relist.append(labels)
    Xl = np.vstack((relist)).T
    Xl_sort = Xl[Xl[..., -1].argsort()]

    # Find where clusters are seperated
    cluster_ID_split = np.where(np.diff(Xl_sort[..., -1]))[0]

    # Checks for clusters that are single or only two points
    # Reassigns them to noise then resorts the data if necessary
    diff_arr = np.append(
        np.diff(cluster_ID_split), (n_samp - 1) - cluster_ID_split[-1]
    )
    idx1 = np.where(diff_arr == 1)[0]
    idx2 = np.where(diff_arr == 2)[0]
    renoise = len(idx1) + (2 * len(idx2))
    if renoise > 0:
        for i in range(len(idx1)):
            Xl_sort[cluster_ID_split[idx1[i]] + 1] = np.append(
                Xl_sort[cluster_ID_split[idx1[i]] + 1][..., :-1], -1
            )
        for i in range(len(idx2)):
            Xl_sort[cluster_ID_split[idx2[i]] + 1] = np.append(
                Xl_sort[cluster_ID_split[idx2[i]] + 1][..., :-1], -1
            )
            Xl_sort[cluster_ID_split[idx2[i]] + 2] = np.append(
                Xl_sort[cluster_ID_split[idx2[i]] + 2][..., :-1], -1
            )
        Xl_sort = Xl_sort[Xl_sort[..., -1].argsort()]   
        cluster_ID_split = np.where(np.diff(Xl_sort[..., -1]))[0]

    # Checks if all data is now noise
    if np.sum(Xl_sort[..., -1]) == -n_samp:
        return _ALL_NOISE, None, None, None, 0, 0, 0


    if Xl_sort[..., -1][0] == -1:
        cluster_sort = Xl_sort[cluster_ID_split[0] + 1:, :]
        cluster_groups = np.split(
            cluster_sort, (cluster_ID_split - (cluster_ID_split[0]))[1:]
        )
        cluster_ind = np.concatenate(
            [cluster_ID_split - (cluster_ID_split[0]), [len(cluster_sort) - 1]]
        )
    else:
        cluster_sort = Xl_sort
        cluster_groups = np.split(cluster_sort, cluster_ID_split + 1)
        cluster_ind = np.concatenate(
            [[0], cluster_ID_split + 1, [len(cluster_sort) - 1]]
        )

    N_clust = len(cluster_groups)
    if N_clust < 2:
        return _NOT_ENOUGH_CLUSTERS, None, None, None, 0, 0, 0

    return _SUCCESS, cluster_sort, cluster_groups, cluster_ind, n_samp, d, N_clust


def predict_memory_allocation(
    labels: npt.NDArray[np.int_]
) -> float:
    """
    Provides a rough estimation of the memory required to perform the
    intracluster_analysis() based on the largest cluster size found in the
    data, as indicated by the labels.

    Args:
        labels (npt.NDArray[np.int_]):
            See format_data() args.

    Returns:
        - float: A maximum memory cutoff (in GB) that will be used to limit the
          size of clusters processed by intracluster_analysis(), to prevent system
          crashes due to insufficient available memory.
    """

    _, l_counts = np.unique(labels[labels >= 0], return_counts=True)
    max_cluster_size = l_counts.max() if l_counts.size > 0 else 0
    predicted_memory = (((max_cluster_size**2) * 8) / 1024**3) * 8

    return predicted_memory


def intracluster_analysis(
    N_clust: int,
    cluster_groups: List[npt.NDArray[np.float_]],
    d: int,
) -> Tuple[
    Dict[int, float],
    npt.NDArray[np.float_],
    Dict[int, npt.NDArray[np.float_]],
    Dict[int, npt.NDArray[np.int_]]
]:
    """
    Analyzes the properties of individual clusters for scoring. Computes the
    all points core distance, identifies core points, and returns the
    sparseness value for each cluster, according to the definitions
    discussed in Moulavi et al.

    Args:
        N_clust (int):
            The total number of clusters to be analyzed.
        cluster_groups (List[npt.NDArray[np.float_]]):
            The sorted clustered points and labels, returned from format_data() in
            Tuple[2].
        d (int):
            The dimensionality of the data.

    Returns:
        Tuple containing:
            - Dict[int, float]: The sparseness values computed for each cluster. The
              key indicates the cluster label and the value indicates the sparseness.
            - npt.NDArray[np.float_]: An array containing the core distances
              computed by all_core_points_distance() for all coordinates.
            - Dict[int, npt.NDArray[np.float_]]: The core distance for core points in
              all clusters. The key indicates the cluster label and the value contains
              an array with the core distance for each core point in the associated
              cluster.
            - Dict[int, npt.NDArray[np.int_]]: The indices of core points in each
              cluster. The key indicates the cluster label and the value contains an
              array of integer indices used to index core points from the full
              coordinate array for each cluster, as contained in the cluster_groups
              data structure.
    """

    core_dists_arr = []
    core_dists_dict = {}
    core_pts = {}
    sparseness = {}
    for i in range(N_clust):
        cluster = cluster_groups[i][:, :-1]

        intraclustmatrix_condensed = pdist(cluster, metric='euclidean') 
        all_pts_core_dists = all_points_core_distance(intraclustmatrix_condensed, d)
        all_core_dists_matrix = np.tile(
            all_pts_core_dists, (all_pts_core_dists.shape[0], 1)
        )
        max_core_dist_matrix  = np.maximum(
            all_core_dists_matrix, all_core_dists_matrix.T
        )

        intraclustmatrix = squareform(intraclustmatrix_condensed)
        intraclust_MRD_matrix = np.maximum(max_core_dist_matrix, intraclustmatrix)
        sparseness[i], core_pts[i] = MST_builder(intraclust_MRD_matrix)
        core_dists_dict[i] = all_pts_core_dists[core_pts[i]]
        core_dists_arr.append(core_dists_dict[i])

    return sparseness, np.hstack(core_dists_arr), core_dists_dict, core_pts


def all_points_core_distance(
    distance_matrix_condensed: npt.NDArray[np.float_],
    d: int
) -> npt.NDArray[np.float_]:
    """
    Helper function for intracluster_analysis() that computes the all points
    core distance of points in a cluster according to the definition
    discussed in Moulavi et al.

    Args:
        distance_matrix_condensed (npt.NDArray[np.float_]):
            The condensed pairwise distance matrix for the points in the current
            cluster, as computed by pdist() in scipy.
        d (int):
            The dimensionality of the data.

    Returns:
        - npt.NDArray[np.float_]: Array containing the all points core distance
          for the current cluster.
    """

    distance_matrix_condensed[distance_matrix_condensed == 0] = np.inf
    distance_matrix_condensed = (1 / distance_matrix_condensed)**d
    distance_matrix = squareform(distance_matrix_condensed)
    all_pts_core_dists = (
        distance_matrix.sum(axis=1) / (distance_matrix.shape[0] - 1)
    )
    
    if np.sum(all_pts_core_dists) > 0:
        return all_pts_core_dists**(-1 / d)
    else:
        return all_pts_core_dists


def MST_builder(
    MRD_matrix: npt.NDArray[np.float_]
) -> Tuple[float, npt.NDArray[np.int_]]:
    """
    Helper function for intracluster_analysis() that identifies core points
    based on the all points core distance, and then computes the sparseness
    of the current cluster.

    Args:
        MRD_matrix (npt.NDArray[np.float_]):
            Array of shape (N, N), where N is the number of points in the current
            cluster. The MRD_matrix contains mutual reachability distances, which
            consider the max core distance and euclidean distance between each point
            pair, and contains max{max_core_distance, euclidean_distance} for each
            pair.

    Returns:   
        - float: The sparseness of the current cluster.
        - npt.NDArray[np.int_]: An array containing the indices of the core points
          identified from the minimum spanning tree for the current cluster.
    """

    MST_arr = minimum_spanning_tree(MRD_matrix).toarray()
    if np.sum(MST_arr) == 0:
        sparseness = 0
        core_pts = np.array([0], dtype='int64')
    else:
        check_MST = np.hstack(np.where(MST_arr > 0))
        unique_vals, index, count = np.unique(
            check_MST, return_counts=True, return_index=True
        )
        core_pts = unique_vals[count > 1]
        sparseness = MST_arr[core_pts][:, core_pts].max()
        if sparseness == 0:
            sparseness = MST_arr.max()

    return sparseness, core_pts


def core_points_analysis(
    cluster_sort: npt.NDArray[np.float_],
    cluster_ind: npt.NDArray[np.int_],
    core_pts: Dict[int, npt.NDArray[np.int_]]
) -> Tuple[
    npt.NDArray[np.float_],
    List[npt.NDArray[np.float_]],
    npt.NDArray[np.int_]
]:
    """
    Formats core points for intercluster_analysis().

    Args:
        cluster_sort (npt.NDArray[np.float_]):
            See format_data(), returned in Tuple[1].
        cluster_ind (npt.NDArray[np.int_]): 
            See format_data(), returned in Tuple[3].
        core_pts (npt.NDArray[np.int_]): 
            See intracluster_analysis(), returned in Tuple[3].

    Returns:
        Tuple containing:
            - npt.NDArray[np.float_]: An array of shape (N, d + 1), where N is the
              total number of core points across all clusters, and d is the
              dimensionality of the data. The last column contains the cluster label
              associated with each core point, while the coordinate is contained in
              the first d columns. The array is sorted in ascending order by the
              cluster labels column.
            - List[npt.NDArray[np.float_]]: A list of arrays containing the core point
              coordinates. Contains the same data as that returned by this function in
              Tuple[0], but formatted such that List[0] contains the core point
              coordinates for cluster label 0, List[1] contains the core point
              coordinates for cluster label 1, etc.
            - npt.NDArray[np.int_]: An array of integer indices indicating the
              transitions between cluster labels for the sorted core points that are
              returned in Tuple[0] of this function. The indices are structured such
              that [index_arr[0]:index_arr[1]] defines the core point coordinates
              belonging to cluster label 0, [index_arr[1]:index_arr[2]] defines the
              core point coordinates belonging to cluster label 1, etc.
    """

    core_pts_arr = np.array(list(core_pts.values()), dtype=object)

    if len(core_pts_arr.shape) > 1:
        core_clust_ind = np.hstack(core_pts_arr + (cluster_ind[:-1])[..., None])
    else:
        core_clust_ind = np.hstack(core_pts_arr + (cluster_ind[:-1]))

    cols = np.arange(cluster_sort.shape[1]) # np.arange() is faster than range() here
    rows = core_clust_ind.astype(int)

    core_X = cluster_sort.ravel()[
        (cols + (rows * cluster_sort.shape[1]).reshape((-1, 1))).ravel()
    ].reshape(rows.size, cols.size)
    
    cluster_ID_split = np.where(np.diff(core_X[..., -1]))[0] + 1 
    core_cluster_groups = np.split(core_X[..., :-1], cluster_ID_split)
    core_X_ind = np.concatenate([[0], cluster_ID_split, [len(core_X)]]) 
    
    return core_X, core_cluster_groups, core_X_ind


def intercluster_analysis(
    core_X: npt.NDArray[np.float_],
    core_cluster_groups: List[npt.NDArray[np.float_]],
    core_X_ind: npt.NDArray[np.int_],
    core_dists_arr: npt.NDArray[np.float_],
    core_dists_dict: Dict[int, npt.NDArray[np.float_]]
) -> Dict[int, float]:
    """
    Calculates the separation value for each cluster according to the
    definitions discussed in Moulavi et al.

    Args:
        core_X (npt.NDArray[np.float_]):
            See core_points_analysis(), returned in Tuple[0].
        core_cluster_groups (List[npt.NDArray[np.float_]]):
            See core_points_analysis(), returned in Tuple[1].
        core_X_ind (npt.NDArray[np.int_]):
            See core_points_analysis(), returned in Tuple[2].
        core_dists_arr (npt.NDArray[np.float_]):
            See intracluster_analysis() returned in Tuple[1].
        core_dists_dict (Dict[int, npt.NDArray[np.float_]]):
            See intracluster_analysis(), returned in Tuple[2].

    Returns:
        - Dict[int, float]: The separation values for all clusters. The key
          indicates the cluster label, and the value indicates the associated
          separation for the cluster.
    """

    separation = {}

    Tree = cKDTree(core_X[:, :-1])

    for i in range(len(core_cluster_groups)):
        cluster = core_cluster_groups[i]
        cluster_size = len(cluster)
        
        NN_array = Tree.query(cluster, k=cluster_size + 1)
        NN_array_min = []
        for j in range(cluster_size):
            NN_array_j = np.vstack((NN_array[0][j], NN_array[1][j])).T 
            NN_array_j = NN_array_j[
                np.where(
                    np.logical_or(
                        NN_array_j[:, 1].astype(int) < core_X_ind[i],
                        NN_array_j[:, 1].astype(int) >= core_X_ind[i + 1]
                    )
                )
            ]
            if len(NN_array_j) > 1:
                NN_array_j = NN_array_j[NN_array_j[:, 0] == np.min(NN_array_j[:, 0])]

            NN_array_min.append(np.hstack((NN_array_j[0], j + core_X_ind[i])))  

        NN_array_min = np.vstack(NN_array_min)
        
        min_Edists = NN_array_min[:, 0]
        outer_core_pts =  NN_array_min[:, 1].astype(int)
        outer_core_dists = core_dists_arr[outer_core_pts]
        inner_core_dists = core_dists_dict[i]
        
        MRD_arr = np.vstack((min_Edists, inner_core_dists, outer_core_dists)).T
        MRD_init = MRD_arr.max(axis=1)
        init_min_MRD = np.min(MRD_init)
        MRD_det = MRD_arr.argmax(axis=1)
        
        check_radially = np.where(
            np.logical_and(
                MRD_det==2,
                MRD_arr[:, 0] < init_min_MRD
            )
        )[0]

        if len(check_radially) == 0:
            separation[i] = init_min_MRD
        else:
            radial_check = Tree.query_ball_point(cluster[check_radially], init_min_MRD)
            for j in range(len(radial_check)):
                pts = np.array(radial_check[j])           
                outer_pts = pts[
                    np.where(
                        np.logical_or(
                            pts < core_X_ind[i], 
                            pts >= core_X_ind[i + 1]
                        )
                    )
                ]
                min_outer_core_dist = min(core_dists_arr[outer_core_pts])
                MRD_arr[check_radially[j]][2] = min_outer_core_dist

            MRD_fin = MRD_arr.max(axis=1)
            separation[i] = np.min(MRD_fin)

    return separation


def weighted_score(
    sparseness: Dict[int, float],
    separation: Dict[int, float],
    N_clust: int,
    cluster_groups: List[npt.NDArray[np.float_]],
    n_samp: int, 
    ind_clust_scores: bool
) -> Tuple[float, Optional[List[float]]]:
    """
    Performs weighted averaging of individual cluster scores to yield the
    aggregate DBCV score. Optionally returns individual scores if desired.

    Args:
        sparseness (Dict[int, int]):
            The sparseness values for all clusters, returned from
            intracluster_analysis() in Tuple[0].
        separation (Dict[int, float]):
            The separation value for all clusters, returned from
            intercluster_analysis().
        N_clust (int):
            The number of clusters to be analyzed.
        cluster_groups (List[npt.NDArray[np.float_]]):
            The sorted clusters, returned from format_data() in Tuple[2].
        n_samp (int):
            The total number of coordinates, returned from format_data() in
            Tuple[4].
        ind_cluster_scores (bool):
            Whether individual cluster scores should be returned (True) or not
            (False).

    Returns:
        - float: The aggregate DBCV score for the clustered data.
        - Optional[List[float]]: The list of individual DBCV scores for each
          cluster.
    """
    
    cluster_score_set = []
    DBCV_val = 0
    for i in range(N_clust):
        cluster_validity = (
            (separation[i] - sparseness[i]) /
            max(separation[i], sparseness[i])
        )
        
        cluster_score_set.append(cluster_validity)
        cluster_size = len(cluster_groups[i])
        DBCV_val += (cluster_size / n_samp) * cluster_validity

    if ind_clust_scores == True:
        return DBCV_val, cluster_score_set  
    else:
        return DBCV_val, None  
