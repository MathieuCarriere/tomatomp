import numpy as np
import math
from collections import defaultdict

import gudhi as gd
from gudhi.clustering.tomato import Tomato

import multipers as mp
from sklearn.neighbors import NearestNeighbors
import mpmath as mp2
import networkx as nx
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

class tomato_graph_slice:

    def __init__(self, graph, module, weights, direction, basepoint, merging_threshold, n_clusters=None, noise_variance=0.0001, X=None):

        self.graph = graph
        self.module = module
        self.weights = weights
        self.direction = np.array(direction)
        self.basepoint = np.array(basepoint)
        self.merging_threshold = merging_threshold
        self.n_clusters = n_clusters
        self.n_parameters = weights.shape[1]
        self.direction_null_index = np.where(self.direction == 0)[0]
        self.direction_mask = (self.direction != 0)
        self.noise_variance = noise_variance
        self.X = X

        barcode_raw = self.slice_mma()
        self.barcode = {i: (a, b) for i, (a, b) in enumerate(barcode_raw[0]) if not (math.isinf(a) and math.isinf(b))}
        tomato_result, tomato_diagram = self.tomato_slice(self.weights)

        #plt.figure()
        #plt.scatter(-tomato_diagram[:, 0], -tomato_diagram[:, 1], s=10, c='blue', label='ToMATo PD')
        #plt.scatter([pt[0] for pt in self.barcode.values()], [pt[1] for pt in self.barcode.values()], label='sliced-MMA PD', s=15, facecolors='none', edgecolors='red')
        #plt.legend()
        #plt.title("ToMATo PD cardinal = " + str(len(tomato_diagram)) + ", sliced-MMA PD cardinal = " + str(len(self.barcode)))
        #plt.show()

        self.tomato_diagram = tomato_diagram
        self.tomato_labels = tomato_result
        list_tomato_labels = np.unique(self.tomato_labels)
        cores = []
        for tl in list_tomato_labels:
            population = np.argwhere(self.tomato_labels == tl).ravel()
            cores.append(self.weights_1D[population].min())
        tomato_cores = np.array(cores)[:,None]
        mma_cores = np.array([pt[0] for pt in list(self.barcode.values())])[:,None]
        #print(np.sort(tomato_cores.flatten())[:100]) 
        #print(np.sort(mma_cores.flatten())[:100])
        mma_colors = list(self.barcode.keys())
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mma_cores)
        _, indices = np.array(nbrs.kneighbors(tomato_cores))
        indices = indices.astype(int).flatten()
        correspondence = {tl: mma_colors[indices[i]] for i, tl in enumerate(list_tomato_labels)}
        self.colored_clusters = {i:correspondence[tl] for i, tl in enumerate(self.tomato_labels)}

    def slice_mma(self):
        return self.module.barcode2(self.basepoint, self.direction, degree=0, full=False, threshold=False, keep_inf=True)

    def filtration(self, weights, direction, basepoint, null_index=None, mask=None):
        #filtration_1d = np.maximum(weights[:,0]+(basepoint[1]-basepoint[0]), weights[:,1])
        #print(basepoint, direction)
        if not np.any(mask):
            raise ValueError("Null direction")
        filtration_1d = np.max((weights[:,mask] - basepoint[None,mask]) / direction[None,mask], axis=1)
        filtration_1d = np.where( np.any(weights[:,null_index] > basepoint[None,null_index], axis=1), np.inf, filtration_1d)
        return filtration_1d

    def tomato_slice(self, weights):
        self.weights_1D = self.filtration(weights, self.direction, self.basepoint, self.direction_null_index, self.direction_mask)
        #plt.figure()
        #plt.scatter(self.X[::10, 0], self.X[::10, 1], c=-np.array(self.weights_1D)[::10], cmap='rainbow', s=10, alpha=1)
        #plt.colorbar()
        #plt.show()
        #plt.figure()
        #plt.scatter(self.X[::10, 0], self.X[::10, 1], c=t.labels_[::10], cmap='rainbow', s=10, alpha=1)
        #plt.show()
        t = Tomato(graph_type='manual', density_type='manual', merge_threshold=self.merging_threshold, n_clusters=self.n_clusters)
        t.fit(self.graph, weights=-np.array(self.weights_1D))
        return t.labels_, t.diagram_


class Tomatomp:

    def compute_knn_graph_weights(self, X, weights, K):
        K = 30
        n_points = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(X)
        _, indices = nbrs.kneighbors(X)
        indices = indices[:, 1:]
        rank = {}
        weights_dict = {}
        for i in range(n_points):
            rank[i] = {indices[i, r]: r + 1 for r in range(K)}
        new_vertex = n_points
        G_subdiv = nx.Graph()
        for i in range(n_points):
            G_subdiv.add_node(i)
            weights_dict[i] = [0] + list(weights[i,:])
            for r in range(K):
                j = indices[i, r]
                if i < j:
                    G_subdiv.add_node(j)
                    weights_dict[j] = [0] + list(weights[j,:])
                    r_ij = rank[i].get(j, np.inf)
                    r_ji = rank[j].get(i, np.inf)
                    k = min(r_ij, r_ji)
                    G_subdiv.add_node(new_vertex)
                    G_subdiv.add_edge(i,new_vertex)
                    G_subdiv.add_edge(j,new_vertex)
                    weights_dict[new_vertex] = [k] + list(np.maximum(weights[i,:], weights[j,:]))
                    new_vertex += 1
        weights_subdiv = np.zeros([np.max(np.array(list(weights_dict.keys())))+1, 1+weights.shape[1]])
        #print(weights_subdiv.shape, G.number_of_nodes(), G.number_of_edges())
        for k,v in weights_dict.items():
            #print(k,v)
            weights_subdiv[int(k),:] = np.array(v)
        return G_subdiv, weights_subdiv

    def __init__(self, direction=None, slice_number=10, merging_threshold=0.1, n_clusters=None, bounding_box=None, 
                 sigma2=1e-10, rescale=False, scale_filts=0.1, mode='radius', verbose=False):

        self.direction = direction
        self.slice_number = slice_number
        self.merging_threshold = merging_threshold
        self.n_clusters = n_clusters
        self.bounding_box = bounding_box
        self.sigma2 = sigma2
        self.most_common_dict_ = None
        self.labels_ = None
        self.rescale = rescale
        self.scale_filts = np.array(scale_filts)
        self.mode = mode
        self.verbose = verbose

    def fit(self, X, weights):
        
        self.n_parameters = weights.shape[1]

        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)

        if self.sigma2 > 0.0:
            noise = np.random.uniform(0, self.sigma2, size=weights.shape)
            weights_noise = weights + noise
        else:
            weights_noise = weights

        # Graph input
        if isinstance(X, nx.Graph) or self.mode == 'neighbors':

            if self.mode == 'neighbors':
                G_subdiv, weights_subdiv = self.compute_knn_graph_weights(X, weights_noise, self.bounding_box[1,0])
                self.graph, weights_noise = G_subdiv, weights_subdiv
                self.n_parameters += 1
            else:
                self.graph = X

            missing_mins = np.argwhere(np.isnan(self.bounding_box[0,:])).flatten()
            missing_maxs = np.argwhere(np.isnan(self.bounding_box[1,:])).flatten()
            self.bounding_box[0, missing_mins] = weights_noise[:, missing_mins].min(axis=0)
            self.bounding_box[1, missing_maxs] = weights_noise[:, missing_maxs].max(axis=0)

            if self.verbose:
                print("Input: Graph")

            if self.direction is None:
                self.direction = np.ones(self.n_parameters)

            if self.rescale:
                wmax = np.max(weights_noise, axis=0).reshape(1,-1)
                wmin = np.min(weights_noise, axis=0).reshape(1,-1)
                weights_noise = np.multiply(self.scale_filts.reshape(1, -1), (weights_noise - wmin) / (wmax - wmin))
                self.bounding_box = np.multiply(self.scale_filts.reshape(1, -1), (self.bounding_box - wmin) / (wmax - wmin))
            
            #plt.figure()
            #plt.scatter(weights_noise[:,0], weights_noise[:,1], c='blue', s=10, alpha=1)
            #plt.title("Graph input - Weights after noise and rescaling")
            #plt.show()

            st_multi = mp.SimplexTreeMulti(num_parameters=self.n_parameters)
            for node in self.graph.nodes():
                multi_filt = list(weights_noise[node])
                st_multi.insert([node], multi_filt)
            for edge in self.graph.edges():
                multi_filt = [max(weights_noise[edge[0]][d], weights_noise[edge[1]][d]) for d in range(self.n_parameters)]
                st_multi.insert(list(edge), multi_filt)
            
            self.mma = mp.module_approximation(st_multi, direction=self.direction, box=[list(self.bounding_box[0,:]), list(self.bounding_box[1,:])], nlines=self.slice_number)

        # Point cloud input
        else:

            self.n_parameters += 1
            missing_mins = np.argwhere(np.isnan(self.bounding_box[0,1:])).flatten()
            missing_maxs = np.argwhere(np.isnan(self.bounding_box[1,1:])).flatten()
            self.bounding_box[0, missing_mins+1] = weights_noise[:, missing_mins].min(axis=0)
            self.bounding_box[1, missing_maxs+1] = weights_noise[:, missing_maxs].max(axis=0)

            if self.verbose:
                print("Input: Point cloud")

            if self.direction is None:
                self.direction = np.ones(self.n_parameters)

            if self.rescale:
                wmax = np.max(weights_noise, axis=0).reshape(1,-1)
                wmin = np.min(weights_noise, axis=0).reshape(1,-1)
                weights_noise = np.multiply(self.scale_filts[1:].reshape(1,-1), (weights_noise - wmin) / (wmax - wmin))
                self.bounding_box[:,1:] = np.multiply(self.scale_filts[1:].reshape(1, -1), (self.bounding_box[:,1:] - wmin) / (wmax - wmin))

                #self.basecoords = self.scale_weights*(np.zeros(self.n_parameters) - wmin) / (wmax - wmin)
                X = X * self.scale_filts[0]
                self.bounding_box[0,0] *= self.scale_filts[0]
                self.bounding_box[1,0] *= self.scale_filts[0]

            st = gd.RipsComplex(points=X, max_edge_length=self.bounding_box[1,0]).create_simplex_tree(max_dimension=1)

            st_multi = mp.SimplexTreeMulti(st, num_parameters=self.n_parameters)
            if self.n_parameters <= 2:
                st_multi.collapse_edges(-2)
            for i in range(weights_noise.shape[1]):
                w = np.array(weights_noise[:, i], dtype=np.float64, copy=True)
                st_multi.fill_lowerstar(w, parameter=i+1)
            #print(f"Number of simplices: {st_multi.num_simplices}")

            self.mma = mp.module_approximation(st_multi, direction=self.direction, box=[self.bounding_box[0,:], self.bounding_box[1,:]], nlines=self.slice_number)

            #self.X_subdiv = X.copy()
            weights_noise_subdiv = np.zeros(shape=[st.num_simplices(), self.n_parameters])
            self.graph = nx.Graph()
            new_vertex = len(X)
            for splx, rips_filt in st.get_skeleton(0):
                vertex = splx[0]
                self.graph.add_node(vertex)
                multi_filt = [rips_filt] + list(weights_noise[splx[0],:])
                weights_noise_subdiv[splx[0],:] = np.array(multi_filt)
            for splx, rips_filt in st.get_skeleton(1):
                if len(splx) == 2:
                    multi_filt = [rips_filt] + list(np.maximum(weights_noise[splx[0],:], weights_noise[splx[1],:]))
                    self.graph.add_node(new_vertex)
                    self.graph.add_edge(splx[0], new_vertex)
                    self.graph.add_edge(splx[1], new_vertex)
                    weights_noise_subdiv[new_vertex,:] = np.array(multi_filt)
                    new_vertex += 1
                    #self.X_subdiv = np.vstack([self.X_subdiv, X[splx].mean(axis=0).reshape(1,-1)])
                else:
                    continue
            weights_noise = weights_noise_subdiv

        self.lower_corner = self.bounding_box[0,:]

        def process_slice(elm, dimension=0):
            return tomato_graph_slice(graph=self.graph, module=self.mma, weights=weights_noise, direction=self.direction, 
                                      basepoint=list(self.lower_corner[:dimension]) + [elm] + list(self.lower_corner[dimension+1:]), 
                                      merging_threshold=self.merging_threshold, 
                                      n_clusters=self.n_clusters).colored_clusters 
                                      #n_clusters=self.n_clusters, X=self.X_subdiv).colored_clusters

        if self.verbose:
            print(f"Slicing in the interval [{self.bounding_box[0,0]},{self.bounding_box[1,0]}]")

        results = []
        for dim in range(self.n_parameters-1):
            if self.verbose:
                print(f"Processing dimension {dim+1}/{self.n_parameters-1}...")
            grid = np.linspace(self.bounding_box[0,dim], self.bounding_box[1,dim], self.slice_number)
            for elm in grid:
                results.append(process_slice(elm, dimension=dim))

        freqs = defaultdict(list)
        for res in results:
            for pt_id, cluster in res.items():
                freqs[pt_id].append(cluster)

        self.most_common_dict_ = {
            pt_id: Counter(clusters).most_common(1)[0][0] if clusters else None
            for pt_id, clusters in freqs.items()
        }

        self.labels_ = [self.most_common_dict_[k] for k in sorted(self.most_common_dict_.keys())[:len(X)]]

        return self

