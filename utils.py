import os
import glob
import io
import math
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx
import gudhi as gd
import multipers as mp

from itertools import combinations
from scipy.sparse import csc_array
from scipy.stats import spearmanr as _spearmanr

from tqdm import tqdm

from gudhi.clustering.tomato import Tomato
from tomatomp import Tomatomp

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

from persist_spatial.network_functions import get_distances
from persist_spatial.smoothed_expression import distance_to_measure_weighted
from persist_spatial.topology_utils import p_norm, diagram_to_array, function_filtration
from persist_spatial.post_ph_functions import find_knee

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def smoothed_expression(df, mesh_type="hexagonal", m=0.1):

    df.fillna(0)
    expression_array = normalize(np.array(df.iloc[:,2:]), axis=0, norm="l1")
    expression_sparse = csc_array(expression_array)

    co_ordinates = np.array(df.iloc[:,:2])
    num_wells = df.shape[0]
    dmat_ut = np.zeros((num_wells, num_wells))
    dmat = np.zeros((num_wells, num_wells))
    for i in range(num_wells):
        dmat[i,:] = np.sqrt( ((co_ordinates[i,0] - co_ordinates[:,0])**2 + (co_ordinates[i,1] - co_ordinates[:,1])**2) )
        dmat_ut[i,i:] = np.sqrt( ((co_ordinates[i,0] - co_ordinates[i:,0])**2 + (co_ordinates[i,1] - co_ordinates[i:,1])**2) )
    dunique = np.unique(dmat)
    well_sep = dunique[dunique>0].min()

    if mesh_type=="hexagonal":
        dthresh = (0.5*(1+np.sqrt(3))) * well_sep
    elif mesh_type=="square":
        dthresh = (0.5*(1+np.sqrt(2))) * well_sep

    grid_distances = get_distances(num_wells, well_sep, mesh_type)
    dmat_comp = np.zeros((num_wells, num_wells))
    for i in range(dmat.shape[0]):
        dmat_comp[i,dmat[i,:].argsort()] = grid_distances

    edges_to_add = np.where((dmat_ut>0) & (dmat_ut <= dthresh))
    edges = np.zeros((edges_to_add[0].size, 2))
    edges[:,0] = np.array(edges_to_add[0])
    edges[:,1] = np.array(edges_to_add[1])
    edges = edges.astype(int)

    st = gd.SimplexTree()
    for edge in edges:
        st.insert(edge)

    A = np.zeros((st.num_vertices(), st.num_vertices()))
    for splx in st.get_skeleton(1):
        if len(splx[0]) == 2:
            A[splx[0][0], splx[0][1]] = 1
            A[splx[0][1], splx[0][0]] = 1
    G = nx.from_numpy_array(A)
    list_neighbors = [list(G.neighbors(i)) for i in range(A.shape[0])]

    gene_list = df.columns[2:]
    compute_ph = np.where(expression_sparse.sum(axis=0)>0)[0]

    dtms = []
    for gene in compute_ph:
        expressed_wells = expression_array[:,gene]>0
        weights = expression_sparse[expressed_wells,[gene]].flatten()
        indices = (-weights).argsort()
        weights = weights[indices]
        dists = dmat_comp[:,expressed_wells][:,indices]
        dtm = distance_to_measure_weighted(weights=weights, dmat=dists, m=m)
        dtms.append((gene, gene_list[gene], dtm))

    return dtms, st, A, G, list_neighbors

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def rank_genes_tomato(dtms, list_neighbors, merge_threshold=None):

    scores = []
    diagrams = {}

    for (_, gene, dtm) in dtms:

        tomato = Tomato(graph_type='manual', density_type='manual', merge_threshold=merge_threshold)
        tomato.fit(X=list_neighbors, weights=dtm)

        dgm = np.asarray(tomato.diagram_, dtype=np.float64)
        diagrams[gene] = dgm

        if dgm.size == 0:
            scores.append(0.0)
            continue

        scores.append(p_norm(dgm, p=2))

    gene_cols = [gene for (_, gene, _) in dtms]
    ranking = pd.DataFrame({'gene': gene_cols, 'score': scores})
    return ranking

def jaccard_score(labels1, labels2):

    clusters1 = np.unique(labels1)
    clusters2 = np.unique(labels2)

    jaccard = np.zeros((len(clusters1), len(clusters2)))
    pops1, pops2 = [], []
    weights1, weights2 = [], []
    for idx1, clus1 in enumerate(clusters1):
        pop1 = np.argwhere(labels1 == clus1).flatten()
        pops1.append(pop1)
        weights1.append(len(pop1)/len(labels1))
    for idx2, clus2 in enumerate(clusters2):
        pop2 = np.argwhere(labels2 == clus2).flatten()
        pops2.append(pop2)
        weights2.append(len(pop2)/len(labels2))

    for idx1, clus1 in enumerate(clusters1):
        pop1 = pops1[idx1]
        for idx2, clus2 in enumerate(clusters2):
            pop2 = pops2[idx2]

            intersection = len(np.intersect1d(pop1, pop2))
            union = len(np.union1d(pop1, pop2))
            if union > 0:
                jaccard[idx1, idx2] = intersection / union
            else:
                jaccard[idx1, idx2] = 0.0
    
    return (np.multiply(jaccard.max(axis=1), weights1)).sum() + (np.multiply(jaccard.max(axis=0), weights2)).sum()

def rank_pair_genes_tomato(dtms, list_neighbors, merge_threshold=None):

    #labels = {}
    #for (_, gene, dtm) in dtms:
    #    tomato = Tomato(graph_type='manual', density_type='manual', merge_threshold=merge_threshold)
    #    tomato.fit(X=list_neighbors, weights=dtm)
    #    labels[gene] = tomato.labels_

    scores = []
    for idx1, (_, gene1, dtm1) in enumerate(dtms):
        for (_, gene2, dtm2) in dtms[idx1+1:]:
            
            #print(f"Computing Jaccard score for {gene1} and {gene2}...")
            #labels1 = labels[gene1]
            #labels2 = labels[gene2]
            #scores.append(((gene1, gene2), jaccard_score(labels1, labels2)))
            
            tomato = Tomato(graph_type='manual', density_type='manual', merge_threshold=merge_threshold)
            tomato.fit(X=list_neighbors, weights=np.array(dtm1) + np.array(dtm2))
            scores.append(((gene1, gene2), p_norm(tomato.diagram_, p=2)))

    pair_genes = [gene1 + ' --- ' + gene2 for ((gene1, gene2), _) in scores]
    ranking = pd.DataFrame({'gene_pair': pair_genes, 'score': [score for (_, score) in scores]})
    return ranking

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def mma_score(mma, minf, maxf, direction, nlines=100, mode=1, quant=0.5, plot_dgms=False):

    if mode == 1:
        scores = []

    summand_bars = {}
    #for idx_bp, basepoint in enumerate([(minf[0], f) for f in np.linspace(minf[1], maxf[1], num=nlines)] + [(f, minf[1]) for f in np.linspace(minf[0], maxf[0], num=nlines)]):
    for idx_bp, basepoint in enumerate([[f] + list(minf[1:]) for f in np.linspace(minf[0], maxf[0], num=nlines)]):
        barcode_raw = mma.barcode2(basepoint, direction, degree=0, full=False, threshold=False, keep_inf=True) 
        
        if mode == 1:
            scores.append(p_norm(barcode_raw[0], p=2))

        if plot_dgms and idx_bp <= 2:
            plt.figure()
            plt.scatter([a for (a,b) in barcode_raw[0]], [b for (a,b) in barcode_raw[0]])
            plt.title(f"Basepoint {idx_bp}")
            plt.xlabel('Birth')
            plt.ylabel('Death')
            plt.show()

        if mode == 2:
            for i, (a, b) in enumerate(barcode_raw[0]):
                if idx_bp == 0:
                    summand_bars[i] = []
                if not (math.isinf(a) or math.isinf(b)):
                    summand_bars[i].append((a,b))

    if mode == 2:
        scores = []
        for i in summand_bars.keys():
            if len(summand_bars[i]) == 0:
                scores.append([0., 0.])
                continue
            sorted_lengths = np.argsort([np.abs(b-a) for (a,b) in summand_bars[i]])
            picked_idx = sorted_lengths[int(quant*len(sorted_lengths))]
            a, b = summand_bars[i][picked_idx]
            scores.append(np.array([a,b]))
        scores = np.array(scores)
        return p_norm(scores, p=2)
    else:
        return np.quantile(scores, quant)

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def rank_genes_tomatomp_radius(dtms, coords, direction=(1., 1.), nlines=100, min_edge_length=None, max_edge_length=None, rescale=True):

    dtm_global_min = min([(-dtm).min() for (_, _, dtm) in dtms])

    scores_mod1, scores_mod2, scores_mod3, scores_mod4, scores_mod5, scores_mod6 = [], [], [], [], [], []
    for (_, gene, dtm) in dtms:

        weights = -dtm.copy()
        wmin = weights.min()
        tomatomp = Tomatomp(
            direction=direction,
            slice_number=nlines,
            bounding_box=np.array([[min_edge_length, np.nan], [max_edge_length, np.nan]]),
            merging_threshold=None,
            n_clusters=None,
            sigma2=0.,
            rescale=rescale, 
            scale_filts=[10, 1.*wmin/dtm_global_min],
        )
        tomatomp.fit(coords, weights=weights[:,None])
        scores_mod1.append((gene, mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.1, plot_dgms=False)))
        scores_mod2.append((gene, mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.5, plot_dgms=False)))
        scores_mod3.append((gene, mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.9, plot_dgms=False)))
        scores_mod4.append((gene, mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.1, plot_dgms=False)))
        scores_mod5.append((gene, mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.5, plot_dgms=False)))
        scores_mod6.append((gene, mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.9, plot_dgms=False)))

    result_mod1 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod1], 'score': [score for (_, score) in scores_mod1]})
    result_mod2 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod2], 'score': [score for (_, score) in scores_mod2]})
    result_mod3 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod3], 'score': [score for (_, score) in scores_mod3]})
    result_mod4 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod4], 'score': [score for (_, score) in scores_mod4]})
    result_mod5 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod5], 'score': [score for (_, score) in scores_mod5]})
    result_mod6 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod6], 'score': [score for (_, score) in scores_mod6]})
    return result_mod1, result_mod2, result_mod3, result_mod4, result_mod5, result_mod6

def rank_pair_genes_tomatomp_radius(dtms, coords, direction=(1., 1., 1.), nlines=100, min_edge_length=None, max_edge_length=None, rescale=True):

    scores_mod1, scores_mod2, scores_mod3, scores_mod4, scores_mod5, scores_mod6 = [], [], [], [], [], []
    global_min = min([(-dtm).min() for (_, _, dtm) in dtms])

    for idx1, (_, gene1, dtm1) in enumerate(dtms):
        wmin1 = (-dtm1).min()
        for (_, gene2, dtm2) in dtms[idx1+1:]:
            wmin2 = (-dtm2).min()

            weights = np.hstack([-dtm1.copy()[:,None], -dtm2.copy()[:,None]])
            tomatomp = Tomatomp(
                direction=direction,
                slice_number=nlines,
                bounding_box=np.array([[min_edge_length, np.nan, np.nan], [max_edge_length, np.nan, np.nan]]),
                merging_threshold=None,
                n_clusters=None,
                sigma2=0.,
                rescale=rescale, 
                scale_filts=[10, 1.*(wmin1/global_min), 1.*(wmin2/global_min)],
            )
            tomatomp.fit(coords, weights=weights)
            scores_mod1.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.1, plot_dgms=False)))
            scores_mod2.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.5, plot_dgms=False)))
            scores_mod3.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.9, plot_dgms=False)))
            scores_mod4.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.1, plot_dgms=False)))
            scores_mod5.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.5, plot_dgms=False)))
            scores_mod6.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.9, plot_dgms=False)))

    result_mod1 = pd.DataFrame({'gene_pair': [f"{gene1} --- {gene2}" for ((gene1, gene2), _) in scores_mod1], 'score': [score for (_, score) in scores_mod1]})
    result_mod2 = pd.DataFrame({'gene_pair': [f"{gene1} --- {gene2}" for ((gene1, gene2), _) in scores_mod2], 'score': [score for (_, score) in scores_mod2]})
    result_mod3 = pd.DataFrame({'gene_pair': [f"{gene1} --- {gene2}" for ((gene1, gene2), _) in scores_mod3], 'score': [score for (_, score) in scores_mod3]})
    result_mod4 = pd.DataFrame({'gene_pair': [f"{gene1} --- {gene2}" for ((gene1, gene2), _) in scores_mod4], 'score': [score for (_, score) in scores_mod4]})
    result_mod5 = pd.DataFrame({'gene_pair': [f"{gene1} --- {gene2}" for ((gene1, gene2), _) in scores_mod5], 'score': [score for (_, score) in scores_mod5]})
    result_mod6 = pd.DataFrame({'gene_pair': [f"{gene1} --- {gene2}" for ((gene1, gene2), _) in scores_mod6], 'score': [score for (_, score) in scores_mod6]})
    return result_mod1, result_mod2, result_mod3, result_mod4, result_mod5, result_mod6

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def rank_genes_tomatomp_outlier(dtms, G, list_neighbors, direction=(1., 1.), nlines=500, rescale=True):
        
    scores_mod1, scores_mod2, scores_mod3, scores_mod4, scores_mod5, scores_mod6 = [], [], [], [], [], []
    dtm_global_min = min([(-dtm).min() for (_, _, dtm) in dtms])

    for (_, gene, dtm) in dtms:
        wmin = (-dtm).min()
        outlier_score = np.array([float(np.mean(np.abs(dtm[i] - dtm[list_neighbors[i]]))) if list_neighbors[i] else 0.0 for i in range(len(list_neighbors))], dtype=np.float64)
        weights = np.column_stack([outlier_score, -dtm.copy()])
        model = Tomatomp(
                direction=direction,
                slice_number=nlines,
                bounding_box=np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                merging_threshold=None,
                n_clusters=None,
                sigma2=0.0,
                rescale=rescale,
                scale_filts=[10., 1.*wmin/dtm_global_min],
                verbose=False,
        )
        model.fit(G, weights=weights)
        scores_mod1.append((gene, mma_score(model.mma, model.bounding_box[0,:], model.bounding_box[1,:], model.direction, model.slice_number, mode=1, quant=0.1, plot_dgms=False)))
        scores_mod2.append((gene, mma_score(model.mma, model.bounding_box[0,:], model.bounding_box[1,:], model.direction, model.slice_number, mode=1, quant=0.5, plot_dgms=False)))
        scores_mod3.append((gene, mma_score(model.mma, model.bounding_box[0,:], model.bounding_box[1,:], model.direction, model.slice_number, mode=1, quant=0.9, plot_dgms=False)))
        scores_mod4.append((gene, mma_score(model.mma, model.bounding_box[0,:], model.bounding_box[1,:], model.direction, model.slice_number, mode=2, quant=0.1, plot_dgms=False)))
        scores_mod5.append((gene, mma_score(model.mma, model.bounding_box[0,:], model.bounding_box[1,:], model.direction, model.slice_number, mode=2, quant=0.5, plot_dgms=False)))
        scores_mod6.append((gene, mma_score(model.mma, model.bounding_box[0,:], model.bounding_box[1,:], model.direction, model.slice_number, mode=2, quant=0.9, plot_dgms=False)))

    result_mod1 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod1], 'score': [score for (_, score) in scores_mod1]})
    result_mod2 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod2], 'score': [score for (_, score) in scores_mod2]})
    result_mod3 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod3], 'score': [score for (_, score) in scores_mod3]})
    result_mod4 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod4], 'score': [score for (_, score) in scores_mod4]})
    result_mod5 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod5], 'score': [score for (_, score) in scores_mod5]})
    result_mod6 = pd.DataFrame({'gene': [gene for (gene, _) in scores_mod6], 'score': [score for (_, score) in scores_mod6]})
    return result_mod1, result_mod2, result_mod3, result_mod4, result_mod5, result_mod6

def rank_pair_genes_tomatomp_outlier(dtms, G, list_neighbors, direction=(1., 1., 1.), nlines=500, rescale=True):
        
    scores_mod1, scores_mod2, scores_mod3, scores_mod4, scores_mod5, scores_mod6 = [], [], [], [], [], []
    global_min = min([(-dtm).min() for (_, _, dtm) in dtms])

    for idx1, (_, gene1, dtm1) in enumerate(dtms):
        outlier_score1 = np.array([float(np.mean(np.abs(dtm1[i] - dtm1[list_neighbors[i]]))) if list_neighbors[i] else 0.0 for i in range(len(list_neighbors))], dtype=np.float64)
        wmin1 = (-dtm1).min()
        for (_, gene2, dtm2) in dtms[idx1+1:]:
            outlier_score2 = np.array([float(np.mean(np.abs(dtm2[i] - dtm2[list_neighbors[i]]))) if list_neighbors[i] else 0.0 for i in range(len(list_neighbors))], dtype=np.float64)
            wmin2 = (-dtm2).min()

            outlier_score = outlier_score1 + outlier_score2
            weights = np.column_stack([outlier_score, -dtm1.copy(), -dtm2.copy()])
            tomatomp = Tomatomp(
                    direction=direction,
                    slice_number=nlines,
                    bounding_box=np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
                    merging_threshold=None,
                    n_clusters=None,
                    sigma2=0.,
                    rescale=rescale,
                    scale_filts=[10., 1.*(wmin1/global_min), 1.*(wmin2/global_min)],
                    verbose=False,
            )
            tomatomp.fit(G, weights=weights)
            scores_mod1.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.1, plot_dgms=False)))
            scores_mod2.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.5, plot_dgms=False)))
            scores_mod3.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=1, quant=0.9, plot_dgms=False)))
            scores_mod4.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.1, plot_dgms=False)))
            scores_mod5.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.5, plot_dgms=False)))
            scores_mod6.append(((gene1, gene2), mma_score(tomatomp.mma, tomatomp.bounding_box[0,:], tomatomp.bounding_box[1,:], tomatomp.direction, tomatomp.slice_number, mode=2, quant=0.9, plot_dgms=False)))

    result_mod1 = pd.DataFrame({'gene_pair': [gene1 + ' --- ' + gene2 for ((gene1, gene2), _) in scores_mod1], 'score': [score for (_, score) in scores_mod1]})
    result_mod2 = pd.DataFrame({'gene_pair': [gene1 + ' --- ' + gene2 for ((gene1, gene2), _) in scores_mod2], 'score': [score for (_, score) in scores_mod2]})
    result_mod3 = pd.DataFrame({'gene_pair': [gene1 + ' --- ' + gene2 for ((gene1, gene2), _) in scores_mod3], 'score': [score for (_, score) in scores_mod3]})
    result_mod4 = pd.DataFrame({'gene_pair': [gene1 + ' --- ' + gene2 for ((gene1, gene2), _) in scores_mod4], 'score': [score for (_, score) in scores_mod4]})
    result_mod5 = pd.DataFrame({'gene_pair': [gene1 + ' --- ' + gene2 for ((gene1, gene2), _) in scores_mod5], 'score': [score for (_, score) in scores_mod5]})
    result_mod6 = pd.DataFrame({'gene_pair': [gene1 + ' --- ' + gene2 for ((gene1, gene2), _) in scores_mod6], 'score': [score for (_, score) in scores_mod6]})

    return result_mod1, result_mod2, result_mod3, result_mod4, result_mod5, result_mod6

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def rank_genes_hierarchical(dtms, coords, add_outlier_score=False, list_neighbors=None):

    coords_to_use = coords.copy()
    spatial_std = coords_to_use.std()
    gene_cols = [gene for (_, gene, _) in dtms]
    scores = []

    for (_, _, dtm) in dtms:
        expr = -dtm.copy().astype(np.float64)
        if add_outlier_score:
            outlier_score = np.array([float(np.mean(np.abs(dtm[i] - dtm[list_neighbors[i]]))) if list_neighbors[i] else 0.0 for i in range(len(list_neighbors))], dtype=np.float64)
            outlier_score_std = outlier_score.std()
            if outlier_score_std > 0:
                outlier_score = outlier_score * (spatial_std / outlier_score_std)
            coords_to_use = np.column_stack([coords_to_use, outlier_score])
        expr_std = expr.std()
        if expr_std > 0:
            expr = expr * (spatial_std / expr_std)
        pts = np.column_stack([coords_to_use, expr])
        clus = AgglomerativeClustering(n_clusters=None, distance_threshold=0.0, linkage='single')
        clus.fit(pts)
        scores.append(np.linalg.norm(clus.distances_, ord=2))

    ranking = pd.DataFrame({'gene': gene_cols, 'score': scores})
    return ranking

def rank_pair_genes_hierarchical(dtms, coords, add_outlier_score=False, list_neighbors=None):

    coords_to_use = coords.copy()
    spatial_std = coords_to_use.std()

    scores = []
    for idx1, (_, gene1, dtm1) in enumerate(dtms):
        for (_, gene2, dtm2) in dtms[idx1+1:]:

            expr1 = -dtm1.copy().astype(np.float64)
            expr2 = -dtm2.copy().astype(np.float64)
            if add_outlier_score:
                outlier_score1 = np.array([float(np.mean(np.abs(dtm1[i] - dtm1[list_neighbors[i]]))) if list_neighbors[i] else 0.0 for i in range(len(list_neighbors))], dtype=np.float64)
                outlier_score2 = np.array([float(np.mean(np.abs(dtm2[i] - dtm2[list_neighbors[i]]))) if list_neighbors[i] else 0.0 for i in range(len(list_neighbors))], dtype=np.float64)
                outlier_score1_std = outlier_score1.std()
                outlier_score2_std = outlier_score2.std()
                if outlier_score1_std > 0:
                    outlier_score1 = outlier_score1 * (spatial_std / outlier_score1_std)
                if outlier_score2_std > 0:
                    outlier_score2 = outlier_score2 * (spatial_std / outlier_score2_std)
                coords_to_use = np.column_stack([coords_to_use, outlier_score1, outlier_score2])
            expr1_std = expr1.std()
            expr2_std = expr2.std()
            if expr1_std > 0:
                expr1 = expr1 * (spatial_std / expr1_std)
            if expr2_std > 0:
                expr2 = expr2 * (spatial_std / expr2_std)
            pts = np.column_stack([coords_to_use, expr1, expr2])
            clus = AgglomerativeClustering(n_clusters=None, distance_threshold=0., linkage='single')
            clus.fit(pts)
            scores.append(((gene1, gene2), np.linalg.norm(clus.distances_, ord=2)))

    ranking = pd.DataFrame({'gene_pair': [gene1 + ' --- ' + gene2 for (gene1, gene2), _ in scores], 'score': [score for _, score in scores]})
    return ranking


###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def spearman_corr(ground_truth_ranking, ranking):
    rk1 = ground_truth_ranking[['gene', 'score']].copy()
    rk1['ground_truth_rank'] = rk1['score'].rank(ascending=False)
    rk2 = ranking[['gene', 'score']].copy()
    rk2['rank'] = rk2['score'].rank(ascending=False)
    rho, pval = _spearmanr(rk2['rank'], rk1['ground_truth_rank'])    
    return rho, pval

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################



###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def read_off(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    if lines[0] != "OFF":
        if lines[0].startswith("OFF"):
            lines[0] = lines[0][3:].strip()
        else:
            raise ValueError(f"Unsupported OFF header in {path}")

    if lines[0] == "OFF":
        n_verts, n_faces, _ = map(int, lines[1].split()[:3])
        cursor = 2
    else:
        n_verts, n_faces, _ = map(int, lines[0].split()[:3])
        cursor = 1

    verts = np.array([
        list(map(float, lines[cursor + i].split()[:3]))
        for i in range(n_verts)
    ])
    cursor += n_verts

    faces = []
    for i in range(n_faces):
        toks = list(map(int, lines[cursor + i].split()))
        k = toks[0]
        idx = toks[1:1 + k]
        for t in range(1, k - 1):
            faces.append([idx[0], idx[t], idx[t + 1]])

    return verts, np.array(faces, dtype=int)

def hks(evals, evecs, t):
    return np.sum((evecs**2) * np.exp(-t * evals)[None, :], axis=1)

###########################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################

def correlation_1g(ranking1, ranking2):
    merged = pd.merge(ranking1, ranking2, on='gene', suffixes=('_1', '_2'))
    return np.corrcoef(merged['score_1'], merged['score_2'])[0, 1]

def top_hits_10_1g(ranking1, ranking2):
    top_genes1 = set(ranking1.sort_values('score', ascending=False).head(10)['gene'])
    top_genes2 = set(ranking2.sort_values('score', ascending=False).head(10)['gene'])
    return len(top_genes1.intersection(top_genes2)) / 10

def correlation_2g(ranking1, ranking2):
    merged = pd.merge(ranking1, ranking2, on='gene_pair', suffixes=('_1', '_2'))
    return np.corrcoef(merged['score_1'], merged['score_2'])[0, 1]

def top_hits_10_2g(ranking1, ranking2):
    top_genes1 = set(ranking1.sort_values('score', ascending=False).head(10)['gene_pair'])
    top_genes2 = set(ranking2.sort_values('score', ascending=False).head(10)['gene_pair'])
    return len(top_genes1.intersection(top_genes2)) / 10