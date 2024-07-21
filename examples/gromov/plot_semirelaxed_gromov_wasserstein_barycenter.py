# -*- coding: utf-8 -*-

r"""
=====================================================
Semi-relaxed (Fused) Gromov-Wasserstein Barycenter as Dictionary Learning
=====================================================

In this example, we illustrate how to learn a semi-relaxed Gromov-Wasserstein
(srGW) barycenter using a Block-Coordinate Descent algorithm, on a dataset of
structured data such as graphs, denoted :math:`\{ \mathbf{C_s} \}_{s \in [S]}`
where every nodes have uniform weights :math:`\{ \mathbf{p_s} \}_{s \in [S]}`.
Given a barycenter structure matrix :math:`\mathbf{C}` with N nodes,
each graph :math:`(\mathbf{C_s}, \mathbf{p_s})` is modeled as a reweighed subgraph
with structure :math:`\mathbf{C}` and weights :math:`\mathbf{w_s} \in \Sigma_N`
where each :math:`\mathbf{w_s}` corresponds to the second marginal of the OT
:math:`\mathbf{T_s}` (s.t :math:`\mathbf{w_s} = \mathbf{T_s}^\top \mathbf{1}`)
minimizing the srGW loss between the s^{th} input and the barycenter.


First, we consider a dataset composed of graphs generated by Stochastic Block models
with variable sizes taken in :math:`\{30, ... , 50\}` and quantities of clusters
varying in :math:`\{ 1, 2, 3\}`. We learn a srGW barycenter with 6 nodes and
visualize the learned structure and the embeddings for some inputs.

Second, we illustrate the extension of this framework to graphs endowed
with node features by using the semi-relaxed Fused Gromov-Wasserstein
divergence (srFGW). Starting from the aforementioned dataset of unattributed graphs, we
add discrete labels uniformly depending on the number of clusters. Then conduct
the analog analysis.


[48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
"Semi-relaxed Gromov-Wasserstein divergence and applications on graphs".
International Conference on Learning Representations (ICLR), 2022.

"""
# Author: Cédric Vincent-Cuaz <cedric.vincent-cuaz@inria.fr>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pylab as pl
from sklearn.manifold import MDS
from ot.gromov import (
    semirelaxed_gromov_barycenters, semirelaxed_fgw_barycenters)
import ot
import networkx
from networkx.generators.community import stochastic_block_model as sbm

#############################################################################
#
# Generate a dataset composed of graphs following Stochastic Block models of 1, 2 and 3 clusters.
# -----------------------------------------------------------------------------------------------

np.random.seed(42)

n_samples = 60  # number of graphs in the dataset
# For every number of clusters, we generate SBM with fixed inter/intra-clusters probability.
clusters = [1, 2, 3]
Nc = n_samples // len(clusters)  # number of graphs by cluster
nlabels = len(clusters)
dataset = []
labels = []

p_inter = 0.1
p_intra = 0.9
for n_cluster in clusters:
    for i in range(Nc):
        n_nodes = int(np.random.uniform(low=30, high=50))

        if n_cluster > 1:
            P = p_inter * np.ones((n_cluster, n_cluster))
            np.fill_diagonal(P, p_intra)
        else:
            P = p_intra * np.eye(1)
        sizes = np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
        G = sbm(sizes, P, seed=i, directed=False)
        C = networkx.to_numpy_array(G)
        dataset.append(C)
        labels.append(n_cluster)


# Visualize samples

def plot_graph(x, C, binary=True, color='C0', s=None):
    for j in range(C.shape[0]):
        for i in range(j):
            if binary:
                if C[i, j] > 0:
                    pl.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=0.2, color='k')
            else:  # connection intensity proportional to C[i,j]
                pl.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], alpha=C[i, j], color='k')

    pl.scatter(x[:, 0], x[:, 1], c=color, s=s, zorder=10, edgecolors='k', cmap='tab10', vmax=9)


pl.figure(1, (12, 8))
pl.clf()
for idx_c, c in enumerate(clusters):
    C = dataset[(c - 1) * Nc]  # sample with c clusters
    # get 2d position for nodes
    x = MDS(dissimilarity='precomputed', random_state=0).fit_transform(1 - C)
    pl.subplot(2, nlabels, c)
    pl.title('(graph) sample from label ' + str(c), fontsize=14)
    plot_graph(x, C, binary=True, color='C0', s=50.)
    pl.axis("off")
    pl.subplot(2, nlabels, nlabels + c)
    pl.title('(matrix) sample from label %s \n' % c, fontsize=14)
    pl.imshow(C, interpolation='nearest')
    pl.axis("off")
pl.tight_layout()
pl.show()

#############################################################################
#
# Estimate the Gromov-Wasserstein dictionary from the dataset
# -----------------------------------------------------------


np.random.seed(0)
ps = [ot.unif(C.shape[0]) for C in dataset]  # uniform weights on nodes
lambdas = [1. / n_samples for _ in range(n_samples)]  # uniform barycenter
N = 3  # 3 nodes in the barycenter

# Optionally provide an initial barycenter structure `init_C`

init_C = np.array([[0.6, 0.2, 0.2],
                   [0.2, 0.6, 0.2],
                   [0.2, 0.2, 0.6]])

print('init_C:', init_C)

C, log = semirelaxed_gromov_barycenters(
    N=N, Cs=dataset, ps=ps, lambdas=lambdas, loss_fun='square_loss', tol=1e-6,
    stop_criterion='loss', warmstartT=True, log=True, init_C=init_C)

# visualize loss evolution over epochs
pl.figure(2, (4, 3))
pl.clf()
pl.title('loss evolution by iteration', fontsize=14)
pl.plot(log['loss'])
pl.xlabel('BCD iterations', fontsize=12)
pl.ylabel('loss', fontsize=12)
pl.tight_layout()
pl.show()

print('C:', C)
