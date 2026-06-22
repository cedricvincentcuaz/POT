# -*- coding: utf-8 -*-
"""
======================================
Optimal Transport Barycenter solvers comparison
======================================

This example illustrates solutions returned for different variants of exact,
regularized and unbalanced OT barycenter problems with free support using our wrapper `ot.solve_bary_sample`.
"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 2

# %%

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.plot import plot2D_samples_mat

# %%
# 2D data example
# ---------------
#
# We first generate two sets of samples in 2D of 8 and 16
# points uniformly separated on circles. The weights of the samples are uniform.

# Problem size
n1, n2 = 8, 16
nbary = 12

# Generate random data
np.random.seed(0)

r1, r2 = 1, 3
x1 = r1 * np.array(
    [(np.cos(2 * i * np.pi / n1), np.sin(2 * i * np.pi / n1)) for i in range(n1)]
)

x2 = r2 * np.array(
    [(np.cos(2 * i * np.pi / n2), np.sin(2 * i * np.pi / n2)) for i in range(n2)]
)

style = {"markeredgecolor": "k"}

pl.figure(1, (4, 4))
pl.plot(x1[:, 0], x1[:, 1], "ob", **style)
pl.plot(x2[:, 0], x2[:, 1], "or", **style)
pl.title("Source distributions")
pl.show()


# %%
# Set up parameters for balanced OT barycenter solvers and solve
# ---------------------------------------

# balanced OT
lst_balanced_solvers = [  # name, param for ot.solve function
    ("Exact OT", dict()),
    ("Entropic Reg. OT", dict(reg=1.0)),
]

lst_balanced_res = []
for name, param in lst_balanced_solvers:
    print(f"-- name = {name} / param = {param}")
    res = ot.solve_bary_sample(X_a_list=[x1, x2], n=nbary, **param)
    lst_balanced_res.append(res)
    list_P = [res.list_res[k].plan for k in range(2)]
    print("X:", res.X)
    print("loss:", res.value)
    print("loss:", res.log)
    print(
        "marginals OT 1:",
        res.list_res[0].plan.sum(axis=1),
        res.list_res[0].plan.sum(axis=0),
    )
    print(
        "marginals OT 2:",
        res.list_res[1].plan.sum(axis=1),
        res.list_res[1].plan.sum(axis=0),
    )

##############################################################################
# Plot distributions and plans for balanced OT barycenter solvers
# ----------


def plot_list_res(lst_res, lst_solvers, fig_num=1, n_cols=2, mass_difference=False):
    n_plots = len(lst_res)
    n_rows = int(np.ceil(n_plots / n_cols))
    pl.figure(fig_num, figsize=(16, n_rows * 4))
    style.update({"markersize": 20})
    for i, (name, param) in enumerate(lst_solvers):
        pl.subplot(n_rows, n_cols, i + 1)
        X = lst_res[i].X
        list_P = [lst_res[i].list_res[k].plan for k in range(2)]
        loss = lst_res[i].value

        plot2D_samples_mat(x1, X, list_P[0])
        plot2D_samples_mat(x2, X, list_P[1])

        if i == 0:  # add labels
            pl.plot(x1[:, 0], x1[:, 1], "ob", label="Source distribution 1", **style)
            pl.plot(x2[:, 0], x2[:, 1], "or", label="Source distribution 2", **style)
            pl.plot(X[:, 0], X[:, 1], "og", label="Barycenter distribution", **style)
            pl.legend(loc="best")
        else:
            pl.plot(x1[:, 0], x1[:, 1], "ob", **style)
            pl.plot(x2[:, 0], x2[:, 1], "or", **style)
            pl.plot(X[:, 0], X[:, 1], "og", **style)

        pl.title(name)


plot_list_res(lst_balanced_res, lst_balanced_solvers, fig_num=2, n_cols=2)


# %%
# Set up parameters for unbalanced OT barycenter solvers and solve
# ---------------------------------------

lambda_unbalanced_vals = [1, 5, 10]

# unbalanced OT KL
lst_unbalanced_solvers = [
    (
        "Unbalanced KL No Reg \n" + r"$\lambda_u$=%s" % lambda_val,
        dict(unbalanced=lambda_val),
    )
    for lambda_val in lambda_unbalanced_vals
] + [
    (
        r"Unbalanced KL with KL Reg \n"
        + r"$\lambda_u$=%s, $\lambda_{ent}$=%s" % (lambda_val, 0.1),
        dict(reg=0.1, unbalanced=lambda_val, unbalanced_type="kl", reg_type="kl"),
    )
    for lambda_val in lambda_unbalanced_vals
]

lst_unbalanced_res = []
for name, param in lst_unbalanced_solvers:
    print(f"-- name = {name} / param = {param}")
    res = ot.solve_bary_sample(X_a_list=[x1, x2], n=nbary, **param)
    lst_unbalanced_res.append(res)
    list_P = [res.list_res[k].plan for k in range(2)]
    print("X:", res.X)
    print("loss:", res.value)
    print("loss:", res.log)
    print(
        "marginals OT 1:",
        res.list_res[0].plan.sum(axis=1),
        res.list_res[0].plan.sum(axis=0),
    )
    print(
        "marginals OT 2:",
        res.list_res[1].plan.sum(axis=1),
        res.list_res[1].plan.sum(axis=0),
    )

##############################################################################
# Plot distributions and plans for unbalanced OT barycenter solvers
# ----------

plot_list_res(lst_unbalanced_res, lst_unbalanced_solvers, fig_num=3, n_cols=3)
