#-
import subprocess
from os.path import abspath
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from complex_heatmap.heatmap import find_stretches, agg_line
from complex_heatmap import GridElement as GE
# from complex_heatmap.heatmap import align
from numpy.ma.testutils import assert_array_equal
from scipy.cluster.hierarchy import linkage, fcluster
import complex_heatmap as ch

import matplotlib.pyplot as plt
import seaborn as sns
#-

# Create 'base' test data ndarray with three clusters across rows and
# three clusters across columns.
# The rows and cols of each cluster are not adjacent and need to be
# correctly arranged by applying clustering.
# Before usage, we'll likely want to add some noise to this template
rows_with_three_different_levels = np.tile([20, 30, 10], (5, 10)).T
# array([[20, 20, 20, 20, 20],
#        [30, 30, 30, 30, 30],
#        [10, 10, 10, 10, 10],
#        [20, 20, 20, 20, 20],
#        ...
cols_with_three_different_levels = np.tile([5, 10, 5, 15, 10], (30, 1))
# array([[ 5, 10,  5, 15, 10],
#        [ 5, 10,  5, 15, 10],
#        ...
data = rows_with_three_different_levels + cols_with_three_different_levels
cluster_ids_row_df = pd.DataFrame({'strict': np.tile([2, 3, 1], 10).T,
                                   'liberal':  np.tile([1, 2, 1], 10).T})

# Create different version of the test ndarray, with different noise
# and as DataFrames
rng = np.random.RandomState(1)
names = 's' + pd.Series(np.arange(5)).astype(str)
std =  1
df1 = pd.DataFrame(data + rng.randn(30, 5) * std, columns=names)
df2 = pd.DataFrame(data + rng.randn(30, 5) * std, columns=names)
df3 = pd.DataFrame(data + rng.randn(30, 5) * std, columns=names)
df4 = pd.DataFrame(data + rng.randn(30, 5) * std, columns=names)
df5 = pd.DataFrame(data + rng.randn(30, 5) * std, columns=names)
df6 = pd.DataFrame(data + rng.randn(30, 5) * std, columns=names)
# df7 = pd.DataFrame(rng.randn(10, 5) * 3 + 1, columns=names)


def test_heatmap_grid():

    profile_plot = (
        ch.ClusterProfilePlot(main_df=df1)
            .cluster_rows(usecols=['s0', 's1'])
            .cluster_cols()
    )
    with sns.plotting_context('paper', font_scale=0.6):
        gm = profile_plot.plot_grid(
                old_grid=[
                    [
                        ch.heatmap.Heatmap(df=df1, cmap='YlOrBr'),
                        ch.heatmap.Heatmap(df=df2, cmap='RdBu_r'),
                    ],
                    [
                        ch.heatmap.Heatmap(df=df3, cmap='YlOrBr'),
                        ch.heatmap.Heatmap(df=df4, cmap='RdBu_r'),
                    ],
                    [
                        ch.heatmap.Heatmap(df=df5, cmap='YlOrBr'),
                        ch.heatmap.Heatmap(df=df6, cmap='RdBu_r'),
                    ],
                ],
                row_dendrogram=True,
                col_dendrogram=True,
                row_annotation=cluster_ids_row_df,
                row_anno_heatmap_args={'colors': [(.8, .8, .8), (.5, .5, .5)],
                                       'show_values':    True},
                row_anno_col_width = 1/2.54,
                figsize=(20/2.54, 15/2.54),
                fig_args = dict(dpi=180)
        )
        gm.create_or_update_figure()
        gm.fig.savefig('test.png')
    subprocess.run(['firefox', abspath('test.png')])



def test_plot_panel():
    def fn(x):
        plt.plot(x)
    class CustomPlot(ch.heatmap.ClusterProfilePlotPanel):
        plotter = staticmethod(fn)
        align_vars = ['data']
    CustomPlot(x = [1, 2]).plot()

def test_agg_line():


    fig = plt.figure(constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0, wspace=0)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlabel('test')
    fig.add_subplot(gs[1, 0])
    fig.add_subplot(gs[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    agg_line(df=df1, ax=ax, fig=fig,
             cluster_ids=np.repeat([1, 2], 5), fn=np.mean)
    fig.show()


    # fig = plt.figure(constrained_layout=True)
    # fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=0, wspace=0)
    # gs = fig.add_gridspec(4, 2)
    # ax = fig.add_subplot(gs[0:2, 1])
    # ax.set_xlabel('test')
    # fig.add_subplot(gs[2:, 0])
    # fig.add_subplot(gs[2:, 1])
    # ax = fig.add_subplot(gs[0, 0])
    # ax.set(xticks=[], xticklabels=[])
    # ax = fig.add_subplot(gs[1, 0])
    # ax.set_xlabel('test\nnew')
    # fig.show()
    #

    # align, supply functions to warp arguments
    # pass list with variables to be supplied and list for alignments to ClusterProfilePlot or to the plotting functions
    # align list in ClusterProfilePlot init could be overwritten with align_targets in PE, eg align_targets = False, or align_targets = ['other']
    # would also also leaving align targets empty during init and using it in a targeted fashion
    # need show rows as well as show columns

    # method to remove clusters and get subsequent cluster numbers again
    # method to merge clusters

    # method: add titles


    #-
# noinspection PyUnresolvedReferences
def api_design():

    complex_heatmap_list = (
        ch.ClusterProfilePlot(main_df=df1)
            .cluster_rows()
            .cluster_cols()
            .plot_heatmap(cmap_sequential='YlOrBr', title='dtype1',
                          row_dendrogram=True, col_dendrogram=True,
                          fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360)
                          )
    )

    cluster_ids = pd.DataFrame
    row_Z = np.ndarray
    custom_fn = lambda x: x

    complex_heatmap_list = (
        ch.ClusterProfilePlot(main_df=df1, cluster_ids = cluster_ids,
                              row_linkage=row_Z)
            .plot_violins(row_dendrogram=True,
                          fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360))
    )

    complex_heatmap_list = (
        ch.ClusterProfilePlot(main_df=df1, cluster_ids = cluster_ids,
                              row_linkage=row_Z)
            .plot_violins(compact=True, nrow=3,
                          fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360))
    )


    complex_heatmap_list = (
        ch.ClusterProfilePlot(main_df=df1, cluster_ids = cluster_ids)
            .plot_layers(ch.Violin(cmap_sequential='YlOrBr', title='dtype1'),
                         ch.AggLines(stats='mean'),
                         compact=True, nrow=3)
    )

    complex_heatmap_list = (
        ch.ClusterProfilePlot(main_df=df1, row_anno=ch.AnnoBar(df3))
            .cluster_rows()
            .plot_grid([[ch.Heatmap(df1), (ch.Violin(df1), ch.AggLines(df1))],
                        [ch.Heatmap(df2), ch.AggLines(df2)]],
                       row_dendrogram=True, col_dendrogram=True,
                       fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360))
    )

    complex_heatmap_list = (
        ch.ClusterProfilePlot(main_df=df1, cluster_ids = cluster_ids,
                              row_anno=custom_fn)
            .plot_grid(
                [[ch.Heatmap(df1),
                  custom_fn], # custom fn takes ax and is not aligned
                 [ch.Heatmap(df2),
                  ch.AlignedPlot(custom_fn, align_targets=['df'], **other_args)] ],
                fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360))
    )

    # add dendrograms manually add arbitrary locations
    # leave {row,col}_dendrogram=False
    complex_heatmap_list = (
        ch.ClusterProfilePlot(main_df=df1,
                              cluster_ids = cluster_ids, row_anno=ch.AnnoBar(df3),
                              )
            .plot_grid([[ch.Heatmap(df1), ch.Dendrogram(), (ch.Violin(df1), ch.AggLines(df1))],
                        [ch.Heatmap(df2), ch.Dendrogram(), ch.AggLines(df2)]
                        ],
            fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360))
    )


    complex_heatmap_list = (
        ch.ClusterProfilePlot(
                main_df=df1, supp_main_df=[df2, df3],
                cluster_ids = cluster_ids, row_linkage = row_Z,
                row_anno=ch.AnnoBar(df3))
            .plot_expanded_to_grid(
                [ch.Heatmap(), ch.AggLines()],
                row_dendrogram=True, col_dendrogram=True,
                fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360))
    )


    # Downsampling
    cluster_profile_plot = (
        ch.ClusterProfilePlot(main_df=df1)
        .cluster_rows()
        .cutree_hybrid('name1')
        .cluster_sample_absolute('name1', n_total=5000, strict=True)
        .plot_grid([ch.Heatmap(), ch.AggLines()])
    )


    #-


"""

def test_clustering_within_complex_heatmap():
    complex_heatmap_list = ch.ClusterProfilePlot(
            [
                ch.Heatmap(df1,
                           is_main=True,
                           cluster_use_cols = ['s1', 's2'],
                           cmap_sequential='YlOrBr',
                           cluster_cols = False,
                           col_show_list = ['s1', 's2', 's3'],
                           col_dendrogram_height=1/2.54,
                           row_dendrogram_width=1/2.54,
                           col_dendrogram_show=False,
                           title='dtype1'
                           ),
                ch.Heatmap(df2, cmap_sequential='Blues',
                           title='dtype2'
                           ),
                ch.Heatmap(df3, cmap_norm='midpoint',
                           title='dtype3'
                           ),
            ],
            figsize=(12/2.54, 6/2.54),
            dpi=360
    )
    fig = complex_heatmap_list.plot()
    
def test_clustering_outside_of_complex_heatmap():

    row_Z = linkage(df1)
    col_Z = linkage(df1.T)
    cluster_ids = pd.DataFrame(fcluster(row_Z, t=3, criterion='maxclust'))

    complex_heatmap_list = ch.ClusterProfilePlot(
            plots= [
                ch.Heatmap(df1,
                           cmap_sequential='YlOrBr',
                           title='dtype1'
                           ),
                ch.Heatmap(df2,
                           cmap_sequential='Blues',
                           title='dtype2'
                           ),
                ch.Heatmap(df3,
                           cmap_norm='midpoint',
                           title='dtype3'
                          ),
            ],
            row_Z = row_Z,
            col_Z = col_Z,
            fig_args = dict(figsize=(12/2.54, 6/2.54), dpi=360)
    )

    fig = complex_heatmap_list.plot()

    #-

def test_row_annotation():
    row_Z = linkage(df1)
    col_Z = linkage(df1.T)
    cluster_ids = pd.DataFrame(fcluster(row_Z, t=3, criterion='maxclust'))

    complex_heatmap_list = ch.ClusterProfilePlot(
            [
                ch.Heatmap(df1,
                           is_main=True,
                           cmap_sequential='YlOrBr',
                           row_linkage_matrix=row_Z,
                           col_linkage_matrix=col_Z,
                           row_anno=cluster_ids,
                           col_dendrogram_height=1/2.54,
                           row_dendrogram_width=1/2.54,
                           title='dtype1'
                           ),
                ch.Heatmap(df2,
                           cmap_sequential='Blues',
                           title='dtype2'
                           ),
                ch.Heatmap(df3,
                           cmap_norm='midpoint',
                           title='dtype3'
                           ),
            ],
            figsize=(12/2.54, 6/2.54),
            dpi=360
    )
    fig = complex_heatmap_list.plot()

def test_find_stretches():
    df = pd.DataFrame({'strict': [1, 1, 2, 2, 3, 3],
                       'less strict': [1, 2, 3, 4, 5, 6],
                       })
    stretches = [ch.find_stretches(df[colname]) for colname in df]

    assert_array_equal(stretches[0][0], np.array([1, 3, 5], dtype='f8'))
    assert_array_equal(stretches[0][1], np.array([1, 2, 3], dtype='i8'))
    assert_array_equal(stretches[1][0], np.array([.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype='f8'))
    assert_array_equal(stretches[1][1], np.array([1, 2, 3, 4, 5, 6], dtype='i8'))

    # for timing
    # ser = pd.Series(np.repeat(np.arange(1000), 1000))

def test_categorical_heatmap(tmpdir):
    tmpdir = Path(tmpdir)

    # White and gray alternating running IDs
    colors = [(1, 1, 1), (0.6, 0.6, 0.6)]
    df = pd.DataFrame({'strict': [1, 1, 2, 2, 3, 3],
                       'less strict': [1, 2, 3, 4, 5, 6],
                       })

    fig, ax = plt.subplots(1, 1)
    ch.categorical_heatmap(df, ax, colors=colors, show_values=True,
                           show_legend=False, despine=False)
    fig.savefig(tmpdir / 'test.png')

    # Set1 based categorical
    cat_type = CategoricalDtype(categories=list('dceabf'), ordered=True)

    df = pd.DataFrame({'varA': list('abc'),
                       'varB': list('efa'),
                       'varC': list('ddd')}, dtype=cat_type)

    fig, ax = plt.subplots(1, 1)
    ch.categorical_heatmap(df, ax, cmap='Set1', show_values=True,
                           show_legend=True, despine=False)
    fig.savefig(tmpdir / 'test.png')

    # requires same dtype and no NAs

    with pytest.raises(ValueError):
        df = pd.DataFrame({'a': [np.nan, 1]})
        fig, ax = plt.subplots(1, 1)
        ch.categorical_heatmap(df, ax=ax)

    with pytest.raises(ValueError):
        df = pd.DataFrame({'a': [1, 1],
                           'b': ['a', 'b']})
        fig, ax = plt.subplots(1, 1)
        ch.categorical_heatmap(df, ax=ax)
"""








