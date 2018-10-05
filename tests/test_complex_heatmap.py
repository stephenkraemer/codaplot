#-
import subprocess
from os.path import abspath
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from complex_heatmap.heatmap import find_stretches
from complex_heatmap import GridElement as GE
# from complex_heatmap.heatmap import align
from numpy.ma.testutils import assert_array_equal
from scipy.cluster.hierarchy import linkage, fcluster
import complex_heatmap as ch

import matplotlib.pyplot as plt
import seaborn as sns
#-

rng = np.random.RandomState(1)
data = np.tile([5, 10], (5, 5)).T + np.tile(np.arange(0, 20, 4), (10, 1))
names = 's' + pd.Series(np.arange(5)).astype(str)
df1 = pd.DataFrame(data + rng.randn(10, 5), columns=names)
df2 = pd.DataFrame(data + rng.randn(10, 5), columns=names)
df3 = pd.DataFrame(rng.randn(10, 5) * 3 + 1, columns=names)

def test_plot_panel():
    def fn(x):
        plt.plot(x)
    class CustomPlot(ch.heatmap.ClusterProfilePlotPanel):
        plotter = staticmethod(fn)
        align_vars = ['data']
    CustomPlot(x = [1, 2]).plot()

def test_agg_line():
    pass

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
                        ch.heatmap.Heatmap(cmap='YlOrBr'),
                        ch.heatmap.Heatmap(df=df2, cmap='RdBu_r')
                    ],
                    # [
                    #     ch.heatmap.AggLine(cluster_ids, 'mean'),
                    #     ch.heatmap.AggLine(df=df2, cluster_ids, 'mean'),
                    #
                    # ]
                          ],
                row_dendrogram=True,
                col_dendrogram=True,
                row_annotation=pd.DataFrame({'cluster_ids1': np.repeat(np.arange(5), 2),
                                             'cluster_ids2': np.repeat(np.arange(2), 5),
                                             }),
                row_anno_heatmap_args={'colors': [(1, 1, 1), (.5, .5, .5)],
                               'show_values':    True},
                row_anno_col_width = 1/2.54,
                figsize=(10/2.54, 5/2.54),
                fig_args = dict(dpi=180)
        )
    gm.create_or_update_figure()
    gm.fig.savefig('test.png')
    subprocess.run(['firefox', abspath('test.png')])


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








