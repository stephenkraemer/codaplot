#-
import subprocess
import time
from os.path import expanduser
from pathlib import Path

import complex_heatmap.clustered_data_grid
import complex_heatmap.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from numpy.testing import assert_array_equal
from pandas.api.types import CategoricalDtype

import complex_heatmap as ch

#-

# Test data setup
# ######################################################################
# Create 'base' test data ndarray with three clusters across rows and
# three clusters across columns.
# The rows and cols of each cluster are not adjacent and need to be
# correctly arranged by applying clustering.
# Before usage, we'll likely want to add some noise to this template
from complex_heatmap.plotting import cluster_size_plot

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

timestamp = time.strftime('%d-%m-%y_%H-%M-%S')
output_dir = Path(expanduser(f'~/temp/complex_heatmap_{timestamp}'))
output_dir.mkdir(exist_ok=True)

def test_heatmap_grids():

    local_output_dir = output_dir / 'heatmap_grids'
    local_output_dir.mkdir(exist_ok=True)

    plot_count = 0


    profile_plot = (
        ch.ClusteredDataGrid(main_df=df1)
            .cluster_rows(usecols=['s0', 's1'])
            .cluster_cols()
    )

    # one row of heatmaps
    # ===================

    # vanilla
    # -------
    gm = profile_plot.plot_grid(
            grid=[[
                complex_heatmap.clustered_data_grid.Heatmap(df=df1, cmap='YlOrBr'),
                complex_heatmap.clustered_data_grid.Heatmap(df=df2, cmap='RdBu_r'),
                complex_heatmap.clustered_data_grid.Heatmap(df=df3, cmap='RdBu_r'),
            ]],
            row_dendrogram=False,
            col_dendrogram=False,
            figsize=(20/2.54, 15/2.54),
            fig_args = dict(dpi=180)
    )
    gm.create_or_update_figure()
    plot_count += 1
    gm.fig.savefig(local_output_dir / f'test_{plot_count}.png')

    # with dendrograms
    # ----------------
    gm = profile_plot.plot_grid(
            grid=[[
                complex_heatmap.clustered_data_grid.Heatmap(df=df1, cmap='YlOrBr'),
                complex_heatmap.clustered_data_grid.Heatmap(df=df2, cmap='RdBu_r'),
                complex_heatmap.clustered_data_grid.Heatmap(df=df3, cmap='RdBu_r'),
            ]],
            row_dendrogram=True,
            col_dendrogram=True,
            figsize=(20/2.54, 15/2.54),
            fig_args = dict(dpi=180)
    )
    gm.create_or_update_figure()
    plot_count += 1
    gm.fig.savefig(local_output_dir / f'test_{plot_count}.png')

    # dendrograms plus annotation, with colors and text
    # -------------------------------------------------
    gm = profile_plot.plot_grid(
            grid=[[
                complex_heatmap.clustered_data_grid.Heatmap(df=df1, cmap='YlOrBr'),
                complex_heatmap.clustered_data_grid.Heatmap(df=df2, cmap='RdBu_r'),
                complex_heatmap.clustered_data_grid.Heatmap(df=df3, cmap='RdBu_r'),
            ]],
            row_dendrogram=True,
            col_dendrogram=True,
            row_annotation=cluster_ids_row_df,
            row_anno_heatmap_args={'colors': [(.8, .8, .8), (.5, .5, .5)],
                                   'show_values': True},
            row_anno_col_width = 1/2.54,
            figsize=(20/2.54, 15/2.54),
            fig_args = dict(dpi=180)
    )
    gm.create_or_update_figure()
    plot_count += 1
    gm.fig.savefig(local_output_dir / f'test_{plot_count}.png')

    # simple grid
    gm = profile_plot.plot_grid(
            grid=[
                [
                    complex_heatmap.clustered_data_grid.Heatmap(df=df1, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.Heatmap(df=df2, cmap='RdBu_r'),
                ],
                [
                    complex_heatmap.clustered_data_grid.Heatmap(df=df3, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.Heatmap(df=df4, cmap='RdBu_r'),
                ],
                [
                    complex_heatmap.clustered_data_grid.Heatmap(df=df5, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.Heatmap(df=df6, cmap='RdBu_r'),
                ],
            ],
            row_dendrogram=True,
            col_dendrogram=True,
            row_annotation=cluster_ids_row_df,
            row_anno_heatmap_args={'colors': [(.8, .8, .8), (.5, .5, .5)],
                                   'show_values': True},
            row_anno_col_width = 1/2.54,
            figsize=(20/2.54, 15/2.54),
            fig_args = dict(dpi=180)
    )
    gm.create_or_update_figure()
    plot_count += 1
    gm.fig.savefig(local_output_dir / f'test_{plot_count}.png')

    # with different style
    with sns.plotting_context('paper', font_scale=0.6), \
         sns.axes_style('darkgrid'):
        # simple grid
        gm = profile_plot.plot_grid(
                grid=[
                    [
                        complex_heatmap.clustered_data_grid.Heatmap(df=df1, cmap='YlOrBr'),
                        complex_heatmap.clustered_data_grid.Heatmap(df=df2, cmap='RdBu_r'),
                    ],
                    [
                        complex_heatmap.clustered_data_grid.Heatmap(df=df3, cmap='YlOrBr'),
                        complex_heatmap.clustered_data_grid.Heatmap(df=df4, cmap='RdBu_r'),
                    ],
                    [
                        complex_heatmap.clustered_data_grid.Heatmap(df=df5, cmap='YlOrBr'),
                        complex_heatmap.clustered_data_grid.Heatmap(df=df6, cmap='RdBu_r'),
                    ],
                ],
                row_dendrogram=True,
                col_dendrogram=True,
                row_annotation=cluster_ids_row_df,
                row_anno_heatmap_args={'colors': [(.8, .8, .8), (.5, .5, .5)],
                                       'show_values': True},
                row_anno_col_width = 1/2.54,
                figsize=(20/2.54, 15/2.54),
                fig_args = dict(dpi=180)
        )
        gm.create_or_update_figure()
        plot_count += 1
        gm.fig.savefig(local_output_dir / f'test_{plot_count}.png')

    subprocess.run(['firefox', output_dir.absolute()])

    # with plt.rc_context({'xtick.bottom': False, 'ytick.left': False,
    #                      'xtick.major.size': 0, 'xtick.minor.size': 0,
    #                      'ytick.major.size': 0, 'ytick.minor.size': 0,
    #                      }):


def test_cluster_size_anno():
    profile_plot = (
        ch.ClusteredDataGrid(main_df=df1)
            .cluster_rows(usecols=['s0', 's1'])
            .cluster_cols()
    )

    gm = profile_plot.plot_grid(
            grid=[
                [
                    complex_heatmap.clustered_data_grid.Heatmap(df=df1, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.Heatmap(panel_width=2, panel_kind='rel', df=df2, cmap='RdBu_r'),
                    complex_heatmap.clustered_data_grid.ClusterSizePlot(
                            panel_width=2/2.54,
                            panel_kind='abs',
                            cluster_ids=cluster_ids_row_df.iloc[:, 0],
                            bar_height=0.1,
                            xlabel='#Elements',
                    )
                ],
                [
                    complex_heatmap.clustered_data_grid.ColAggPlot(df=df1, fn=np.mean, xlabel='Mean'),
                    complex_heatmap.clustered_data_grid.ColAggPlot(
                            panel_width=2, panel_kind='rel',
                            df=df2, fn=np.mean, xlabel='Mean'),
                    ch.dynamic_grid.Spacer(width=2/2.54, kind='abs')
                ],
            ],
            height_ratios=[(1, 'rel'), (2.5/2.54, 'abs')],
            row_dendrogram=True,
            col_dendrogram=True,
            row_annotation=cluster_ids_row_df,
            row_anno_heatmap_args={'colors': [(.8, .8, .8), (.5, .5, .5)],
                                   'show_values': True},
            row_anno_col_width = 1/2.54,
            figsize=(20/2.54, 12/2.54),
            fig_args = dict(dpi=180)
    )
    gm.create_or_update_figure()
    fp = output_dir / f'cluster-size-plot.png'
    gm.fig.savefig(fp)
    subprocess.run(['firefox', fp])


def test_merge_grid_element_across_rows():
    profile_plot = (
        ch.ClusteredDataGrid(main_df=df1)
            .cluster_rows(usecols=['s0', 's1'])
            .cluster_cols()
    )

    gm = profile_plot.plot_grid(
            grid=[
                [
                    complex_heatmap.clustered_data_grid.Heatmap(name='h1', df=df1, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.ColAggPlot(
                            panel_width=2/2.54, panel_kind='abs',
                            df=df1, fn=np.mean, xlabel='Mean'),
                ],
                [
                    complex_heatmap.clustered_data_grid.Heatmap(name='h1', df=df1, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.ClusterSizePlot(
                            panel_width=2/2.54,
                            panel_kind='abs',
                            cluster_ids=cluster_ids_row_df.iloc[:, 0],
                            bar_height=0.1,
                            xlabel='#Elements',
                    )
                ],
            ],
            height_ratios=[(1, 'rel'), (2.5/2.54, 'abs')],
            row_dendrogram=True,
            col_dendrogram=True,
            row_annotation=cluster_ids_row_df,
            row_anno_heatmap_args={'colors': [(.8, .8, .8), (.5, .5, .5)],
                                   'show_values': True},
            row_anno_col_width = 1/2.54,
            figsize=(20/2.54, 12/2.54),
            fig_args = dict(dpi=180)
    )
    gm.create_or_update_figure()
    fp = output_dir / f'heatmap-merged-cols-plot.png'
    gm.fig.savefig(fp)
    subprocess.run(['firefox', fp])

@pytest.mark.slow()
def test_complex_grid_with_heatmaps_and_deco_plots():

    profile_plot = (
        ch.ClusteredDataGrid(main_df=df1)
            .cluster_rows(usecols=['s0', 's1'])
            .cluster_cols()
    )

    plot_count = 0
    gm = profile_plot.plot_grid(
            grid=[
                [
                    complex_heatmap.clustered_data_grid.Heatmap(df=df1, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.Heatmap(df=df2, cmap='RdBu_r'),
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                ],
                [
                    complex_heatmap.clustered_data_grid.Heatmap(df=df3, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.Heatmap(df=df4, cmap='RdBu_r'),
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                ],
                [
                    complex_heatmap.clustered_data_grid.Heatmap(df=df5, cmap='YlOrBr'),
                    complex_heatmap.clustered_data_grid.Heatmap(df=df6, cmap='RdBu_r'),
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                ],
                [
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                ],
                [
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                ],
                [
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                    complex_heatmap.clustered_data_grid.SimpleLine(),
                ]
            ],
            row_dendrogram=True,
            col_dendrogram=True,
            row_annotation=cluster_ids_row_df,
            row_anno_heatmap_args={'colors': [(.8, .8, .8), (.5, .5, .5)],
                                   'show_values': True},
            row_anno_col_width = 1/2.54,
            figsize=(20/2.54, 15/2.54),
            fig_args = dict(dpi=180)
    )
    gm.create_or_update_figure()
    plot_count += 1
    fp = output_dir / f'complex-grid_{plot_count}.png'
    gm.fig.savefig(fp)

    # subprocess.run(['firefox', tmpdir.absolute()])
    subprocess.run(['firefox', fp])


class TestClusterProfilePlotPanel:
    def test_raises_when_plotter_is_not_staticmethod(self):
        def fn(x):
            plt.plot(x)
        class CustomPlot(ch.clustered_data_grid.ClusteredDataGridElement):
            plotter = fn
            align_vars = ['data']
        with pytest.raises(TypeError):
            CustomPlot()


def test_find_stretches():
    df = pd.DataFrame({'strict': [1, 1, 2, 2, 3, 3],
                       'less strict': [1, 2, 3, 4, 5, 6],
                       })
    stretches = [complex_heatmap.plotting.find_stretches(df[colname].values) for colname in df]

    assert_array_equal(stretches[0][0], np.array([1, 3, 5], dtype='f8'))
    assert_array_equal(stretches[0][1], np.array([1, 2, 3], dtype='i8'))
    assert_array_equal(stretches[1][0], np.array([.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype='f8'))
    assert_array_equal(stretches[1][1], np.array([1, 2, 3, 4, 5, 6], dtype='i8'))

    # for timing
    # ser = pd.Series(np.repeat(np.arange(1000), 1000))

def test_categorical_heatmap():

    # White and gray alternating running IDs
    colors = [(1, 1, 1), (0.6, 0.6, 0.6)]
    df = pd.DataFrame({'strict': [1, 1, 2, 2, 3, 3],
                       'less strict': [1, 2, 3, 4, 5, 6],
                       })

    fig, ax = plt.subplots(1, 1)
    complex_heatmap.plotting.categorical_heatmap(df, ax, colors=colors, show_values=True,
                                                 show_legend=False, despine=False)
    fp = output_dir / 'categorical-heatmap_two-colors.png'
    fig.savefig(fp)
    subprocess.run(['firefox', fp])

    # Set1 based categorical
    cat_type = CategoricalDtype(categories=list('dceabf'), ordered=True)

    df = pd.DataFrame({'varA': list('abc'),
                       'varB': list('efa'),
                       'varC': list('ddd')}, dtype=cat_type)

    fig, ax = plt.subplots(1, 1)
    complex_heatmap.plotting.categorical_heatmap(df, ax, cmap='Set1', show_values=True,
                                                 show_legend=True, despine=False)
    fp = output_dir / 'categorical-heatmap_set1.png'
    fig.savefig(fp)
    subprocess.run(['firefox', fp])

    with pytest.raises(ValueError):
        df = pd.DataFrame({'a': [np.nan, 1]})
        fig, ax = plt.subplots(1, 1)
        complex_heatmap.plotting.categorical_heatmap(df, ax=ax)

    with pytest.raises(ValueError):
        df = pd.DataFrame({'a': [1, 1],
                           'b': ['a', 'b']})
        fig, ax = plt.subplots(1, 1)
        complex_heatmap.plotting.categorical_heatmap(df, ax=ax)

def test_cluster_size_plot():
    rng = np.random.RandomState(1)
    cluster_ids = pd.Series(rng.choice([1, 2, 3], 20, replace=True))
    fig, ax = plt.subplots(1, 1)
    cluster_size_plot(cluster_ids, ax=ax, xlabel='Cluster sizes')
    fp = output_dir / 'cluster-size-plot.png'
    fig.savefig(fp)
    subprocess.run(['firefox', fp])
