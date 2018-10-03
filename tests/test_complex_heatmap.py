#-
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from complex_heatmap.heatmap import find_stretches
from numpy.ma.testutils import assert_array_equal
from scipy.cluster.hierarchy import linkage, fcluster

import complex_heatmap as ch

import matplotlib.pyplot as plt
#-

rng = np.random.RandomState(1)
data = np.tile([5, 10], (5, 5)).T + np.tile(np.arange(0, 20, 4), (10, 1))
names = 's' + pd.Series(np.arange(5)).astype(str)
df1 = pd.DataFrame(data + rng.randn(10, 5), columns=names)
df2 = pd.DataFrame(data + rng.randn(10, 5), columns=names)
df3 = pd.DataFrame(rng.randn(10, 5) * 3 + 1, columns=names)

def test_clustering_within_complex_heatmap():
    complex_heatmap_list = ch.ComplexHeatmapList(
            [
                ch.ComplexHeatmap(df1,
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
                ch.ComplexHeatmap(df2, cmap_sequential='Blues',
                                  title='dtype2'
                                  ),
                ch.ComplexHeatmap(df3, cmap_norm='midpoint',
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

    complex_heatmap_list = ch.ComplexHeatmapList(
            [
                ch.ComplexHeatmap(df1,
                                  is_main=True,
                                  cmap_sequential='YlOrBr',
                                  row_linkage_matrix=row_Z,
                                  col_linkage_matrix=col_Z,
                                  row_anno=cluster_ids,
                                  col_dendrogram_height=1/2.54,
                                  row_dendrogram_width=1/2.54,
                                  title='dtype1'
                                  ),
                ch.ComplexHeatmap(df2,
                                  cmap_sequential='Blues',
                                  title='dtype2'
                                  ),
                ch.ComplexHeatmap(df3,
                                  cmap_norm='midpoint',
                                  title='dtype3'
                                  ),
            ],
            figsize=(12/2.54, 6/2.54),
            dpi=360
    )
    fig = complex_heatmap_list.plot()

    #-

def test_row_annotation():
    row_Z = linkage(df1)
    col_Z = linkage(df1.T)
    cluster_ids = pd.DataFrame(fcluster(row_Z, t=3, criterion='maxclust'))

    complex_heatmap_list = ch.ComplexHeatmapList(
            [
                ch.ComplexHeatmap(df1,
                                  is_main=True,
                                  cmap_sequential='YlOrBr',
                                  row_linkage_matrix=row_Z,
                                  col_linkage_matrix=col_Z,
                                  row_anno=cluster_ids,
                                  col_dendrogram_height=1/2.54,
                                  row_dendrogram_width=1/2.54,
                                  title='dtype1'
                                  ),
                ch.ComplexHeatmap(df2,
                                  cmap_sequential='Blues',
                                  title='dtype2'
                                  ),
                ch.ComplexHeatmap(df3,
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









