from typing import List, Optional, Callable, Union, Sequence, Iterable, Dict

from dataclasses import dataclass
import matplotlib as mpl
from matplotlib import pyplot as plt, patches as mpatches
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
import numba
import pandas as pd
from pandas.core.groupby import GroupBy
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from codaplot.cluster_ids import ClusterIDs
from codaplot.linkage_mat import Linkage

CMAP_DICT = dict(
        divergent_meth_heatmap = plt.get_cmap('RdBu_r'),
        sequential_meth_heatmap = plt.get_cmap('YlOrBr'),
        cluster_id_to_color = dict(zip(np.arange(1,101),
                                       sns.color_palette('Set1', 100))),
)
CMAP_DICT['cluster_id_to_color'][-1] = (0,0,0)
CMAP_DICT['cluster_id_to_color'][0] = (0,0,0)

def remove_axis(ax: Axes, y=True, x=True):
    """Remove spines, ticks, labels, ..."""
    if y:
        ax.set(yticks=[], yticklabels=[])
        sns.despine(ax=ax, left=True)
    if x:
        ax.set(xticks=[], xticklabels=[])
        sns.despine(ax=ax, bottom=True)


def dendrogram_wrapper(linkage_mat, ax: Axes, orientation: str):
    """Wrapper around scipy dendrogram - nicer plot

    Despines plot and removes ticks and labels
    """
    dendrogram(linkage_mat, ax=ax, color_threshold=-1,
               above_threshold_color='black', orientation=orientation)
    ax.set(xticks=[], yticks=[], yticklabels=[], xticklabels=[])
    sns.despine(ax=ax, bottom=True, left=True)


def categorical_heatmap(df: pd.DataFrame, ax: Axes,
                        cmap: str = 'Set1', colors: Optional[List] = None,
                        show_values = True,
                        show_legend = False,
                        legend_ax: Optional[Axes] = None,
                        despine = True,
                        show_yticklabels=False,
                        show_xticklabels=False,
                        ):
    """Categorical heatmap

    Args:
        df: Colors will be assigned to df values in sorting order.
            Columns must have homogeneous types.
        cmap: a cmap name that sns.color_palette understands; ignored if
              colors are given
        colors: List of color specs

    Does not accept NA values.
    """

    if df.isna().any().any():
        raise ValueError('NA values not allowed')

    if not df.dtypes.unique().shape[0] == 1:
        raise ValueError('All columns must have the same dtype')

    is_categorical = df.dtypes.iloc[0].name == 'category'

    if is_categorical:
        levels = df.dtypes.iloc[0].categories.values
    else:
        levels = np.unique(df.values)
    n_levels = len(levels)

    if colors is None:
        color_list = sns.color_palette(cmap, n_levels)
    else:
        # tile colors to get n_levels color list
        color_list = (np.ceil(n_levels / len(colors)).astype(int) * colors)[:n_levels]
    cmap = mpl.colors.ListedColormap(color_list)


    # Get integer codes matrix for pcolormesh, ie levels are represented by
    # increasing integers according to the level ordering
    if is_categorical:
        codes_df = df.apply(lambda ser: ser.cat.codes, axis=0)
    else:
        value_to_code = {value: code for code, value in enumerate(levels)}
        codes_df = df.replace(value_to_code)

    qm = ax.pcolormesh(codes_df, cmap=cmap)
    if despine:
        sns.despine(ax=ax, bottom=True, left=True)
    if not show_xticklabels:
        ax.set(xticks=[], xticklabels=[])
    else:
        plt.setp(ax.get_xticklabels(), rotation=90)
    if not show_yticklabels:
        ax.set(yticks=[], yticklabels=[])

    # Create dummy patches for legend
    patches = [mpatches.Patch(facecolor=c, edgecolor='black') for c in color_list]
    if show_legend:
        if legend_ax is not None:
            legend_ax.legend(patches, levels)
        else:
            ax.legend(patches, levels)

    if show_values:
        for i in range(df.shape[1]):
            # note: categorical_series.values is not a numpy array
            # any series has __array__ method - so instead of values
            # attribute, use np.array
            y, s = find_stretches(np.array(df.iloc[:, i]))
            x = i + 0.5
            for curr_y, curr_s in zip(list(y), s):
                ax.text(x, curr_y, str(curr_s), va='center', ha='center')

    return {'quadmesh': qm, 'patches': patches, 'levels': levels}


# Note: jit compilation does not seem to provide speed-ups. Haven't
# checked it out yet.
numba.jit(nopython=True)
def find_stretches(arr):
    """Find stretches of equal values in ndarray"""
    assert isinstance(arr, np.ndarray)
    stretch_value = arr[0]
    start = 0
    end = 1
    marks = np.empty(arr.shape, dtype=np.float64)
    values = np.empty_like(arr)
    mark_i = 0
    for value in arr[1:]:
        if value == stretch_value:
            end += 1
        else:
            marks[mark_i] = start + (end - start)/2
            values[mark_i] = stretch_value
            start = end
            end += 1
            stretch_value = value
            mark_i += 1
    # add last stretch
    marks[mark_i] = start + (end - start)/2
    values[mark_i] = stretch_value
    return marks[:(mark_i + 1)], values[:(mark_i + 1)]


# From matplotlib docs:
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def heatmap(df: pd.DataFrame,
            ax: Axes, fig: Figure,
            cmap: str,
            midpoint_normalize: bool =  False,
            col_labels_show: bool = True,
            row_labels_show: bool = False,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            ):

    if midpoint_normalize:
        norm: Optional[mpl.colors.Normalize] = MidpointNormalize(
                vmin=df.min().min(), vmax=df.max().max(), midpoint=0)
    else:
        norm = None

    qm = ax.pcolormesh(df, cmap=cmap, norm=norm)

    if col_labels_show:
        ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        ax.set_xticklabels(df.columns, rotation=90)

    if row_labels_show:
        ax.set_yticklabels(df.index)
    else:
        ax.set_yticks([])

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    fig.colorbar(qm, ax=ax)


def simple_line(ax):
    ax.plot([1, 2, 3, 4, 5])


def cluster_size_plot(cluster_ids: pd.Series, ax: Axes,
                      bar_height = 0.6, xlabel=None):
    """Horizontal cluster size barplot

    Currently, this assumes that the cluster IDs can be sorted directly
    to get the correct ordering desired in the plot. This means that
    either:
    1. The cluster IDs are numerical OR
    2. The cluster IDs are ordered categorical OR
    3. The cluster IDs are strings, which can be ordered correctly *by
       alphabetic sorting*

    Args:
        cluster_ids: index does not have to be the same as for the dfs
            used in a ClusterProfile plot. This may change in the future.
            Best practice is to use the same index for the data and the cluster
            IDs.
        bar_height: between 0 and 1
        xlabel: x-Axis label for the plot
    """
    cluster_sizes = cluster_ids.value_counts()
    cluster_sizes.sort_index(inplace=True)
    ax.barh(y=np.arange(0.5, cluster_sizes.shape[0]),
            width=cluster_sizes.values,
            height=bar_height)
    ax.set_ylim(0, cluster_sizes.shape[0])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    remove_axis(ax, x=False)


def col_agg_plot(df: pd.DataFrame, fn: Callable, ax: Axes,
                 xlabel=None):
    """Plot aggregate statistic as line plot

    Args:
        fn: function which will be passed to pd.DataFrame.agg
    """
    agg_stat = df.agg(fn)
    ax.plot(np.arange(0.5, df.shape[1]),
            agg_stat.values)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def row_group_agg_plot(data: pd.DataFrame, fn, row: Optional[Union[str, Sequence]], ax: List[Axes],
                       marker='o', linestyle='-', color='black', linewidth=None,
                       show_all_xticklabels=False, xlabel=None,
                       sharey=True, ylim=None):

    axes = ax

    if isinstance(row, str):
        if row in data:
            levels = data[row].unique()
        elif row in data.index.names:
            levels = data.index.get_level_values(row).unique()
        else:
            raise ValueError()
    else:
        levels = pd.Series(row).unique()

    agg_values = data.groupby(row).agg(fn).loc[levels, :]
    if sharey and ylim is None:
        ylim = compute_padded_ylim(agg_values)

    ncol = data.shape[1]
    x = np.arange(0.5, ncol)
    xlim = (0, ncol)
    for curr_ax, (unused_name, agg_row) in zip(axes[::-1], agg_values.iterrows()):
        curr_ax.plot(x, agg_row.values, marker=marker, linestyle=linestyle,
                     linewidth=linewidth, color=color)
        curr_ax.set_xlim(xlim)
        if ylim is not None:
            curr_ax.set_ylim(ylim)
        if not show_all_xticklabels:
            curr_ax.set(xticks=[], xticklabels=[], xlabel='')
        else:
            curr_ax.set_xticks(x)
            curr_ax.set_xticklabels(data.columns, rotation=90)
            curr_ax.set_xlabel('')
        sns.despine(ax=curr_ax)

    # Add xticks and xticklabels to the bottom Axes
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(data.columns, rotation=90)
    axes[-1].set_xlabel(xlabel if xlabel is not None else '')


def compute_padded_ylim(df):
    ymax = np.max(df.values)
    pad = abs(0.1 * ymax)
    padded_ymax = ymax + pad
    if ymax < 0 < padded_ymax:
        padded_ymax = 0
    ymin = np.min(df.values)
    padded_ymin = ymin - pad
    if ymin > 0 > padded_ymin:
        padded_ymin = 0
    ylim = (padded_ymin, padded_ymax)
    return ylim


def grouped_rows_violin(data: pd.DataFrame, row: Union[str, Iterable],
                        ax: List[Axes], n_per_group=2000, sort=False,
                        sharey=False, ylim=None, show_all_xticklabels=False,
                        xlabel=None) -> None:
    # axes list called ax because currently only ax kwarg recognized by Grid
    # currently, axes are given in top down order, but we want to fill them
    # in bottom up order (aka the pcolormesh plotting direction)
    axes = ax[::-1]
    grouped: GroupBy = data.groupby(row)
    levels = get_groupby_levels_in_order_of_appearance(data, row)
    if sort:
        levels = np.sort(levels)
    if sharey and ylim is None:
        ylim = compute_padded_ylim(data)

    # seaborn violin places center of first violin above x=0
    xticks = np.arange(0, data.shape[1])
    xticklabels = data.columns.values
    for curr_ax, curr_level in zip(axes, levels):
        group_df: pd.DataFrame = sample_n(grouped.get_group(curr_level), n_per_group)
        group_df.columns.name = 'x'
        long_group_df = group_df.stack().to_frame('y').reset_index()
        sns.violinplot(x='x', y='y', data=long_group_df, ax=curr_ax)
        if ylim:
            curr_ax.set_ylim(ylim)
        curr_ax.set_xlabel('')
        if not show_all_xticklabels:
            curr_ax.set(xticks=[], xticklabels=[], xlabel='')
        else:
            # sns.violinplot already sets the same labels (data.columns.values),
            # but does not rotate them
            curr_ax.set_xticklabels(xticklabels, rotation=90)
            curr_ax.set_xlabel('')

    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, rotation=90)
    axes[0].set_xlabel('' if xlabel is None else xlabel)


def grouped_rows_line_collections(data: pd.DataFrame, row: Union[str, Iterable],
                                  ax: List[Axes], n_per_group=2000, sort=False,
                                  sharey=False, ylim=None, show_all_xticklabels=False,
                                  xlabel=None, alpha=0.01):
    axes = ax[::-1]
    grouped: GroupBy = data.groupby(row)
    levels = get_groupby_levels_in_order_of_appearance(data, row)
    if sort:
        levels = np.sort(levels)
    if ylim is not None:
        shared_ylim = ylim
    elif sharey:
        shared_ylim = compute_padded_ylim(data)
    else:
        shared_ylim = None

    xticks = np.arange(0.5, data.shape[1])
    xticklabels = data.columns.values

    for curr_ax, curr_level in zip(axes, levels):
        group_df: pd.DataFrame = sample_n(grouped.get_group(curr_level), n_per_group)
        segments = np.zeros(group_df.shape + (2,))
        segments[:, :, 1] = group_df.values
        segments[:, :, 0] = np.arange(0.5, group_df.shape[1])
        line_collection = LineCollection(segments, color='black', alpha=alpha)
        curr_ax.add_collection(line_collection)
        curr_ax.plot([1, 2, 3])

        # need to set plot limits, they will not autoscale
        curr_ax.set_xlim(0, data.shape[1])
        if shared_ylim is None:
            curr_ylim = compute_padded_ylim(group_df)
        else:
            curr_ylim = shared_ylim
        curr_ax.set_ylim(curr_ylim)

        if show_all_xticklabels:
            curr_ax.set_xticks(xticks)
            curr_ax.set_xticklabels(xticklabels, rotation=90)
            curr_ax.set_xlabel('')
        else:
            curr_ax.set(xticks=[], xticklabels=[], xlabel='')

    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, rotation=90)
    axes[0].set_xlabel(xlabel if xlabel is not None else '')




def sample_n(df, n):
    if df.shape[0] < 50:
        print('Warning: less than 50 elements for calculating violin plot')
    if df.shape[0] <= n:
        return df
    return df.sample(n=n)

def get_groupby_levels_in_order_of_appearance(df, groupby_var) -> np.ndarray:
    if isinstance(groupby_var, str):
        if groupby_var in df:
            levels = df[groupby_var].unique()
        elif groupby_var in df.index.names:
            levels = df.index.get_level_values(groupby_var).unique()
        else:
            raise ValueError(f'Can not find {groupby_var} in df')
    else:
        levels = pd.Series(groupby_var).unique()
    return levels


@dataclass
class CutDendrogram:
    """Plot dendrogram with link filtering, coloring and downsampling"""

    Z: np.ndarray
    cluster_ids_data_order: pd.Series
    ax: Axes
    pretty: bool = False
    orientation: str = 'horizontal'


    def __post_init__(self):
        self._args_processed = False


    def plot_links_until_clusters(self):

        if not self._args_processed:
            self._process_args()

        def get_cluster_ids(obs_idx, link_cluster_ids, leave_cluster_ids):
            n_obs = leave_cluster_ids.shape[0]
            if obs_idx < n_obs:
                return leave_cluster_ids[obs_idx]
            link_idx = obs_idx - n_obs
            return link_cluster_ids[link_idx]

        for i in range(-1, -self.Z.shape[0] - 1, -1):
            if self.link_cluster_ids[i] != -1:
                continue
            x, y = self.xcoords.iloc[i, :].copy(), self.ycoords.iloc[i, :].copy()
            left_child_idx, right_child_idx = self.Z_df[['left_child', 'right_child']].iloc[i]
            left_cluster_id = get_cluster_ids(left_child_idx,
                                              self.link_cluster_ids, self.cluster_ids_data_order)
            right_cluster_id = get_cluster_ids(right_child_idx,
                                               self.link_cluster_ids, self.cluster_ids_data_order)
            if left_cluster_id != -1:
                y['ylow_left'] = 0
            if right_cluster_id != -1:
                y['ylow_right'] = 0

            if self.orientation == 'horizontal':
                self.ax.plot(y, x, color='black')
            else:
                self.ax.plot(x, y, color='black')


        self._plot_post_processing()


    def plot_links_with_cluster_inspection(self, show_cluster_points: bool = False,
                                           point_params: Optional[Dict] = None,
                                           min_cluster_size: Optional[int] = None,
                                           min_height: str = 'auto'):

        if not self._args_processed:
            self._process_args()

        if point_params is None:
            point_params = {'s': 2, 'marker': 'x', 'c': 'black'}
        if min_height == 'auto':
            min_height = self._linkage_estimate_min_height()
        if min_cluster_size is None:
            min_cluster_size = self.n_leaves * 0.02
        link_colors = pd.Series(self.link_cluster_ids).map(CMAP_DICT['cluster_id_to_color'])
        # cluster_ids_ser, link_colors = linkage_get_link_cluster_ids(Z, cluster_ids_ser)
        # ys = []
        # _linkage_get_child_y_coords(Z, ys, Z.shape[0] * 2, 3)
        point_xs = []
        point_ys = []
        for i in range(self.Z.shape[0]):
            x = self.xcoords.loc[i, :].copy()
            y = self.ycoords.loc[i, :].copy()

            if self.Z[i, 3] < min_cluster_size:
                continue
            if y['yhigh1'] < min_height:
                continue

            try:
                left_child_size = self.Z[int(self.Z[i, 0]) - self.n_leaves, 3]
            except IndexError:
                left_child_size = 1
            try:
                right_child_size = self.Z[int(self.Z[i, 1]) - self.n_leaves, 3]
            except IndexError:
                right_child_size = 1

            if (left_child_size < min_cluster_size
                    and right_child_size < min_cluster_size):
                y['ylow_left'] = 0
                y['ylow_right'] = 0
                if self.orientation == 'vertical':
                    self.ax.plot(x, y, color=link_colors[i])
                else:
                    self.ax.plot(y, x, color=link_colors[i])

            else:
                if y['ylow_left'] < min_height or left_child_size < min_cluster_size:
                    y['ylow_left'] = 0

                    curr_point_x = x['xleft1']
                    obs_idx = self.Z[i, 0]

                    # if obs_idx > self.Z.shape[0]:
                    lookup_idx = int(obs_idx - (self.Z.shape[0] + 1))
                    point_ys.append(self.Z[lookup_idx, 2])
                    point_xs.append(curr_point_x)
                    # curr_size = self.Z[lookup_idx, 3]
                    curr_ys: List[float] = []
                    curr_ys = self._linkage_get_child_y_coords(curr_ys, obs_idx, 8)
                    point_ys += curr_ys
                    point_xs += [curr_point_x] * len(curr_ys)
                    # else:
                    #     curr_size = 1

                if y['ylow_right'] < min_height or right_child_size < min_cluster_size:
                    y['ylow_right'] = 0
                    curr_point_x = x['xright1']
                    obs_idx = self.Z[i, 1]
                    if obs_idx > self.Z.shape[0]:
                        lookup_idx = int(obs_idx - (self.Z.shape[0] + 1))
                        point_ys.append(self.Z[lookup_idx, 2])
                        point_xs.append(curr_point_x)
                        curr_ys = []
                        curr_ys = self._linkage_get_child_y_coords(curr_ys, obs_idx, 4)
                        point_ys += curr_ys
                        point_xs += [curr_point_x] * len(curr_ys)

                if self.orientation == 'vertical':
                    self.ax.plot(x, y, color=link_colors[i])
                else:
                    self.ax.plot(y, x, color=link_colors[i])

            if show_cluster_points:
                if self.orientation == 'vertical':
                    self.ax.scatter(point_xs, point_ys, **point_params, zorder=20_000)
                else:
                    self.ax.scatter(point_ys, point_xs, **point_params, zorder=20_000)


    def _process_args(self):
        self.n_leaves = self.Z.shape[0] + 1
        linkage_mat = Linkage(
                self.Z,
                cluster_ids=ClusterIDs(
                        self.cluster_ids_data_order.to_frame('clustering1')))
        self.link_cluster_ids = linkage_mat.get_link_cluster_ids('clustering1')
        self.Z_df = linkage_mat.df
        self.Z_df['cluster_ids'] = self.link_cluster_ids
        self.xcoords, self.ycoords = self._linkage_get_coord_dfs()

        assert self.orientation in ['horizontal', 'vertical']

        if self.orientation == 'horizontal':
            self.ax.invert_xaxis()
            #     self.ax.invert_yaxis()
            self.ax.margins(0)

        self._args_processed = True


    def _plot_post_processing(self):
        """Orientation, axis limits, despine..."""

        max_xcoord = self.xcoords.max().max()
        xcoord_axis = 'x' if self.orientation == 'vertical' else 'y'
        self.ax.set(**{f'{xcoord_axis}lim': [0, max_xcoord]})

        if self.pretty:
            self.ax.margins(0)
            self.ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
            sns.despine(ax=self.ax, left=True, bottom=True)


    def _linkage_estimate_min_height(self):
        min_cluster_height = (self.Z_df
                              .groupby('cluster_ids')['height'].max()
                              .drop((-1.0)).min())
        return 0.7 * min_cluster_height


    def _linkage_get_child_y_coords(self, ys, obs_idx, n_children):
        n = self.Z.shape[0] + 1
        n_children -= 1
        lookup_idx = int(obs_idx - n)
        left_obs_idx = int(self.Z[lookup_idx, 0])
        if left_obs_idx > n:
            ys.append(self.Z[left_obs_idx - n, 2])
            if n_children > 0:
                _ = self._linkage_get_child_y_coords(ys, obs_idx, n_children)
        right_obs_idx = int(self.Z[lookup_idx, 1])
        if right_obs_idx > n:
            ys.append(self.Z[right_obs_idx - n, 2])
            if n_children > 0:
                _ = self._linkage_get_child_y_coords(ys, right_obs_idx, n_children)
        return ys


    def _linkage_get_coord_dfs(self):
        dendrogram_dict = dendrogram(self.Z, no_plot=True)
        dcoords = pd.DataFrame.from_records(dendrogram_dict['dcoord'])
        dcoords.columns = ['ylow_left', 'yhigh1', 'yhigh2', 'ylow_right']
        dcoords = dcoords.sort_values('yhigh1')
        icoords = pd.DataFrame.from_records(dendrogram_dict['icoord'])
        icoords.columns = ['xleft1', 'xleft2', 'xright1', 'xright2']
        icoords = icoords.loc[dcoords.index, :]
        xcoords = icoords.reset_index(drop=True)
        ycoords = dcoords.reset_index(drop=True)
        return xcoords, ycoords

def cut_dendrogram(linkage_mat: np.ndarray,
                   cluster_ids_data_order: pd.Series,
                   ax: Axes,
                   pretty: bool = False,
                   stop_at_cluster_level=True,
                   orientation: str = 'horizontal',
                   show_cluster_points: bool = False,
                   point_params: Optional[Dict] = None,
                   min_cluster_size: Optional[int] = None,
                   min_height: str = 'auto'
                   ):
    cut_dendrogram = CutDendrogram(Z=linkage_mat,
                                   cluster_ids_data_order=cluster_ids_data_order,
                                   ax=ax, pretty=pretty, orientation=orientation)
    if stop_at_cluster_level:
        cut_dendrogram.plot_links_until_clusters()
    else:
        cut_dendrogram.plot_links_with_cluster_inspection(
                show_cluster_points=show_cluster_points,
                point_params=point_params,
                min_cluster_size=min_cluster_size,
                min_height=min_height
        )


def grouped_rows_heatmap(df: pd.DataFrame, row_: Union[str, Iterable],
                         fn: Union[str, Callable],
                         cmap: str, ax: Axes, fig=Figure, sort=False,
                         **kwargs):
    agg_df = df.groupby(row_).agg(fn)
    levels = get_groupby_levels_in_order_of_appearance(df, row_)
    if sort:
        levels = np.sort(levels)
    agg_df = agg_df.loc[levels, :]
    heatmap(df=agg_df, ax=ax, fig=fig, cmap=cmap, **kwargs)

    # xticks = np.arange(0.5, df.shape[1])
    # xticklabels = df.columns.values
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels, rotation=90)
    # ax.set_xlabel(xlabel if xlabel is not None else '')


