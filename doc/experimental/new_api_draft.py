# %% [markdown]
# # Imports

import codaplot as co

# %% [markdown]
# # Axes-level heatmaps


# %% [markdown]
# ## Simple heatmap

# %%
fig, ax = plt.subplots(1, 1)
h = (
    co.Heatmap(
        color=mlog10_pvalues,
            vmin=0,
            vmax=3,
        cmap='Reds',
        color_legend_title="-log10(q-values)"
        color_legend_cbar_kwds = dict(
                shrink=0.5,
                aspect=2,
                expand="both",
            ),
        )
    .draw(ax=ax)
)
ax.set(xlabel="abc", ylabel="abc")


# %% [markdown]
# ## Simple spaced heatmap

# %%
fig, ax = plt.subplots(1, 1)
h = (
    co.Heatmap()
    .add_layer(
        data=pvalues,
        mesh=co.HeatmapMesh(
            vmin="p1",
            vmax="p99",
            vcenter=0,
            color_bar=ColorBarLegend(
                title="-log10(q-values)",
                shrink=0.5,
                aspect=2,
                expand="both",
            ),
        ),
    )
    .insert_spacers(
        row_group_ids=dmr_cluster_compartments,
        col_group_ids=pop_compartments,
        row_spacer_sizes=0.01,
        col_spacer_sizes=0.01,
    )
    .draw(ax=ax)
)
ax.set(xlabel="abc", ylabel="abc")


# %% [markdown]
# ## Categorical heatmap

# %%
fig, ax = plt.subplots(1, 1)
h = (
    co.Heatmap()
    .add_layer(
        data=categorical_df,
        mesh=co.HeatmapMesh(
            # only necessary when categorical df is not object, pd.Str or pd.Categorical
            is_categorical=True,
            cmap="Set1",
            legend=co.CategoricalColorLegend(title="classes"),
        ),
    )
    .draw_on_axes(ax)
)


# %% [markdown]
# ## Clustermap

# %%
fig, ax = plt.subplots(1, 1)
h = (
    co.Heatmap()
    .add_layer(
        data=pvalues,
        mesh=co.HeatmapMesh(
            vmin="p1",
            vmax="p99",
            vcenter=0,
            color_bar=ColorBarLegend(
                title="-log10(q-values)",
                shrink=0.5,
                aspect=2,
                expand="both",
            ),
        ),
    )
    .hclust_cols(
        method="average",
        metric="euclidean",
        cut_method="cutree",
        cut_kwds=dict(cutoff=0.8),
    )
    .hclust_rows(method="average", metric="euclidean")
    .add_col_dendrogram(co.Dendrogram(show_only_above=0.8))
    .add_row_dendrogram()
    .insert_spacers(
        row_spacer_sizes=0.01,
        col_spacer_sizes=0.01,
    )
    .draw(figsize=(4, 4))
    # or
    .draw_on_axes()
)


# %% [markdown]
# ## Nicely formatted ticklabels
