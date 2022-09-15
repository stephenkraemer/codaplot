# %% [markdown]
# # Imports

import codaplot as co

# %% [markdown]
# # Axes-level heatmaps


# %% [markdown]
# ## Simple heatmaps


# %% [markdown]
# ### Simple heatmap

# %%
fig, ax = plt.subplots(1, 1)
h = co.Heatmap(
    color=mlog10_pvalues,
    vmin=0,
    vmax=3,
    cmap="Reds",
    color_legend_title="-log10(q-values)",
    color_legend_cbar_kwds=dict(
        shrink=0.5,
        aspect=2,
        expand="both",
    ),
).draw(ax=ax)
ax.set(xlabel="abc", ylabel="abc")


# %% [markdown]
# ### Simple spaced heatmap

# %%
fig, ax = plt.subplots(1, 1)
h = co.Heatmap(
    color=mlog10_pvalues,
    vmin=0,
    vmax=3,
    cmap="Reds",
    color_legend_title="-log10(q-values)",
    color_legend_cbar_kwds=dict(
        shrink=0.5,
        aspect=2,
        expand="both",
    ),
    row_spacing_group_ids=list('aaaabbbbcccc'),
    col_spacing_group_ids=np.tile([1, 2, 3], 5),
).draw(ax=ax)
ax.set(xlabel="abc", ylabel="abc")


# %% [markdown]
# ### Categorical heatmap

# %%
fig, ax = plt.subplots(1, 1)
h = co.Heatmap(
    color=categorical_data,
    cmap="Set1",
    color_legend_title="groups",
    color_legend_legend_kwds=dict(loc=(0, 1), bbox_to_anchor='upper_left'),
).draw(ax=ax)
ax.set(xlabel="abc", ylabel="abc")



# %% [markdown]
# ## Powerful meshes

# %% [markdown]
# ### Meshes with multiple mark properties

# %% [markdown]
# - size is specified proportinal to rectangle space, so will always work with any figure size

# %%
for DemoMesh in [co.CircleMesh, co.RectangleMesh, co.TriangleMesh]:
    fig, ax = plt.subplots(1, 1)
    h = co.Heatmap(
        color=log_odds,
        size=p_values,
        marker=DemoMesh(
            vmin=0,
            vmax=3,
            cmap='Reds',
            sizemin=0.1,
            sizemax=0.9,
            ),
        color_legend_title="-log10(q-values)",
        color_legend_cbar_kwds=dict(
            shrink=0.5,
            aspect=2,
            expand="both",
        ),
        row_spacing_group_ids=list('aaaabbbbcccc'),
        col_spacing_group_ids=np.tile([1, 2, 3], 5),
    ).draw(ax=ax)
    ax.set(xlabel="abc", ylabel="abc")

# %% [markdown]
# ### MultiMarkMesh


# %% [markdown]
# ## Handle multiple legends


# %% [markdown]
# ## Control mark placement to overlay complex information in one heatmap


# %% [markdown]
# ### Example 1: Methylation and expression side by side

# %%
fig, ax = plt.subplots(1, 1)

shared_kwds = dict(
    color_legend_cbar_kwds=dict(
        shrink=0.4,
        aspect=10,
    ),
    row_spacing_group_ids=list('aaaabbbbcccc'),
    col_spacing_group_ids=np.tile([1, 2, 3], 5),
    )

# Add average methylation data
meth_legend_d = co.Heatmap(
    color=average_meth,
    marker=co.TriangleMesh(
        vmin=0,
        vmax=1,
        cmap='cividis',
        ),
    marker_placement_max_size=0.9,
    marker_placement_loc=(0, 0),
    marker_placement_bbox_to_anchor='upper_left',
    color_legend_title="Average meth.",
    **shared_kwds,
).draw(ax=ax)


# add expression data
expr_legend_d = co.Heatmap(
    color=expression_data,
    marker=co.TriangleMesh(
        vmin=0,
        vmax=10,
        cmap='Reds',
        ),
    marker_placement_max_size=0.9,
    marker_placement_loc=(1, 1),
    marker_placement_bbox_to_anchor='lower_right',
    color_legend_title="Normalized expression",
    **shared_kwds
).draw(ax=ax)

co.add_legends([meth_legend_d, expr_legend_d], loc=(1.05, 0), bbox_to_anchor='upper left')

ax.set(xlabel="Samples", ylabel="Genes")

# Implementation notes
# - both layers set the same ticklabels, so thats fine


# %% [markdown]
# ### Example 2: Oncoprints


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
