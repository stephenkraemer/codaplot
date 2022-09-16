# %% [markdown]
# # Imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import codaplot.new_api as co

# %% [markdown]
# # Axes-level heatmaps


# %% [markdown]
# ## Simple heatmaps


# %% [markdown]
# ### Simple heatmap


# %%
mlog10_pvalues = pd.DataFrame()

fig, ax = plt.subplots(1, 1)
h = co.Heatmap(
    color=mlog10_pvalues,
    marker_kwargs=dict(
        vmin=0,
        vmax=3,
        cmap="Reds",
    ),
    guides_color_title="-log10(q-values)",
    guides_color_cbar_kwds=dict(
        shrink=0.5,
        aspect=2,
        expand="both",
    ),
).draw(ax=ax)
ax.set(xlabel="abc", ylabel="abc")


# %% [markdown]
# ### Simple spaced heatmap


# %% [markdown]
# #### Same spacer size everywhere

# %%
fig, ax = plt.subplots(1, 1)
h = co.Heatmap(
    color=mlog10_pvalues,
    marker_kwargs=dict(
        vmin=0,
        vmax=3,
        cmap="Reds",
    ),
    guides_color_title="-log10(q-values)",
    guides_color_cbar_kwds=dict(
        shrink=0.5,
        aspect=2,
        expand="both",
    ),
    spacing_row_ids=list("aaaabbbbcccc"),
    spacing_col_ids=np.tile([1, 2, 3], 5),
).draw(ax=ax)
ax.set(xlabel="abc", ylabel="abc")


# %% [markdown]
# #### Use different spacer sizes to add more structure

# %% [markdown]
# ### Categorical heatmap

# %%
categorical_data = pd.DataFrame()

fig, ax = plt.subplots(1, 1)
h = co.Heatmap(
    color=categorical_data,
    # autodetected for categorical and object dtypes
    # specify for numeric dtypes
    # is_categorical=True,
    marker_kwargs=dict(
        cmap="Set1",
    ),
    guides_color_title="groups",
    guides_color_legend_kwds=dict(loc=(0, 1), bbox_to_anchor="upper_left"),
).draw(ax=ax)
ax.set(xlabel="abc", ylabel="abc")

# %% [markdown]
# ## Powerful meshes

# %% [markdown]
# ### Meshes with multiple mark properties

# %% [markdown]
# - size is specified proportinal to rectangle space, so will always work with any figure size

# %%
log_odds = pd.DataFrame()
p_values = pd.DataFrame()


for DemoMesh in [co.CircleMesh, co.RectangleMesh, co.TriangleMesh]:
    fig, ax = plt.subplots(1, 1)
    h = co.Heatmap(
        color=log_odds,
        size=p_values,
        marker_mesh=DemoMesh(
            color_min=0,
            color_max=3,
            color_map="Reds",
            size_min=0.1,
            size_max=0.9,
        ),
        guides_color_title="log-odds",
        guides_size_title="p-value",
        guides_color_cbar_kwds=dict(
            shrink=0.5,
            aspect=2,
            expand="both",
        ),
        guides_size_legend_kwds=None,
        spacing_row_ids=list("aaaabbbbcccc"),
        spacing_col_ids=np.tile([1, 2, 3], 5),
    ).draw(ax=ax)
    ax.set(xlabel="abc", ylabel="abc")

# %% [markdown]
# ### MultiMarkMesh

co.MultiMarkMesh

# %% [markdown]
# ### PcolorMeshPlus

co.PcolorMeshPlus

# %% [markdown]
# ## Handle multiple legends


# %% [markdown]
# ## Control mark placement to overlay complex information in one heatmap


# %% [markdown]
# ### Example 1: Methylation and expression side by side

# %%
fig, ax = plt.subplots(1, 1)

shared_kwargs = dict(
    guides_color_cbar_kwargs=dict(
        shrink=0.4,
        aspect=10,
    ),
    spacing_row_ids=list("aaaabbbbcccc"),
    spacing_col_ids=np.tile([1, 2, 3], 5),
)

# Add average methylation data
average_meth = pd.DataFrame()
meth_legend_d = co.Heatmap(
    color=average_meth,
    marker_mesh=co.TriangleMesh(
        color_min=0,
        color_max=1,
        color_map="cividis",
    ),
    marker_max_field_fraction=0.9,
    marker_loc=(0, 0),
    marker_bbox_to_anchor="upper_left",
    guides_color_title="Average meth.",
    **shared_kwargs,
).draw(ax=ax)


# add expression data
expression_data = pd.DataFrame()
expr_legend_d = co.Heatmap(
    color=expression_data,
    marker_mesh=co.TriangleMesh(
        color_min=0,
        color_max=10,
        color_map="Reds",
    ),
    marker_max_field_fraction=0.9,
    marker_loc=(1, 1),
    marker_bbox_to_anchor="lower_right",
    guides_color_title="Normalized expression",
    **shared_kwargs,
).draw(ax=ax)

co.add_guides(
    ax=ax,
    guides=[meth_legend_d, expr_legend_d],
    loc=(1.05, 0),
    bbox_to_anchor="upper_left",
)

ax.set(xlabel="Samples", ylabel="Genes")

# Implementation notes
# - both layers set the same ticklabels, so thats fine


# %% [markdown]
# ### Example 2: Oncoprints


# %% [markdown]
# ## Clustermap

# %%
# fig, ax = plt.subplots(1, 1)
# h = (
#     co.Heatmap()
#     .add_layer(
#         data=pvalues,
#         mesh=co.HeatmapMesh(
#             vmin="p1",
#             vmax="p99",
#             vcenter=0,
#             color_bar=ColorBarLegend(
#                 title="-log10(q-values)",
#                 shrink=0.5,
#                 aspect=2,
#                 expand="both",
#             ),
#         ),
#     )
#     .hclust_cols(
#         method="average",
#         metric="euclidean",
#         cut_method="cutree",
#         cut_kwds=dict(cutoff=0.8),
#     )
#     .hclust_rows(method="average", metric="euclidean")
#     .add_col_dendrogram(co.Dendrogram(show_only_above=0.8))
#     .add_row_dendrogram()
#     .insert_spacers(
#         row_spacer_sizes=0.01,
#         col_spacer_sizes=0.01,
#     )
#     .draw(figsize=(4, 4))
#     # or
#     .draw_on_axes()
# )
