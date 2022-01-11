# # do jitter plot for homogeneity plot runner

# if do_jitter_plot:
#     fig = cluster_homogeneity_jitter_plot(
#         group_df=group_df,
#         stat_name=data_name,
#         color=color_ser,
#         color_name=color_name,
#         geom_point_kwargs=geom_point_kwargs,
#         figsize=jitter_figsize,
#     )
#     if trunk_path:
#         # noinspection PyUnboundLocalVariable
#         png_path = png_pattern_jitter.format(cluster_id=cluster_id)
#         fig.savefig(png_path, bbox_inches="tight")
#         if "pdf" in additional_formats:
#             fig.savefig(ut.pdf(png_path), bbox_inches="tight")
#         if "svg" in additional_formats:
#             fig.savefig(ut.svg(png_path), bbox_inches="tight")

#         figs[("jitter", cluster_id)] = (png_path, fig)
#     else:
#         figs[("jitter", cluster_id)] = ("", fig)
