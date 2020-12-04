Development version

2020-12-04

- fixes:
  - get_list_of_stretch_annos
     - change series.replace to series.map
   - legend title is not handled by constrained layout
     - legend is added as Artist, the title of this Legend artist is apparently not considered by CL
        fixed by adding dummy Text artist with same extent as legend title
   - remove borders from rectangle patches in categorical heatmap legend
   - circular co.plotting import
  - add observed=True to flow_plot groupby logic to deal with categorical cluster ids

- new features
    - allow using legend_kwargs in cross_plot
    - allow vertical colorbar labels in add_guides (and thus in cross_plot etc.)

- api improvements
    - export more objects into co namespace

