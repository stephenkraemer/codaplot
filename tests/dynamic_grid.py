from pathlib import Path

import complex_heatmap as ch

def test_grid_manager_without_plotting(tmpdir):
    tmpdir = Path(tmpdir)
    GE = ch.GridElement
    gm = ch.GridManager(
            grid = [
                [
                    GE('spacer', 4, 'abs'), GE('col_dendro1', 2),
                ],
                [
                    GE('row_dendro1', 2, 'abs'), GE('anno1', 2, 'abs'), GE('var1', 2),
                ],
                [
                    GE('row_dendro1', 2, 'abs'), GE('anno1', 2, 'abs'), GE('var3', 2),
                ],
            ],
            figsize = (10, 5),
            height_ratios=(1, 1, 1)
    )
    gm.axes['anno1'].scatter([1, 2, 3], [1, 2, 3])
    gm.fig.savefig(tmpdir / 'test.png')

def test_grid_manager_with_mixed_plotting(tmpdir):
    tmpdir = Path(tmpdir)
    GE = ch.GridElement

    def anno_plot(ax, x, y, s):
        ax.scatter(x, y, s=s)

    anno_ge = ch.GridElement('anno1', 2, 'abs', anno_plot,
                          x=[1, 2, 3], y=[5, 6, 7], s=[1, 10, 10])

    gm = ch.GridManager(
            grid = [
                [
                    GE('spacer', 4, 'abs'), GE('col_dendro1', 2),
                ],
                [
                    GE('row_dendro1', 2, 'abs'), anno_ge, GE('var1', 2),
                ],
                [
                    GE('row_dendro1', 2, 'abs'), anno_ge, GE('var3', 2),
                ],
            ],
            figsize = (10, 5),
            height_ratios=(1, 1, 1)
    )

    gm.axes['var1'].plot([1, 2, 3], [1, 2, 3])
    gm.fig.savefig(tmpdir / 'test.png')
