from pathlib import Path

import complex_heatmap as ch
from complex_heatmap import GridElement as GE


def test_insert_matched_row():
    def fn():
        pass
    gm = ch.GridManager(grid=[[GE('a', plotter=fn, width=3, kind='abs'),
                               GE('b', plotter=fn, width=4, kind='rel'),
                               ],
                              [GE('c', plotter=fn, width=5, kind='abs'),
                               GE('d', plotter=fn, width=6, kind='rel'),
                               GE('e', plotter=fn, width=4, kind='abs'),
                               ]
                              ],
                        height_ratios=[(1, 'rel'), (1, 'abs')],
                        figsize=(4, 4))
    gm.insert_matched_row(1, GE('f', plotter=fn), height=(2, 'abs'),
                          only_cols=[1, 2])
    assert gm.grid[1][0].name.startswith('spacer')
    assert gm.grid[1][0].width == 5
    assert gm.grid[1][0].kind == 'abs'
    assert gm.grid[1][1].name == 'f_1'
    assert gm.grid[1][1].width == 6
    assert gm.grid[1][1].kind == 'rel'
    assert gm.grid[1][2].name == 'f_2'
    assert gm.grid[1][2].width == 4
    assert gm.grid[1][2].kind == 'abs'

def test_prepend_col():
    def fn():
        pass
    gm = ch.GridManager(grid=[[GE('a', plotter=fn, width=3, kind='abs'),
                               GE('b', plotter=fn, width=4, kind='rel'),
                               ],
                              [GE('c', plotter=fn, width=5, kind='abs'),
                               GE('d', plotter=fn, width=6, kind='rel'),
                               GE('e', plotter=fn, width=4, kind='abs'),
                               ]
                              ],
                        height_ratios=[(1, 'rel'), (1, 'abs')],
                        figsize=(4, 4))
    gm.prepend_col(GE('f', plotter=fn, width=10, kind='abs'), only_rows=[1])
    assert gm.grid[0][0].name.startswith('spacer')
    assert gm.grid[0][0].width == 10
    assert gm.grid[0][0].kind == 'abs'
    assert gm.grid[1][0].name == 'f'
    assert gm.grid[1][0].width == 10
    assert gm.grid[1][0].kind == 'abs'


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
            height_ratios=((1, 'rel'), (1, 'rel'), (1, 'rel'))
    )
    gm.create_or_update_figure()
    gm.axes_dict['anno1'].scatter([1, 2, 3], [1, 2, 3])
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
            height_ratios=((1, 'rel'), (1, 'rel'), (1, 'rel'))
    )
    gm.create_or_update_figure()

    gm.axes_dict['var1'].plot([1, 2, 3], [1, 2, 3])
    gm.fig.savefig(tmpdir / 'test.png')
