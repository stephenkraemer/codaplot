"""
Correctly place multiple legends at different locations relative to an Axes
can be main axes where heatmap was drawn, or separate axes
"""

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import mouse_hema_meth.utils as ut
from mouse_hema_meth.utils import cm
from dataclasses import dataclass

def add_guides(
        ax: Axes,
        guides: List[Dict],
        loc: Tuple[float, float],
        bbox_to_anchor: Literal['upper_left', 'center_left', '...'],
        ):
    pass
