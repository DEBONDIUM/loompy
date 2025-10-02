#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:17:05 2025

@author: lbremaud
"""


from .segment import *
from .section import *
from .node import *
from .mesh import *
from .element import *
from .frame import *
#from .operators import *
#from .plotter import *

# Aggregate __all__ from submodules
__all__ = []
from .segment import __all__ as _segment_all
from .section import __all__ as _section_all
from .node import __all__ as _node_all
from .mesh import __all__ as _mesh_all
from .element import __all__ as _element_all
from .frame import __all__ as _frame_all
#from .operators import __all__ as _operators_all
#from .plotter import __all__ as _plotter_all

__all__ += _segment_all + _section_all + _node_all + _mesh_all
__all__ += _element_all + _frame_all
#+ _operators_all + _plotter_all
