#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 21:33:56 2025

@author: lbremaud
"""

# =============================================================================
# LIB
# =============================================================================
import numpy as np
import pyvista as pv

from .node import Node
from .element import Element
from .frame import Frame

# =============================================================================
# PLOTTER
# =============================================================================
def plot_pv(
    nodelist: list[tuple[list[Node], str, str | None, bool]],
    elementlist: list[tuple[list[Element], str, bool]],
    framelist: list[Frame],
    grid: bool,
    save: str,
    filename: str) -> None:
    """
    Generic plot function using pyvista.
    
    Args:
        nodelist (list[tuple[list[Node], str, str | None, bool]]): Nodes tuple (list[Node], color, linestyle, show_idx).
        elementlist (list[tuple[list[Element], str, bool]]): Element tuple (list[Element], color, show_idx).
        framelist (list[Frame]): Frame objects.
        grid (bool): Show grid option.
        save (bool): Save option.
        filename (str): Filename.
    """    
    plotter = pv.Plotter()

    for nl, c, ls, idx in nodelist:
        xyz = np.array([(n.x, n.y, n.z) for n in nl])
        plotter.add_points(xyz, color = c, point_size = 10, render_points_as_spheres = True)
        if ls != None: plotter.add_mesh(pv.lines_from_points(xyz), color = c, line_width = 2)
        if idx:
            ni = [n.i for n in nl]
            plotter.add_point_labels(xyz, ni, font_size = 14, text_color = 'black', always_visible = True, shape_opacity = 0)
    
    for el, c, idx in elementlist:
        for e in el:
            xyz = np.array([(n.x, n.y, n.z) for n in e.nodelist])
            lines = np.hstack([[2, j, k] for (j, k) in e.connectivity])
            poly = pv.PolyData(xyz, lines = lines)
            plotter.add_mesh(poly, color = c, line_width = 2)
            if idx:
                c = e.centroid()
                plotter.add_point_labels([c.x, c.y, c.z], [e.i], font_size = 14, text_color = 'black', always_visible = True, shape_opacity = 0)

    for f in framelist:
        c = f.origin
        scale = 0.5
        plotter.add_arrows(np.array([c.x, c.y, c.z]), f.vx, mag = scale, color = 'magenta')
        plotter.add_arrows(np.array([c.x, c.y, c.z]), f.vy, mag = scale, color = 'yellow')
        plotter.add_arrows(np.array([c.x, c.y, c.z]), f.vz, mag = scale, color = 'green')
    
    if grid:
        plotter.show_bounds(
            xtitle = 'X-axis', 
            ytitle = 'Y-axis', 
            ztitle = 'Z-axis',
            grid = 'front', 
            location = 'outer', 
            all_edges = True, 
            ticks = 'both', 
            font_size = 10)
    
    plotter.show_axes()
    plotter.show(cpos = 'iso')
    
    if save:
        plotter.screenshot(filename)
