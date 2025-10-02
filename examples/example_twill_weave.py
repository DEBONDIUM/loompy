#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:21:42 2025

@author: lbremaud
"""

# =============================================================================
# INTRODUCTION
# =============================================================================
'''
Example: Twill Weave Mesh Generation and Visualization
======================================================

This example demonstrates how to use the `loompy` package to:

1. Define and save a custom cross-section geometry (here, a large elliptic section).
2. Build a `Section` object from the cross-section file.
3. Construct unit `Segment` that represent basic fiber paths (warp/weft).
4. Assemble major segments by combining unit segments with transformations
   (translations and rotations).
5. Generate a woven `Mesh` by replicating warp and weft segments in a twill-weave
   pattern.
6. Visualize the sections, segments, and final mesh using PyVista.
7. Export the generated mesh to standard formats:
   - `.msh` for Gmsh
   - `.inp` for Abaqus
   - `.ori` for orientation definitions

The workflow shows how geometric definitions (sections and segments) can be
hierarchically composed into a full 3D woven structure and then exported
for simulation or further analysis.

Requirements:
-------------
- loompy (this package)
- numpy
- pyvista (for visualization)

Author: lbremaud
Created on Tue Sep 30 16:21:42 2025
'''

# =============================================================================
# LIB
# =============================================================================
import loompy as lp
import numpy as np

# =============================================================================
# EXAMPLE SECTION: thin elliptic section
# =============================================================================
semi_major_axis = 0.55 / 2.
semi_minor_axis = 0.05 / 2.
length_cut = 0.5

num_rows = 3
num_cols = 7

xcoord = np.linspace(-length_cut / 2., length_cut / 2., num_cols)
ycoord = semi_minor_axis * np.sqrt(1 - pow(xcoord, 2.) / pow(semi_major_axis, 2))

n_section = np.empty((num_cols * num_rows, 2))
n_section[:, 0] = np.tile(xcoord, num_rows)
n_section[:, 1] = np.concatenate((ycoord, np.zeros(num_cols), -ycoord))

np.savetxt('section/thin_elliptic_section.txt', n_section)

# =============================================================================
# SECTION
# =============================================================================
# section parameters
filename = 'section/thin_elliptic_section.txt'
num_rows = 3
num_cols = 7

# build section
section = lp.Section.from_file(filename, num_rows, num_cols)

# visualize section
section.plot(frame = False, grid = False, save = True, save_dir = 'twill_weave/img')

# =============================================================================
# UNIT SEGMENTS
# =============================================================================
# unit segment parameters
num_division = 10
length = 1.

# build first unit segment (under to over)
unit_segment_01 = lp.Segment(section = section, shape = [0, 1], num_division = num_division, length = length, dy = 1.5)

# build second unit segment (under to over)
unit_segment_10 = lp.Segment(section = section, shape = [1, 0], num_division = num_division, length = length, dy = 1.5)

# build third unit segment (straight)
unit_segment_00 = lp.Segment(section = section, shape = [0, 0], num_division = num_division, length = length, dy = 1.5)

# visualize second unit segment
unit_segment_10.plot(frame_section = True, frame_element = False, grid = False)

# =============================================================================
# MAJOR SEGMENTS
# =============================================================================
dy = 0.5 * section.dim[1] * unit_segment_01.dy

# build warp major segments
warp_segment_0110 = lp.Segment.from_assembly([unit_segment_01.copy(), 
                                              unit_segment_00.copy().translate(dy = dy, dz = length), 
                                              unit_segment_10.copy().translate(dz = 2. * length),
                                              unit_segment_00.copy().translate(dy = -dy, dz = 3. * length)])
warp_segment_0011 = lp.Segment.from_assembly([unit_segment_00.copy().translate(dy = -dy), 
                                              unit_segment_01.copy().translate(dz = length), 
                                              unit_segment_00.copy().translate(dy = dy, dz = 2. * length),
                                              unit_segment_10.copy().translate(dz = 3. * length),])
warp_segment_1001 = lp.Segment.from_assembly([unit_segment_10.copy(), 
                                              unit_segment_00.copy().translate(dy = -dy, dz = length), 
                                              unit_segment_01.copy().translate(dz = 2. * length),
                                              unit_segment_00.copy().translate(dy = dy, dz = 3. * length)])
warp_segment_1100 = lp.Segment.from_assembly([unit_segment_00.copy().translate(dy = dy), 
                                              unit_segment_10.copy().translate(dz = length), 
                                              unit_segment_00.copy().translate(dy = -dy, dz = 2. * length),
                                              unit_segment_01.copy().translate(dz = 3. * length)])

# build weft major segments
weft_segment_0110 = warp_segment_0110.copy().rotate('y', np.pi / 2.).translate(dx = 2. * length, dz = -2. * length)
weft_segment_0011 = warp_segment_0011.copy().rotate('y', np.pi / 2.).translate(dx = 2. * length, dz = -2. * length)
weft_segment_1001 = warp_segment_1001.copy().rotate('y', np.pi / 2.).translate(dx = 2. * length, dz = -2. * length)
weft_segment_1100 = warp_segment_1100.copy().rotate('y', np.pi / 2.).translate(dx = 2. * length, dz = -2. * length)

# visualize first warp segment
warp_segment_0110.plot(frame = True, frame_element = False, grid = False, save = True, save_dir = 'twill_weave/img')

# =============================================================================
# MESH
# =============================================================================
# mesh parameters
num_warp = 4
num_weft = 4
num_segment_warp = 1
num_segment_weft = 1

# build mesh
m = lp.Mesh(warp = [warp_segment_0110, warp_segment_0011, warp_segment_1001, warp_segment_1100], 
            weft = [weft_segment_1100, weft_segment_0110, weft_segment_0011, weft_segment_1001], 
            space = length, 
            num_warp = num_warp, num_weft = num_weft, 
            num_segment_warp = num_segment_warp, num_segment_weft = num_segment_weft)

# visualize mesh (this may take a little time)
m.plot(frame_warp = False, frame_weft = False, grid = False, save = True, save_dir = 'twill_weave/img')

# export to gmsh
m.export('msh', save_dir = 'twill_weave')

# export to abaqus
m.export('inp', save_dir = 'twill_weave')
m.export_ori(save_dir = 'twill_weave')

