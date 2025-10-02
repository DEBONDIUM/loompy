#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 18:10:24 2025

@author: lbremaud
"""

# =============================================================================
# LIB
# =============================================================================
from __future__ import annotations

import os
import copy
import bezier
import numpy as np

from . import operators as op
from . import plotter as pt

from .frame import Frame
from .node import Node
from .element import Element
from .section import Section

# =============================================================================
# CLASS SEGMENT
# =============================================================================
class Segment():
    def __init__(self,
        section: Section,
        shape: list[int],
        num_division: int,
        length: float,
        dx: float = 0.,
        dy: float = 1.25,
        i: int | None = None):
        """
        Define a segment.
        
        Args:
            section (Section): Unit section.
            shape (list[int]): Segment shape (`0` for under, `1` for over).
            num_division (int): Number of divisions of the segment.
            length (float): Segment length.
            dx (float): Scaling factor proportional to the segment semi-length, valid in [0, 1] (default to `0`).
            dy (float): Scaling factor proportional to the section height (default to `1.25`).
            i (int | None): Segment ID (default to `None`).
        """
        if not isinstance(section, Section): 
            raise TypeError('Segment must be initialized with a Section object')
        if not all(b in (0, 1) for b in shape) : 
            raise TypeError('Segment shape must be initialized with a list of int (`0` or `1`)')
        if num_division < 2: 
            raise ValueError('The number of divisions must be >= 2')
        if not length > 0.: 
            raise ValueError('The segment length must be positive')
        if dx is not None and not 0. <= dx <= 1. : 
            raise ValueError('The caling factor proportional to the segment semi-length must be in [0, 1]')
        if dy is not None and dy < 1.: 
            raise ValueError('The scaling factor proportional to the section height must be >= 1')
        
        self._section = section
        self._shape = shape
        self._num_division = int(num_division)        
        self._length = float(length)
        self._dx = float(dx) if dx is not None else None
        self._dy = float(dy) if dy is not None else None
        self._i = int(i) if i is not None else None

        self._nl_ctrl = []
        self._nl_fiber = []
        self._sl = []
        self._el = []

        self._assembly: bool = False
        self._generators: list[Segment] | None = None

        self._build_ctrl()
        self._build_fiber()
        self._build_sections()
        self._orient_sections()
        self._build_elements()
        
        self._frame = Frame(origin = op.centroid(self.nodelist))
        self.centre()
    
    # ---- General methods ----
    def copy(self) -> Segment:
        """
        Deep copy of the segment.
        """
        return copy.deepcopy(self)
    
    # ---- Class methods ----
    @classmethod
    def empty(cls) -> Segment:
        """
        Make a Segment object without calling __init__ (no auto-build).
        """
        obj = cls.__new__(cls)
        obj._section = None
        obj._shape = None
        obj._num_division = None
        obj._length = None
        obj._dx = None
        obj._dy = None
        obj._i = None
        obj._nl_ctrl = []
        obj._nl_fiber = []
        obj._sl = []
        obj._el = []
        obj._assembly = False
        obj._generators = None
        obj._frame = None
        return obj
    
    @classmethod
    def from_assembly(cls, segmentlist: list[Segment]) -> Segment:
        """
        Build a new Segment object by merging other Segment objects.
        """
        if not all(isinstance(sg, Segment) for sg in segmentlist) : 
            raise TypeError('Segment assembly must be initialized with a list of Segment objects')
        ref = segmentlist[0].section
        if any(
            (sg.section.dim != ref.dim or
             sg.section.centroid.x != ref.centroid.x or
             sg.section.centroid.y != ref.centroid.y or
             sg.section.centroid.z != ref.centroid.z)
            for sg in segmentlist[1:]):
            raise ValueError('The list of Segment objects must have the same section')
            
        segment = cls.empty()
        
        segment._section = copy.deepcopy(segmentlist[0].section)
        segment._shape = segmentlist[0].shape + [x for sg in segmentlist[1:] for x in sg.shape[1:]]
        segment._num_division = segmentlist[0].num_division + np.sum([sg.num_division - 1 for sg in segmentlist[1:]])
        segment._length = np.sum([sg._length for sg in segmentlist])
        segment._dx = None
        segment._dy = None
        segment._nl_ctrl = [n.copy() for n in segmentlist[0].nodelist_ctrl] + [n.copy() for sg in segmentlist[1:] for n in sg.nodelist_ctrl[1:]]
        segment._nl_fiber = [n.copy() for n in segmentlist[0].nodelist_fiber] + [n.copy() for sg in segmentlist[1:] for n in sg.nodelist_fiber[1:]]
        segment._sl = []
        segment._el = []
        segment._assembly = True
        segment._generators = [sg for sg in segmentlist]
        for sg in segmentlist[1:]: segment._frame = op.average_frame(segmentlist[0].frame, sg.frame)
        
        op.renumber(segment.nodelist_ctrl)
        op.renumber(segment.nodelist_fiber)
        
        segment._build_sections()
        segment._orient_sections()
        segment._build_elements()
        segment.centre()
        
        return segment
    
    # ---- Geometric methods ----
    def centre(self, node: Node | None = None) -> None:
        """
        Centre the segment around the specified target node (default around [0, 0, 0]).
        
        Args:
            node (Node | None): Target node (default to `None`).
        """
        if (node is not None) and (not isinstance(node, Node)):
            raise TypeError('The target node must be initialized either with a Node object or `None` (corresponding to [0, 0, 0])')
            
        n = node if node is not None else Node(0, 0, 0)
        c = self.centroid
        dx, dy, dz = n.x - c.x, n.y - c.y, n.z - c.z
        self.translate(dx, dy, dz)
        return self

    def translate(self, dx: float = 0., dy: float = 0., dz: float = 0.) -> None:
        """
        Translate segment.
        
        Args:
            dx (float): Displacement in the x-direction (default to `0`).
            dy (float): Displacement in the y-direction (default to `0`).
            dz (float): Displacement in the z-direction (default to `0`).
        """
        op.translate(self._nl_ctrl, dx, dy, dz)
        op.translate(self._nl_fiber, dx, dy, dz)
        for st in self._sl: 
            st.translate(dx, dy, dz)
        return self
    
    def rotate(self, axis: str | np.ndarray, angle: float, node: Node | None = None) -> None:
        """
        Rotate segment around a given axis (global or specified).
        
        Args:
            axis (str | np.ndarray): Axis of rotation (`x`, `y`, `z` for global or specified vector).
            angle (float): Rotation angle in radians.
            node (Node | None): Origin of the rotation (default to `None`, corresponding to centroid).
        """
        n = node if node is not None else self.centroid
        op.rotate(self._nl_ctrl, axis, angle, n)
        op.rotate(self._nl_fiber, axis, angle, n)
        for st in self._sl: 
            st.rotate(axis, angle, n)
        for e in self._el: 
            e.frame.rotate(axis, angle)
        return self
    
    # ---- Private helpers ----
    def _build_ctrl(self) -> None:
        """
        Build the control nodes of the Bezier curve.
        """    
        if self._shape[0] != self._shape[1]:
            shape = [-1 if x == 0 else x for x in self._shape]
            sy = self._section.dim[1]
            
            ns_ctrl = np.array([
                [0., shape[0] * 0.5 * sy * self._dy, 0.],
                [0., shape[0] * 0.5 * sy * self._dy, 0.5 * self._length * (1. + self._dx)],
                [0., shape[1] * 0.5 * sy * self._dy, 0.5 * self._length * (1. + self._dx)],
                [0., shape[1] * 0.5 * sy * self._dy, self._length]])
        else:
            ns_ctrl = np.array([
            [0., 0., 0.],
            [0., 0., self._length]])

        self._nl_ctrl.extend([Node(x = x, y = y, z = z, i = i) for i, (x, y, z) in enumerate(ns_ctrl)])

    def _build_fiber(self) -> None:
        """
        Build the fiber nodes with Bezier curve.
        """
        nl_yz = np.array([(n.y, n.z) for n in self._nl_ctrl])
        bz_curve = bezier.Curve(nl_yz.transpose(), degree = len(self._nl_ctrl) - 1)
        s_vals = np.linspace(0., 1., self._num_division)
        self._nl_fiber.extend([Node(x = 0., y = y, z = z, i = i) for i, (y, z) in enumerate(np.transpose(bz_curve.evaluate_multi(s_vals)))])
    
    def _build_sections(self) -> None:
        """
        Build the sections along the fiber nodes of the Bezier curve.
        """
        for i, n in enumerate(self._nl_fiber):
            s = self._section.copy()
            s.centre(n)
            s.i = i
            for n in s.nodelist: n.i += i * len(s.nodelist)
            self._sl.append(s)
    
    def _orient_sections(self) -> None:
        """
        Orient the yarn sections along the Bezier curve (rotation of yz-plane around x-axis).
        """
        for i in range(self._num_division):
            if i != 0 and i != self._num_division - 1:
                nprev = self._nl_fiber[i - 1]
                npost = self._nl_fiber[i + 1]

                tangent = np.array([0.5 * (npost.y - nprev.y), 0.5 * (npost.z - nprev.z)])
                tangent /= np.linalg.norm(tangent)
                angle = np.arctan2(tangent[0], tangent[1])
                
                op.rotate(self._sl[i].nodelist, 'x', -angle)
                self._sl[i].frame.rotate('x', -angle)
    
    def _build_elements(self) -> None:
        """
        Build yarn elements.
        """
        num_col = self._section.num_cols
        num_row = self._section.num_rows
        
        C3D8_matrix = []
        for col in range(num_col - 1):
            for row in range(num_row - 1):
                elem = [
                    (1, col + (row + 1) * num_col + 1),
                    (1, col +  row      * num_col + 1),
                    (1, col +  row      * num_col    ),
                    (1, col + (row + 1) * num_col    ),
                    (2, col + (row + 1) * num_col + 1),
                    (2, col +  row      * num_col + 1),
                    (2, col +  row      * num_col    ),
                    (2, col + (row + 1) * num_col    )]
                C3D8_matrix.append(elem)
        
        for i in range(self._num_division - 1):
            for j, pair in enumerate(C3D8_matrix):
                ei = i * len(C3D8_matrix) + j
                
                nl = []
                for si, ni in pair:
                    nl.append(self._sl[i + si - 1].nodelist[ni])
                
                e = Element(nl, ei)
                e.frame = op.average_frame(self._sl[i].frame, self._sl[i + 1].frame)
                self._el.append(e)
    
    # ---- Geometric properties ----
    @property
    def centroid(self) -> Node:
        """
        Node: Segment centroid.
        """  
        return op.centroid(self.nodelist)
    
    @property
    def dim(self) -> tuple[float, float, float]:
        """
        tuple[float, float, float]: Segment dimension.
        """
        return op.dimensions(self.nodelist)
    
    # ---- Initial properties ----
    @property
    def section(self) -> Section:
        """
        Section: Segment section unit.
        """
        return self._section
    
    @property
    def shape(self) -> list[int]:
        """
        list[int]: Segment shape.
        """
        return self._shape
    
    @property
    def num_division(self) -> float:
        """
        float: Number of division of segment fiber.
        """
        return self._num_division
    
    @property
    def length(self) -> float:
        """
        float: Segment length.
        """
        return self._length
    
    @property
    def dx(self) -> float:
        """
        float: Scaling factor proportional to the segment semi-length, valid in [0, 1].
        """
        if self._assembly: 
            raise ValueError('The segment is built from an assembly')
        return self._dx
    
    @property
    def dy(self) -> float:
        """
        float: Scaling factor proportional to the section height.
        """
        if self._assembly: 
            raise ValueError('The segment is built from an assembly')
        return self._dy
    
    # ---- Helpers properties ----
    @property
    def nodelist_ctrl(self) -> list[Node]:
        """
        list[Node]: List of Bezier curve control nodes.
        """
        return self._nl_ctrl
    
    @property
    def nodelist_fiber(self) -> list[Node]:
        """
        list[Node]: List of fiber nodes.
        """
        return self._nl_fiber
    
    @property
    def assembly(self) -> bool:
        """
        bool: `True` if the segment is built from an assembly.
        """
        return self._assembly
    
    @property
    def generators(self) -> list[Segment]:
        """
        list[Segment]: List of segments of the assembly.
        """
        if not self._assembly: 
            raise ValueError('The segment is not built from an assembly')
        return self._generators
    
    # ---- Output properties ----
    @property
    def sectionlist(self) -> list[Section]:
        """
        list[Section]: List of segment sections.
        """
        return self._sl
    
    @property
    def nodelist(self) -> list[Node]:
        """
        list[Node]: List of segment nodes.
        """
        return [n for st in self._sl for n in st.nodelist]
    
    @property
    def elementlist(self) -> list[Element]:
        """
        list[Element]: List of segment elements.
        """
        return self._el
    
    # ---- Node boundaries ----
    @property
    def nl_xmin(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the x-min boundary.
        """  
        xs = np.array([n.x for n in self.nodelist])
        xmin = xs.min()
        idx = np.where(np.abs(xs - xmin) < tol)[0]
        return [self.nodelist[i] for i in idx]
    
    @property
    def nl_xmax(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the x-max boundary.
        """  
        xs = np.array([n.x for n in self.nodelist])
        xmax = xs.max()
        idx = np.where(np.abs(xs - xmax) < tol)[0]
        return [self.nodelist[i] for i in idx]
    
    @property
    def nl_ymin(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the y-min boundary.
        """  
        ys = np.array([n.y for n in self.nodelist])
        ymin = ys.min()
        idx = np.where(np.abs(ys - ymin) < tol)[0]
        return [self.nodelist[i] for i in idx]
    
    @property
    def nl_ymax(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the y-max boundary.
        """  
        ys = np.array([n.y for n in self.nodelist])
        ymax = ys.max()
        idx = np.where(np.abs(ys - ymax) < tol)[0]
        return [self.nodelist[i] for i in idx]
    
    @property
    def nl_zmin(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the z-min boundary.
        """  
        zs = np.array([n.z for n in self.nodelist])
        zmin = zs.min()
        idx = np.where(np.abs(zs - zmin) < tol)[0]
        return [self.nodelist[i] for i in idx]
    
    @property
    def nl_zmax(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the z-max boundary.
        """  
        zs = np.array([n.z for n in self.nodelist])
        zmax = zs.max()
        idx = np.where(np.abs(zs - zmax) < tol)[0]
        return [self.nodelist[i] for i in idx]
    
    # ---- Element boundaries ----
    @property
    def el_xmin(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the x-min boundary.
        """  
        xs = np.array([e.centroid.x for e in self._el])
        xmin = xs.min()
        idx = np.where(np.abs(xs - xmin) < tol)[0]
        return [self._el[i] for i in idx]
    
    @property
    def el_xmax(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the x-max boundary.
        """  
        xs = np.array([e.centroid.x for e in self._el])
        xmax = xs.max()
        idx = np.where(np.abs(xs - xmax) < tol)[0]
        return [self._el[i] for i in idx]
    
    @property
    def el_ymin(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the y-min boundary.
        """  
        ys = np.array([e.centroid.y for e in self._el])
        ymin = ys.min()
        idx = np.where(np.abs(ys - ymin) < tol)[0]
        return [self._el[i] for i in idx]
    
    @property
    def el_ymax(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the y-max boundary.
        """  
        ys = np.array([e.centroid.y for e in self._el])
        ymax = ys.max()
        idx = np.where(np.abs(ys - ymax) < tol)[0]
        return [self._el[i] for i in idx]
    
    @property
    def el_zmin(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the z-min boundary.
        """  
        zs = np.array([e.centroid.z for e in self._el])
        zmin = zs.min()
        idx = np.where(np.abs(zs - zmin) < tol)[0]
        return [self._el[i] for i in idx]
    
    @property
    def el_zmax(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the z-max boundary.
        """  
        zs = np.array([e.centroid.z for e in self._el])
        zmax = zs.max()
        idx = np.where(np.abs(zs - zmax) < tol)[0]
        return [self._el[i] for i in idx]
    
    # ---- Index ----
    @property
    def i(self) -> int:
        """
        int: Segment ID.
        """
        if self._i is None: 
            raise ValueError('The segment is not indexed')
        return self._i
    @i.setter
    def i(self, i: int) -> None:
        self._i = int(i)
    
    # ---- Frame ----
    @property
    def frame(self) -> Frame:
        """
        Frame: Segment frame.
        """
        self._frame.origin = self.centroid
        return self._frame
    @frame.setter
    def frame(self, f: Frame) -> None:
        if not isinstance(f, Frame): 
            raise TypeError('Segment frame must be a Frame object')
        f.origin = self.centroid
        self._frame = f

    # ---- Plot method ----
    def plot(self,
        node_ctrl: bool = True,
        node_ctrl_idx: bool = False,
        node_fiber: bool = True,
        node_fiber_idx: bool = False,
        node: bool = True,
        node_idx: bool = False,
        element: bool = True,
        element_idx: bool = False,
        frame: bool = False,
        frame_section: bool = False,
        frame_element: bool = True,
        legend: bool = True, 
        grid: bool = True,
        save: bool = False,
        save_dir: str = os.path.join(os.getcwd(), 'img'),
        save_fmt: str = 'png') -> None:
        """
        Plot the segment.

        Args:
            node_ctrl (bool): Show control nodes (default to `True`).
            node_ctrl_idx (bool): Show control node indices (default to `False`).
            node_fiber (bool): Show fiber nodes (default to `True`).
            node_fiber_idx (bool): Show fiber node indices (default to `False`).
            node (bool): Show nodes (default to `True`).
            node_idx (bool): Show node indices (default to `False`).
            element (bool): Show elements (default to `True`).
            element_idx (bool): Show element indices (default to `False`).
            frame (bool): Show segment frame (default to `False`).
            frame_section (bool): Show section frames (default to `False`).
            frame_element (bool): Show section frames (default to `True`).
            legend (bool): Show legend (default to `True`).
            grid (bool): Show grid. (default to `True`)
            save (bool): Save option (default to `False`).
            save_dir (str): Save directory (default to current directory).
            save_fmt (str): File format (default to `png`).
        """        
        if save: os.makedirs(save_dir, exist_ok = True)
        filename = os.path.join(save_dir, f'segment.{save_fmt}')
        
        nl = []
        if node_ctrl: nl.append((self._nl_ctrl, 'orange', '-', node_ctrl_idx))
        if node_fiber: nl.append((self._nl_fiber, 'blue', '-', node_fiber_idx))
        if node: nl.append((self.nodelist, 'red', None, node_idx))
        
        el = []
        if element: el.append((self._el, 'red', element_idx))
        
        fl = []
        if frame: fl.append(self.frame)
        if frame_section: fl.extend([st.frame for st in self._sl])
        if frame_element: fl.extend([e.frame for e in self._el])
        
        pt.plot_pv(nl, el, fl, grid, save, filename)

__all__ = ['Segment']    
    
    
    

    
    
    
    
    

    
    
    
    
    
    
    
    
        
    
    
        
    
            
    
    
   
    
    
            
    
    
    

    
    
    
    
    
    

    
    
    
    
    

    
    
    
    
    
    
    
    
        
    
    
    






