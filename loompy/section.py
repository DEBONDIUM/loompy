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

import numpy as np
import os
import copy

from . import operators as op
from . import plotter as pt

from .node import Node
from .frame import Frame

# =============================================================================
# CLASS SECTION
# =============================================================================
class Section:
    def __init__(self, 
        nodelist: list[Node],
        num_rows: int,
        num_cols: int,
        i: int | None = None):
        """
        Define a section.
        
        Args:
            nodelist (list[Node]): List of nodes.
            num_rows (int): Number of rows on the grid-organized section.
            num_cols (int): Number of columns on the grid-organized section.
            i (int | None): Section ID (default to `None`).
        """
        if not all(isinstance(n, Node) for n in nodelist):
            raise TypeError('Section must be initialized with a NodeList object')
            
        self._nodelist = nodelist
        self._num_rows = int(num_rows)
        self._num_cols = int(num_cols)
        self._i = int(i) if i is not None else None
        
        self._frame = Frame(origin = self.centroid)
        self.centre()
    
    # ---- General methods ----
    def copy(self) -> Section:
        """
        Deep copy of the section.
        """
        return copy.deepcopy(self)
        
    # ---- Class methods ----
    @classmethod
    def from_file(cls, 
        filename: str, 
        num_rows: int, 
        num_cols: int) -> 'Section':
        """
        Read nodes from a text file with 2 columns.
    
        Args:
            filename (str): Path to the .txt file.
            num_rows (int): Number of rows on the grid-organized section.
            num_cols (int): Number of columns on the grid-organized section.
        Returns:
            Section: Section object.
        """
        data = np.loadtxt(filename)
        
        if data.shape[1] != 2: 
            raise ValueError('File must have 2 columns')
        
        if not cls._check_nodes_order(data, num_rows, num_cols):
            print('The node positions have been reordered')
            data = cls._reorder_nodes(data, num_rows, num_cols)
        
        nodelist = [Node(x = float(row[0]), y = float(row[1]), z = 0., i = j) for j, row in enumerate(data)]
        
        return cls(nodelist = nodelist, num_rows = num_rows, num_cols = num_cols)
    
    @classmethod
    def from_array(cls, 
        arr: np.ndarray, 
        num_rows: int, 
        num_cols: int) -> 'Section':
        """
        Read nodes from an array.
    
        Args:
            arr (np.ndarray): Array of the node coordinates.
            num_rows (int): Number of rows on the grid-organized section.
            num_cols (int): Number of columns on the grid-organized section.
        Returns:
            Section: Section object.
        """
        if arr.shape[1] != 2: 
            raise ValueError('Array must have shape (n, 2)')
        
        if not cls._check_nodes_order(arr, num_rows, num_cols):
            print('The node positions have been reordered')
            arr = cls._reorder_nodes(arr, num_rows, num_cols)
        
        nodelist = [Node(x = float(row[0]), y = float(row[1]), z = 0., i = j) for j, row in enumerate(arr)]
        
        return cls(nodelist = nodelist, num_rows = num_rows, num_cols = num_cols)
    
    # ---- Static methods ----
    @staticmethod
    def _check_nodes_order(arr: np.ndarray, num_rows: int, num_cols: int) -> bool:
        """
        Nodes order verification.
        
        Args:
            arr (np.ndarray): Shape (n, 2) array of [x, y] points.
            num_rows (int): Number of rows on the grid-organized section.
            num_cols (int): Number of columns on the grid-organized section.
        Returns:
            bool: `True` if nodes order is correct.
        """
        grid = arr.reshape(num_rows, num_cols, 2)
        
        rows_ok = all(np.all(np.diff(row[:, 0]) > 0) for row in grid)
        cols_ok = all(np.all(np.diff(grid[:, j, 1]) < 0) for j in range(num_cols))
        
        return rows_ok and cols_ok
    
    @staticmethod
    def _reorder_nodes(arr: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
        """
        Reorder 2D coordinates into row-major order:
        - rows sorted by decreasing y (top -> bottom)
        - within each row, sorted by increasing x (left -> right)
    
        Args:
            arr (np.ndarray): Shape (n, 2) array of [x, y] points.
            num_rows (int): Number of rows on the grid-organized section.
            num_cols (int): Number of columns on the grid-organized section.
        Returns:
            np.ndarray: Reordered node coordinates
        """
        keys_y = np.argsort(arr[:, 1])[::-1]
        
        sorted_points = []
        for row_start in range(0, len(arr), num_cols):
            row_indices = keys_y[row_start:row_start + num_cols]
            row_points = arr[row_indices]
            
            keys_x = np.argsort(row_points[:, 0])
            row_sorted = row_points[keys_x]
            
            sorted_points.append(row_sorted)
        
        return np.vstack(sorted_points)
    
    # ---- Geometric methods ----
    def centre(self, node: Node | None = None) -> None:
        """
        Centre the nodes around the specified target node (default around [0, 0, 0]).
        
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
        Translate nodes.
        
        Args:
            dx (float): Displacement in the x-direction (default to `0`).
            dy (float): Displacement in the y-direction (default to `0`).
            dz (float): Displacement in the z-direction (default to `0`).
        """
        op.translate(self._nodelist, dx, dy, dz)
        return self
    
    def rotate(self, axis: str | np.ndarray, angle: float, node: Node | None = None) -> None:
        """
        Rotate nodes around a given axis (global or specified).
        
        Args:
            axis (str | np.ndarray): Axis of rotation (`x`, `y`, `z` for global or specified vector).
            angle (float): Rotation angle in radians.
            node (Node | None): Origin of the rotation (default to `None`, corresponding to centroid).
        """
        op.rotate(self._nodelist, axis, angle, node)
        self.frame.rotate(axis, angle)
        return self
    
    # ---- Geometric properties ----
    @property
    def centroid(self) -> Node:
        """
        Node: Centroid.
        """
        return op.centroid(self._nodelist)
    
    @property
    def dim(self) -> tuple[float, float, float]:
        """
        tuple[float, float, float]: Dimensions.
        """
        return op.dimensions(self._nodelist)
    
    # ---- Section properties ----
    @property
    def nodelist(self) -> list[Node]:
        """
        list[Node]: List of nodes.
        """
        return self._nodelist
    
    @property
    def num_rows(self) -> int:
        """
        int: Number of rows on the grid-organized section.
        """
        return self._num_rows
    
    @property
    def num_cols(self) -> int:
        """
        int: Number of columns on the grid-organized section.
        """
        return self._num_cols
    
    # ---- Index ----
    @property
    def i(self) -> int:
        """
        int: Section ID.
        """
        if self._i is None:
            raise ValueError('The section is not indexed')
        return self._i
    @i.setter
    def i(self, i: int) -> None:
        self._i = int(i)
    
    # ---- Frame ----
    @property
    def frame(self) -> Frame:
        """
        Frame: Section frame.
        """
        self._frame.origin = self.centroid
        return self._frame
    @frame.setter
    def frame(self, f: Frame) -> None:
        if not isinstance(f, Frame): 
            raise TypeError('Section frame must be a Frame object')
        self._frame = f
        self._frame.origin = self.centroid

    # ---- Plot method ----
    def plot(self,
        node_idx: bool = True,
        frame: bool = True,
        legend: bool = True, 
        grid: bool = True,
        save: bool = False,
        save_dir: str = os.path.join(os.getcwd(), 'img'),
        save_fmt: str = 'png') -> None:
        """
        Plot the section.

        Args:
            node_idx (bool): Show node indices (default to `True`).
            frame (bool): Show section frame (default to `True`).
            legend (bool): Show legend (default to `True`).
            grid (bool): Show grid (default to `True`).
            save (bool): Save option (default to `False`).
            save_dir (str): Save directory (default to current directory).
            save_fmt (str): File format (default to `png`).
        """        
        if save: os.makedirs(save_dir, exist_ok = True)
        filename = os.path.join(save_dir, f'section.{save_fmt}')
        
        nl = [(self._nodelist, 'red', None, node_idx)]
        
        fl = []
        if frame: fl.append(self.frame)
            
        pt.plot_pv(nl, [], fl, grid, save, filename)

__all__ = ['Section']    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    

    
    
    
    
    
    
        
    

    
            

