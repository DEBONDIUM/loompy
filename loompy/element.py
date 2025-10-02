#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:29:26 2025

@author: lbremaud
"""

# =============================================================================
# LIB
# =============================================================================
from __future__ import annotations

import copy

from . import operators as op

from .node import Node
from .frame import Frame

# =============================================================================
# CLASS ELEMENT
# =============================================================================
class Element:
    def __init__(self, nodelist: list[Node], i: int | None = None):
        """
        Define an element.
        
        Args:
            nodelist (list[Node]): Element nodes.
            i (int | None, optional): Element ID (default to `None`).
        """
        if not all(isinstance(n, Node) for n in nodelist):
            raise TypeError('Element must be initialized with Node objects')
            
        self._nodelist = nodelist
        self._i = int(i) if i is not None else None
        
        self._frame = Frame(origin = op.centroid(nodelist))
    
    # ---- General methods ----
    def copy(self) -> Element:
        """
        Deep copy of the element.
        """
        return copy.deepcopy(self)
    
    # ---- Node properties----
    @property
    def nodelist(self) -> list[Node]:
        """
        list[Node]: List of nodes.
        """
        return self._nodelist
    
    @property
    def connectivity(self) -> list[tuple[int, int]]:
        """
        list[tuple[int, int]]: Element node connectivity.
        """
        connection = [
            (0, 1), (1, 2),
            (2, 3), (3, 0),
            (4, 5), (5, 6),
            (6, 7), (7, 4),
            (0, 4), (1, 5),
            (2, 6), (3, 7)]
        return connection
    
    # ---- Geometric properties----
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
    
    # ---- Index ----
    @property
    def i(self) -> int:
        """
        int: Element ID.
        """
        if self._i is None: 
            raise ValueError('The element is not indexed')
        return self._i
    @i.setter
    def i(self, i: int) -> None:
        self._i = int(i)
    
    # ---- Frame ----
    @property
    def frame(self) -> Frame:
        """
        Frame: Element frame.
        """
        self._frame.origin = self.centroid
        return self._frame
    @frame.setter
    def frame(self, f: Frame) -> None:
        if not isinstance(f, Frame): 
            raise TypeError('Element frame must be a Frame object')
        f.origin = self.centroid
        self._frame = f
    
__all__ = ['Element']    
