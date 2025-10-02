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

# =============================================================================
# CLASS NODE
# =============================================================================
class Node:
    def __init__(self, x: float, y: float, z: float, i: int | None = None):
        """
        Define a node.
        
        Args:
            x (float): Node x-coordinate.
            y (float): Node y-coordinate.
            z (float): Node z-coordinate.
            i (int | None): Node ID (default to `None`).
        """
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        self._i = int(i) if i is not None else None
    
    # ---- General methods ----
    def copy(self) -> Node:
        """
        Deep copy of the node.
        """
        return copy.deepcopy(self)
    
    # ---- Index ----
    @property
    def i(self) -> int:
        """
        int: Node ID.
        """
        if self._i is None:
            raise ValueError('The node is not indexed')
        return self._i
    @i.setter
    def i(self, i: int) -> None:
        self._i = int(i)
    
    # ---- Coordinates ----
    @property
    def x(self) -> float:
        """
        float: Node x-coordinate.
        """
        return self._x
    @x.setter
    def x(self, x: float) -> None:
        self._x = float(x)
    
    @property
    def y(self) -> float:
        """
        float: Node y-coordinate.
        """
        return self._y
    @y.setter
    def y(self, y: float) -> None:
        self._y = float(y)
    
    @property
    def z(self) -> float:
        """
        float: Node z-coordinate.
        """
        return self._z
    @z.setter
    def z(self, z: float) -> None:
        self._z = float(z)
    
__all__ = ['Node']     

        

