#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 09:22:12 2025

@author: lbremaud
"""

# =============================================================================
# LIB
# =============================================================================
from __future__ import annotations

import numpy as np
import copy

from . import operators as op

from .node import Node

# =============================================================================
# CLASS FRAME
# =============================================================================
class Frame:
    def __init__(self, 
        origin: Node | None = None, 
        vx: np.ndarray | None = None, 
        vy: np.ndarray | None = None, 
        vz: np.ndarray | None = None):
        """
        Define a frame.
        
        Args:
            origin (Node): Origin (default to `[0, 0, 0]`).
            vx (np.ndarray): Vector frame on the x-axis (default to `[1, 0, 0]`).
            vy (np.ndarray): Vector frame on the y-axis (default to `[0, 1, 0]`).
            vz (np.ndarray): Vector frame on the z-axis (default to `[0, 0, 1]`).
        """
        if origin is None:
            origin = Node(0., 0., 0.)
        if not isinstance(origin, Node):
            raise TypeError('Origin must be a Node')

        self._origin = origin
        self._vx = np.array([1., 0., 0.]) if vx is None else np.asarray(vx, dtype=float)
        self._vy = np.array([0., 1., 0.]) if vy is None else np.asarray(vy, dtype=float)
        self._vz = np.array([0., 0., 1.]) if vz is None else np.asarray(vz, dtype=float)

        for vec in [self._vx, self._vy, self._vz]:
            if vec.shape != (3,):
                raise ValueError('Frame vectors must have shape (3,)')
    
    # ---- General methods ----
    def copy(self) -> Frame:
        """
        Deep copy of the frame.
        """
        return copy.deepcopy(self)
    
    # ---- Geometric methods ----
    def rotate(self, axis: str | np.ndarray, angle: float) -> None:
        """
        Rotate frame around a given axis (global or specified).
        
        Args:
            axis (str | np.ndarray): Axis of rotation (`x`, `y`, `z` for global or specified vector).
            angle (float): Rotation angle in radians.
        """        
        if isinstance(axis, str):
            if axis == 'x': 
                v = np.array([1., 0., 0.])
            elif axis == 'y': 
                v = np.array([0., 1., 0.])
            elif axis == 'z': 
                v = np.array([0., 0., 1.])
            else: 
                raise ValueError('Axis must be `x`, `y`, `z`, or a vector')
        elif not axis.shape == (3,): 
            raise TypeError('The rotation must be computed from an axis of shape (3,)')
        else:
            v = np.asarray(axis, dtype = float)

        R = op.rotation_matrix(v, float(angle))
        
        self._vx = np.dot(R, self._vx)
        self._vy = np.dot(R, self._vy)
        self._vz = np.dot(R, self._vz)

        if np.linalg.det(self.matrix) < 0:
            self._vz *= -1
        return self
   
    # ---- Origin ----    
    @property
    def origin(self) -> Node:
        """
        int: Frame origin.
        """
        return self._origin
    @origin.setter
    def origin(self, o: Node) -> None:
        if not isinstance(o, Node): 
            raise TypeError('Origin must be initialized with a Node object')
        self._origin = o
    
    # ---- Matrix ----   
    @property
    def matrix(self) -> np.ndarray:
        """
        np.ndarray: Frame with [vx vy vz] as columns.
        """
        return np.column_stack([self._vx, self._vy, self._vz])
    @matrix.setter
    def matrix(self, m: np.ndarray) -> None:
        if m.shape != (3, 3): 
            raise ValueError('Frame matrix must be a 3x3 matrix with [vx vy vz] as columns')
        self._vx = m[:, 0]
        self._vy = m[:, 1]
        self._vz = m[:, 2]
    
    # ---- Vectors ----   
    @property
    def vx(self) -> np.ndarray:
        """
        np.ndarray: Vector frame on the x-axis.
        """
        return self._vx
    @vx.setter
    def vx(self, vx: np.ndarray) -> None:
        if vx.shape != (3,): 
            raise ValueError('Frame vectors should be on the shape (3,)')
        self._vx = vx
    
    @property
    def vy(self) -> np.ndarray:
        """
        np.ndarray: Vector frame on the y-axis.
        """
        return self._vy
    @vy.setter
    def vy(self, vy: np.ndarray) -> None:
        if vy.shape != (3,): 
            raise ValueError('Frame vectors should be on the shape (3,)')
        self._vy = vy
    
    @property
    def vz(self) -> np.ndarray:
        """
        np.ndarray: Vector frame on the z-axis.
        """
        return self._vz
    @vz.setter
    def vz(self, vz: np.ndarray) -> None:
        if vz.shape != (3,): 
            raise ValueError('Frame vectors should be on the shape (3,)')
        self._vz = vz

__all__ = ['Frame']    



