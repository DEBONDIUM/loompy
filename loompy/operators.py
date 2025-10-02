#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:58:12 2025

@author: lbremaud
"""
# =============================================================================
# LIB
# =============================================================================
from __future__ import annotations

import numpy as np

from .node import Node
from .frame import Frame

# =============================================================================
# OPERATORS
# =============================================================================
def dimensions(nodelist: list[Node]) -> tuple[float, float, float]:
    """
    Compute the dimesions of the node list.
    
    Args:
        nodelist (list[Node]): List of nodes.
    Returns:
        tuple[float, float, float]: Dimensions.
    """
    if not all(isinstance(n, Node) for n in nodelist):
        raise TypeError('The centroid must be computed from Node objects')
        
    xyz = np.array([(n.x, n.y, n.z) for n in nodelist])
    return tuple(xyz.max(axis = 0) - xyz.min(axis = 0))

def centroid(nodelist: list[Node]) -> Node:
    """
    Compute the centroid of the node list.
    
    Args:
        nodelist (list[Node]): List of nodes.
    Returns:
        Node: Centroid.
    """
    if not all(isinstance(n, Node) for n in nodelist):
        raise TypeError('The centroid must be computed from Node objects')
        
    xyz = np.array([(n.x, n.y, n.z) for n in nodelist])
    cx, cy, cz = xyz.mean(axis = 0)
    
    return Node(cx, cy, cz)

def translate(nodelist: list[Node], dx: float, dy: float, dz: float) -> None:
    """
    Translate nodes.
    
    Args:
        nodelist (list[Node]): List of nodes to translate.
        dx (float): Displacement in the x-direction.
        dy (float): Displacement in the y-direction.
        dz (float): Displacement in the z-direction.
    """
    if not all(isinstance(n, Node) for n in nodelist):
        raise TypeError('The translate method must be computed from Nodes objects')
    
    for n in nodelist:
        n.x += float(dx)
        n.y += float(dy)
        n.z += float(dz)

def rotate(nodelist: list[Node], axis: str | np.ndarray, angle: float, node: Node | None = None) -> None:
    """
    Rotate nodes around a given axis (global or specified).
    
    Args:
        nodelist (list[Node]): List of nodes to rotate.
        axis (str | np.ndarray): Axis of rotation (`x`, `y`, `z` for global or specified vector).
        angle (float): Rotation angle in radians.
        node (Node | None): Origin of the rotation. 
    """
    if not all(isinstance(n, Node) for n in nodelist):
        raise TypeError('The rotate method must be computed from Nodes objects')
    
    if isinstance(axis, str):
        if axis == 'x':   v = np.array([1., 0., 0.])
        elif axis == 'y': v = np.array([0., 1., 0.])
        elif axis == 'z': v = np.array([0., 0., 1.])
        else: raise ValueError('Axis must be `x`, `y`, `z`, or a vector')
    elif not axis.shape == (3,): 
        raise TypeError('The rotate method must be computed from an axis of shape (3,)')
    else:
        v = np.asarray(axis, dtype = float)
        
    R = rotation_matrix(v, angle)
    
    if node == None: 
        node = centroid(nodelist)
    
    xyz = np.array([(n.x, n.y, n.z) for n in nodelist])
    xyz_rot = np.dot(xyz - np.array([node.x, node.y, node.z]), R.T) + np.array([node.x, node.y, node.z])
    
    for i, n in enumerate(nodelist):
        n.x = xyz_rot[i, 0]
        n.y = xyz_rot[i, 1]
        n.z = xyz_rot[i, 2]
    
# =============================================================================
# HELPERS
# =============================================================================
def renumber(objects: list[object], offset: int = 0):
    for i, o in enumerate(objects):
        o.i = i + offset

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Compute rotation matrix around axis using Rodrigues' formula.
    
    Args:
        axis (np.ndarray): Axis of rotation.
        angle (float): Rotation angle.
    Returns:
        np.ndarray: Rotation matrix.
    """
    if not axis.shape == (3,): 
        raise TypeError('The rotation matrix must be computed from an axis of shape (3,)')
    
    axis = axis / np.linalg.norm(axis)
    
    c, s = np.cos(float(angle)), np.sin(float(angle))
    
    K = np.array([[0., -axis[2], axis[1]],
                  [ axis[2], 0., -axis[0]],
                  [-axis[1], axis[0], 0.]])
    
    R = np.eye(3) + s * K + (1 - c) * np.dot(K, K)
    
    return R
    
def average_frame(frame1: Frame, frame2: Frame) -> Frame:
    """
    Average two frames.

    Args:
        frame1 (Frame): First frame.
        frame2 (Frame): Second frame.
    Returns:
        Frame: Averaged frame.
    """
    if not isinstance(frame1, Frame) or not isinstance(frame2, Frame):
        raise TypeError('The averaging must be computed from Frame objects')
    
    vx = frame1.vx + frame2.vx
    vy = frame1.vy + frame2.vy
    vz = frame1.vz + frame2.vz

    vx = vx / np.linalg.norm(vx)
    vy = vy / np.linalg.norm(vy) 
    vz = vz / np.linalg.norm(vz)

    vx = vx / np.linalg.norm(vx)
    
    vz = np.cross(vx, vy)
    vz /= np.linalg.norm(vz)
    
    vy = np.cross(vz, vx)
    vy /= np.linalg.norm(vy)
    
    c1 = frame1.origin
    c2 = frame2.origin
    c = Node(0.5 * (c1.x + c2.x), 0.5 * (c1.y + c2.y), 0.5 * (c1.z + c2.z))

    return Frame(c, vx, vy, vz)


