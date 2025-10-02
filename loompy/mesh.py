#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 2 10:57:18 2025

@author: lbremaud
"""

# =============================================================================
# LIB
# =============================================================================
from __future__ import annotations

import numpy as np
import os
import copy

from . import plotter as pt
from . import operators as op

from .frame import Frame
from .node import Node
from .element import Element
from .segment import Segment

# =============================================================================
# CLASS MESH
# =============================================================================
class Mesh():
    def __init__(self,
        warp: list[Segment],
        weft: list[Segment],
        space: float,
        num_warp: int = 1,
        num_weft: int = 1,
        num_segment_warp: int = 1,
        num_segment_weft: int = 1,):
        """
        Define the mesh.
        
        Args:
            warp (list[Segment]): Segment objects list compsing the warp part of the mesh.
            weft (list[Segment]): Segment objects list compsing the weft part of the mesh.
            space (int): Space between segments.
            num_warp (int): Number of warp segments (default to `1`).
            num_weft (int): Number of weft segments (default to `1`).
            num_segment_warp (int): Number of unit segments repeated on warp (default to `1`).
            num_segment_weft (int): Number of unit segments repeated on weft (default to `1`).
        """
        if not all(isinstance(sg, Segment) for sg in warp + weft):
            raise TypeError('Mesh must be initialized with lists of Segment objects')
        
        if len(warp) == 0: num_warp = num_segment_warp = 0
        if len(weft) == 0: num_weft = num_segment_weft = 0
        
        if space < 0.: raise ValueError('The space between segments should be positive')
        
        if num_warp == 0 and num_segment_warp != 0: raise ValueError('If `num_warp` == 0, then `num_segment_warp` must be 0')
        if num_warp > 0 and num_segment_warp < 1: raise ValueError("If `num_warp` > 0, then `num_segment_warp` must be >= 1")
        
        if num_weft == 0 and num_segment_weft != 0: raise ValueError('If `num_weft` == 0, then `num_segment_weft` must be 0')
        if num_weft > 0 and num_segment_weft < 1: raise ValueError("If `num_weft` > 0, then `num_segment_weft` must be >= 1")
                
        self._warp = warp
        self._weft = weft
        self._space = float(space)
        self._num_warp = num_warp
        self._num_weft = num_weft
        self._num_segment_warp = num_segment_warp
        self._num_segment_weft = num_segment_weft
        
        self._segmentlist_warp = []
        self._segmentlist_weft = []
        
        self._build_warp()
        self._build_weft()
        self._reorder_indices()
        self.centre()
    
    # ---- General methods ----
    def copy(self) -> Mesh:
        """
        Deep copy of the mesh.
        """
        return copy.deepcopy(self)
    
    # ---- Geometric methods ----
    def centre(self, node: Node | None = None) -> None:
        """
        Centre the mesh around the specified target node (default around [0, 0, 0]).
        
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
        Translate mesh.
        
        Args:
            dx (float): Displacement in the x-direction (default to `0`).
            dy (float): Displacement in the y-direction (default to `0`).
            dz (float): Displacement in the z-direction (default to `0`).
        """
        for sg in self._segmentlist_warp + self._segmentlist_weft: 
            sg.translate(dx, dy, dz)
        return self
    
    def rotate(self, axis: str | np.ndarray, angle: float, node: Node | None = None) -> None:
        """
        Rotate mesh around a given axis (global or specified).
        
        Args:
            axis (str | np.ndarray): Axis of rotation (`x`, `y`, `z` for global or specified vector).
            angle (float): Rotation angle in radians.
            node (Node | None): Origin of the rotation (default to `None`, corresponding to centroid).
        """
        n = node if node is not None else self.centroid
        for sg in self._segmentlist_warp + self._segmentlist_weft: 
            sg.rotate(axis, angle, n)
        self.frame.rotate(axis, angle)
        return self
        
    # ---- Export methods ----
    def export(self,
        save_fmt: str = 'msh',
        filename: str = 'mesh',
        save_dir: str = os.path.join(os.getcwd(), 'mesh')) -> None:
        """
        Export mesh.

        Args:
            save_fmt (str): Mesh file format (`msh` or `inp`).
            filename (str): File name.
            save_dir (str): Directory to save mesh file.
        """
        os.makedirs(save_dir, exist_ok = True)
        
        if save_fmt == 'msh':
            self._export_to_gmsh(filename, save_dir)
        elif save_fmt == 'inp':
            self._export_to_abq(filename, save_dir)
        else:
            raise TypeError('Save format should be `msh` (GMSH) or `inp` (ABAQUS)')

    def export_ori(self, filename: str = 'orientations', save_dir: str = os.path.join(os.getcwd(), 'mesh')) -> None:
        """
        Export elements orientation vectors.

        Args:
            filename (str): File name.
            save_dir (str): Directory to save file.
        """
        os.makedirs(save_dir, exist_ok = True)
        with open(os.path.join(save_dir, filename + '.ori'), 'w') as f:
            f.write('********************\n'
                    '** ORIENTATIONS **\n'
                    '********************\n'
                    '** 1st vector represents the fiber direction\n'
                    '** 2nd vector is an arbitrary vector perpendicular to the first\n'
                    ', 1.0, 0.0, 0.0,   0.0, 1.0, 0.0\n')

            for e in self.elementlist:
                vy = ', '.join(f"{float(v):.10f}" for v in e.frame.vy)
                vz = ', '.join(f"{float(v):.10f}" for v in e.frame.vz)
                f.write(f"{int(e.i + 1)}, {vy},\t{vz}\n")
    
    # ---- Private helpers ----
    def _reorder_indices(self):
        op.renumber([sg for sg in (self._segmentlist_warp + self._segmentlist_weft)])
        op.renumber([n for sg in (self._segmentlist_warp + self._segmentlist_weft) for n in sg.nodelist_ctrl])
        op.renumber([n for sg in (self._segmentlist_warp + self._segmentlist_weft) for n in sg.nodelist_fiber])
        op.renumber([sl for sg in (self._segmentlist_warp + self._segmentlist_weft) for sl in sg.sectionlist])
        op.renumber([n for sg in (self._segmentlist_warp + self._segmentlist_weft) for n in sg.nodelist])
        op.renumber([e for sg in (self._segmentlist_warp + self._segmentlist_weft) for e in sg.elementlist])
        
    def _build_warp(self) -> None:
        """
        Build the warp part of the mesh.
        """
        offset_lx = 0.
        for i in range(self._num_warp):
            offset_lz = 0.
            for j in range(self._num_segment_warp):
                s = self._warp[i % len(self._warp)].copy()
                s.translate(dx = offset_lx, dz = offset_lz)
                offset_lz += s.length
                self._segmentlist_warp.append(s)
            
            offset_lx += self._space
    
    def _build_weft(self) -> None:
        """
        Build the weft part of the mesh.
        """
        offset_lz = 0.
        for i in range(self._num_weft):
            offset_lx = 0.
            for j in range(self._num_segment_weft):
                s = self._weft[i % len(self._warp)].copy()
                s.translate(dx = offset_lx, dz = offset_lz)
                offset_lx += s.length
                self._segmentlist_weft.append(s)
            
            offset_lz += self._space

    def _export_to_gmsh(self, filename: str, save_dir: str) -> None:
        """
        Export mesh to a GMSH format.

        Args:
            filename (str): File name.
            save_dir (str): Directory to save mesh file.
        """
        with open(os.path.join(save_dir, filename + '.msh'), 'w') as f:
            f.write('$MeshFormat\n')
            f.write('2.2 0 8\n')
            f.write('$EndMeshFormat\n')

            f.write('$Nodes\n')
            f.write(f'{len(self.nodelist)}\n')
            
            for n in self.nodelist:
                f.write(f'{int(n.i)} {float(n.x):.10f} {float(n.y):.10f} {float(n.z):.10f}\n')
            f.write('$EndNodes\n')

            f.write('$Elements\n')
            f.write(f'{len(self.elementlist)}\n')
            for e in self.elementlist:
                elem_type = 5
                physical_group = 1
                entity_tag = 1
                nli = ' '.join(map(str, (int(n.i) for n in e.nodelist)))
                f.write(f'{int(e.i)} {elem_type} {physical_group} {entity_tag} {nli}\n')
            f.write('$EndElements\n')
    
    def _export_to_abq(self, filename: str, save_dir: str) -> None:
        """
        Export mesh to a INP format.

        Args:
            filename (str): File name.
            save_dir (str): Directory to save mesh file.
        """        
        with open(os.path.join(save_dir, filename + '.inp'), 'w') as f:
            f.write('*NODE\n')
            for n in self.nodelist:
                f.write(f'{int(n.i + 1)},\t{float(n.x):.10f},\t{float(n.y):.10f},\t{float(n.z):.10f}\n')
            
            for bc in ('nl_xmin_warp', 'nl_xmax_warp', 'nl_ymin_warp', 'nl_ymax_warp', 'nl_zmin_warp', 'nl_zmax_warp',
                       'nl_xmin_weft', 'nl_xmax_weft', 'nl_ymin_weft', 'nl_ymax_weft', 'nl_zmin_weft', 'nl_zmax_weft'):
                try:
                    nli = [int(n.i + 1) for n in getattr(self, bc)]
                except:
                    continue
                f.write(f'*NSET, NSET={bc.split("_")[2].upper()}_{bc.split("_")[1].upper()}\n')
                for i in range(0, len(nli), 16):
                    chunk = nli[i:i + 16]
                    f.write(', '.join(map(str, chunk)) + '\n')
            
            for sg in self._segmentlist_warp:
                f.write(f'*NSET, NSET=WARP_SEGMENT_{int(sg.i)}\n')
                nli = [int(n.i + 1) for n in sg.nodelist]
                for i in range(0, len(nli), 16):
                    chunk = nli[i:i + 16]
                    f.write(', '.join(map(str, chunk)) + '\n')
            
            for sg in self._segmentlist_weft:
                f.write(f'*NSET, NSET=WEFT_SEGMENT_{int(sg.i)}\n')
                nli = [int(n.i + 1) for n in sg.nodelist]
                for i in range(0, len(nli), 16):
                    chunk = nli[i:i + 16]
                    f.write(', '.join(map(str, chunk)) + '\n')
            
            f.write('*ELEMENT, TYPE=C3D8R\n')
            for e in self.elementlist:
                nli = ', '.join(map(str, (int(n.i + 1) for n in e.nodelist)))
                f.write(f'{int(e.i + 1)}, {nli}\n')
            
            for bc in ('el_xmin_warp', 'el_xmax_warp', 'el_ymin_warp', 'el_ymax_warp', 'el_zmin_warp', 'el_zmax_warp',
                       'el_xmin_weft', 'el_xmax_weft', 'el_ymin_weft', 'el_ymax_weft', 'el_zmin_weft', 'el_zmax_weft'):
                try:
                    eli = [int(e.i + 1) for e in getattr(self, bc)]
                except:
                    continue
                f.write(f'*ELSET, ELSET={bc.split("_")[2].upper()}_{bc.split("_")[1].upper()}\n')
                for i in range(0, len(eli), 16):
                    chunk = eli[i:i + 16]
                    f.write(', '.join(map(str, chunk)) + '\n')
            
            for sg in self._segmentlist_warp:
                f.write(f'*ELSET, ELSET=WARP_SEGMENT_{int(sg.i)}\n')
                eli = [int(e.i + 1) for e in sg.elementlist]
                for i in range(0, len(eli), 16):
                    chunk = eli[i:i + 16]
                    f.write(', '.join(map(str, chunk)) + '\n')
            
            for sg in self._segmentlist_weft:
                f.write(f'*ELSET, ELSET=WEFT_SEGMENT_{int(sg.i)}\n')
                eli = [int(e.i + 1) for e in sg.elementlist]
                for i in range(0, len(eli), 16):
                    chunk = eli[i:i + 16]
                    f.write(', '.join(map(str, chunk)) + '\n')
    
    # ---- Geometric properties ----
    @property
    def centroid(self) -> Node:
        """
        Node: Mesh centroid.
        """  
        return op.centroid(self.nodelist)
    
    @property
    def dim(self) -> tuple[float, float, float]:
        """
        tuple[float, float, float]: Mesh dimension.
        """
        return op.dimensions(self.nodelist)
    
    # ---- Global properties ----
    @property
    def nodelist(self) -> list[Node]:
        """
        list[Node]: List of mesh nodes.
        """
        return [n for n in self.nodelist_warp + self.nodelist_weft]
    
    @property
    def elementlist(self) -> list[Element]:
        """
        list[Element]: List of mesh elements.
        """
        return [e for e in self.elementlist_warp + self.elementlist_weft]
    
    # ---- Properties warp ----
    @property
    def nodelist_warp(self) -> list[Node]:
        """
        list[Node]: List of warp nodes.
        """
        return [n for sg in self._segmentlist_warp for n in sg.nodelist]
    
    @property
    def elementlist_warp(self) -> list[Element]:
        """
        list[Element]: List of warp elements.
        """
        return [e for sg in self._segmentlist_warp for e in sg.elementlist]
    
    # ---- Boundaries warp ----
    @property
    def nl_xmin_warp(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the x-min boundary.
        """  
        xs = np.array([n.x for n in self.nodelist_warp])
        xmin = xs.min()
        idx = np.where(np.abs(xs - xmin) < tol)[0]
        return [self.nodelist_warp[i] for i in idx]
    
    @property
    def nl_xmax_warp(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the x-max boundary.
        """  
        xs = np.array([n.x for n in self.nodelist_warp])
        xmax = xs.max()
        idx = np.where(np.abs(xs - xmax) < tol)[0]
        return [self.nodelist_warp[i] for i in idx]
    
    @property
    def nl_ymin_warp(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the y-min boundary.
        """  
        ys = np.array([n.y for n in self.nodelist_warp])
        ymin = ys.min()
        idx = np.where(np.abs(ys - ymin) < tol)[0]
        return [self.nodelist_warp[i] for i in idx]
    
    @property
    def nl_ymax_warp(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the y-max boundary.
        """  
        ys = np.array([n.y for n in self.nodelist_warp])
        ymax = ys.max()
        idx = np.where(np.abs(ys - ymax) < tol)[0]
        return [self.nodelist_warp[i] for i in idx]
    
    @property
    def nl_zmin_warp(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the z-min boundary.
        """  
        zs = np.array([n.z for n in self.nodelist_warp])
        zmin = zs.min()
        idx = np.where(np.abs(zs - zmin) < tol)[0]
        return [self.nodelist_warp[i] for i in idx]
    
    @property
    def nl_zmax_warp(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the z-max boundary.
        """  
        zs = np.array([n.z for n in self.nodelist_warp])
        zmax = zs.max()
        idx = np.where(np.abs(zs - zmax) < tol)[0]
        return [self.nodelist_warp[i] for i in idx]
    
    @property
    def el_xmin_warp(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the x-min boundary.
        """  
        xs = np.array([e.centroid.x for e in self.elementlist_warp])
        xmin = xs.min()
        idx = np.where(np.abs(xs - xmin) < tol)[0]
        return [self.elementlist_warp[i] for i in idx]
    
    @property
    def el_xmax_warp(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the x-max boundary.
        """  
        xs = np.array([e.centroid.x for e in self.elementlist_warp])
        xmax = xs.max()
        idx = np.where(np.abs(xs - xmax) < tol)[0]
        return [self.elementlist_warp[i] for i in idx]
    
    @property
    def el_ymin_warp(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the y-min boundary.
        """  
        ys = np.array([e.centroid.y for e in self.elementlist_warp])
        ymin = ys.min()
        idx = np.where(np.abs(ys - ymin) < tol)[0]
        return [self.elementlist_warp[i] for i in idx]
    
    @property
    def el_ymax_warp(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the y-max boundary.
        """  
        ys = np.array([e.centroid.y for e in self.elementlist_warp])
        ymax = ys.max()
        idx = np.where(np.abs(ys - ymax) < tol)[0]
        return [self.elementlist_warp[i] for i in idx]
    
    @property
    def el_zmin_warp(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the z-min boundary.
        """  
        zs = np.array([e.centroid.z for e in self.elementlist_warp])
        zmin = zs.min()
        idx = np.where(np.abs(zs - zmin) < tol)[0]
        return [self.elementlist_warp[i] for i in idx]
    
    @property
    def el_zmax_warp(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the z-max boundary.
        """  
        zs = np.array([e.centroid.z for e in self.elementlist_warp])
        zmax = zs.max()
        idx = np.where(np.abs(zs - zmax) < tol)[0]
        return [self.elementlist_warp[i] for i in idx]
    
    # ---- Properties weft ----
    @property
    def nodelist_weft(self) -> list[Node]:
        """
        list[Node]: List of weft nodes.
        """
        return [n for sg in self._segmentlist_weft for n in sg.nodelist]
    
    @property
    def elementlist_weft(self) -> list[Element]:
        """
        list[Element]: List of weft elements.
        """
        return [e for sg in self._segmentlist_weft for e in sg.elementlist]
    
    # ---- Boundaries weft ----
    @property
    def nl_xmin_weft(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the x-min boundary.
        """  
        xs = np.array([n.x for n in self.nodelist_weft])
        xmin = xs.min()
        idx = np.where(np.abs(xs - xmin) < tol)[0]
        return [self.nodelist_weft[i] for i in idx]
    
    @property
    def nl_xmax_weft(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the x-max boundary.
        """  
        xs = np.array([n.x for n in self.nodelist_weft])
        xmax = xs.max()
        idx = np.where(np.abs(xs - xmax) < tol)[0]
        return [self.nodelist_weft[i] for i in idx]
    
    @property
    def nl_ymin_weft(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the y-min boundary.
        """  
        ys = np.array([n.y for n in self.nodelist_weft])
        ymin = ys.min()
        idx = np.where(np.abs(ys - ymin) < tol)[0]
        return [self.nodelist_weft[i] for i in idx]
    
    @property
    def nl_ymax_weft(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the y-max boundary.
        """  
        ys = np.array([n.y for n in self.nodelist_weft])
        ymax = ys.max()
        idx = np.where(np.abs(ys - ymax) < tol)[0]
        return [self.nodelist_weft[i] for i in idx]
    
    @property
    def nl_zmin_weft(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the z-min boundary.
        """  
        zs = np.array([n.z for n in self.nodelist_weft])
        zmin = zs.min()
        idx = np.where(np.abs(zs - zmin) < tol)[0]
        return [self.nodelist_weft[i] for i in idx]
    
    @property
    def nl_zmax_weft(self, tol: float = 1e-8) -> list[Node]:
        """
        list[Node]: List of the nodes on the z-max boundary.
        """  
        zs = np.array([n.z for n in self.nodelist_weft])
        zmax = zs.max()
        idx = np.where(np.abs(zs - zmax) < tol)[0]
        return [self.nodelist_weft[i] for i in idx]
    
    @property
    def el_xmin_weft(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the x-min boundary.
        """  
        xs = np.array([e.centroid.x for e in self.elementlist_weft])
        xmin = xs.min()
        idx = np.where(np.abs(xs - xmin) < tol)[0]
        return [self.elementlist_weft[i] for i in idx]
    
    @property
    def el_xmax_weft(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the x-max boundary.
        """  
        xs = np.array([e.centroid.x for e in self.elementlist_weft])
        xmax = xs.max()
        idx = np.where(np.abs(xs - xmax) < tol)[0]
        return [self.elementlist_weft[i] for i in idx]
    
    @property
    def el_ymin_weft(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the y-min boundary.
        """  
        ys = np.array([e.centroid.y for e in self.elementlist_weft])
        ymin = ys.min()
        idx = np.where(np.abs(ys - ymin) < tol)[0]
        return [self.elementlist_weft[i] for i in idx]
    
    @property
    def el_ymax_weft(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the y-max boundary.
        """  
        ys = np.array([e.centroid.y for e in self.elementlist_weft])
        ymax = ys.max()
        idx = np.where(np.abs(ys - ymax) < tol)[0]
        return [self.elementlist_weft[i] for i in idx]
    
    @property
    def el_zmin_weft(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the z-min boundary.
        """  
        zs = np.array([e.centroid.z for e in self.elementlist_weft])
        zmin = zs.min()
        idx = np.where(np.abs(zs - zmin) < tol)[0]
        return [self.elementlist_weft[i] for i in idx]
    
    @property
    def el_zmax_weft(self, tol: float = 1e-8) -> list[Element]:
        """
        list[Element]: List of the elements on the z-max boundary.
        """  
        zs = np.array([e.centroid.z for e in self.elementlist_weft])
        zmax = zs.max()
        idx = np.where(np.abs(zs - zmax) < tol)[0]
        return [self.elementlist_weft[i] for i in idx]
    
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
            raise TypeError('Mesh frame must be a Frame object')
        f.origin = self.centroid
        self._frame = f
    
    # ---- Plot method ----
    def plot(self,
        node_warp: bool = True,
        node_warp_idx: bool = False,
        node_weft: bool = True,
        node_weft_idx: bool = False,
        element_warp: bool = True,
        element_warp_idx: bool = False,
        element_weft: bool = True,
        element_weft_idx: bool = False,
        frame_warp: bool = True,
        frame_weft: bool = True,
        legend: bool = True, 
        grid: bool = True,
        save: bool = False,
        save_dir: str = os.path.join(os.getcwd(), 'img'),
        save_fmt: str = 'png') -> None:
        """
        Plot the mesh.

        Args:
            node_warp (bool): Show warp nodes (default to `True`).
            node_warp_idx (bool): Show warp node indices (default to `False`).
            node_weft (bool): Show weft nodes (default to `True`).
            node_weft_idx (bool): Show weft node indices (default to `False`).
            element_warp (bool): Show warp elements (default to `True`).
            element_warp_idx (bool): Show warp element indices (default to `False`).
            element_weft (bool): Show weft elements (default to `True`).
            element_weft_idx (bool): Show weft element indices (default to `False`).
            frame_warp (bool): Show warp elements frames (default to `True`).
            frame_weft (bool): Show weft elements frames (default to `True`).
            legend (bool): Show legend (default to `True`).
            grid (bool): Show grid. (default to `True`)
            save (bool): Save option (default to `False`).
            save_dir (str): Save directory (default to current directory).
            save_fmt (str): File format (default to `png`).
        """        
        if save: os.makedirs(save_dir, exist_ok = True)
        filename = os.path.join(save_dir, f'mesh.{save_fmt}')
        
        nl = []
        if node_warp and self._num_warp > 0: nl.append((self.nodelist_warp, 'orange', None, node_warp_idx))
        if node_weft and self._num_weft > 0: nl.append((self.nodelist_weft, 'purple', None, node_weft_idx))
        
        el = []
        if element_warp and self._num_warp > 0: el.append((self.elementlist_warp, 'orange', element_warp_idx))
        if element_weft and self._num_weft > 0: el.append((self.elementlist_weft, 'purple', element_weft_idx))

        fl = []
        if frame_warp and self._num_warp > 0: fl.extend([e.frame for e in self.elementlist_warp])
        if frame_weft and self._num_weft > 0: fl.extend([e.frame for e in self.elementlist_weft])
        
        pt.plot_pv(nl, el, fl, grid, save, filename)
    
__all__ = ['Mesh']    
  
    
    
    
    
    
    
    
    
    
    
    




