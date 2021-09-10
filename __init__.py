# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

#  Gridbreaker
#  Mesh Generator add-on for Blender 2.93
#  (c) 2021 Andrey Sokolov (so_records)

# ----------------------------------- Add-on ----------------------------------- ADD-ON

bl_info = {
    "name": "Gridbreaker",
    "author": "Andrey Sokolov",
    "version": (1, 0, 1),
    "blender": (2, 90, 0),
    "location": "View 3D > N-Panel > Grid Break",
    "description": "Generate grids",
    "warning": "",
    "wiki_url": "https://github.com/sorecords/gridbreaker/blob/main/README.md",
    "tracker_url": "https://github.com/sorecords/gridbreaker/issues",
    "category": "Mesh"
}

'''GRID BREAKER'''

# ---------------------------------- IMPORT ------------------------------------ IMPORT

import bpy, time
from bpy.types import Context, Operator, Panel, PropertyGroup, Scene, \
    Object, BlenderRNA, UIList, Collection, Mesh, VertexGroup, Material, \
    MeshPolygon, MaterialSlot, LayerCollection, ViewLayer
from bpy.props import *
from bpy.utils import register_class, unregister_class
from bpy.app.handlers import persistent, render_init as RenderInit, \
    render_complete as RenderComplete, frame_change_post as FrameChange,\
    render_cancel as RenderCancel, depsgraph_update_post as DepsgraphUpdate,\
    load_post as LoadPost, render_pre as RenderPre
from typing import List, Tuple, Union, Dict
from mathutils import Vector
from random import randint, seed
import random
import numpy as np
from _ctypes import PyObj_FromPtr as Pointer

# ---------------------------------- Support ----------------------------------- SUPPORT

class GRDBRK_Support:
    
    @staticmethod
    def move_ob_to_col(ob : Object, col : Collection):
        for c in bpy.data.collections:
            if ob.name in c.objects:
                c.objects.unlink(ob)
        for sc in bpy.data.scenes:
            if ob.name in sc.collection.objects:
                sc.collection.objects.unlink(ob)
        col.objects.link(ob)

# ----------------------- Grid Breaker internal classes ------------------------ GRIDBREAKER

class GRDBRK_FaceCount:
    '''Calculate total cells, faces and vertices numbers'''
    
    def __init__(self, props):
        assert props.obj is not None
        self.props = props
        self.double = self._double()
        self.use_btm = self.props.btm
        self.use_top = self.props.top and (self.props.height or not self.props.btm)
        self.use_sides = self.props.sides and self.props.height
        self.limit = self._limit()
        
        self.cells_info = self._cells_info()
        self.cc = self.cells_info[0]            # cells count
        self.fc = self._fc()                    # faces count
        self.ir_btm = self._ir_btm()            # index ranges bottom
        self.ir_top = self._ir_top()            # index ranges top
        
    def _double(self) -> int:
        '''
        Return 2 if both top and bottom vertices are going to be used.
        Return 1 if not.
        '''
        return 2 if self.props.height and ((self.props.top and self.props.btm)
                                                    or self.props.sides) else 1
    
    def _limit(self) -> None:
        '''Set actual limits'''
        limit_btm = self._cuts_threshold(self.props.margin_btm) if \
                        self.use_btm or self.use_sides else None
        if not self.props.split_margins:
            return limit_btm
        limit_top = self._cuts_threshold(self.props.margin_top) if \
                        self.use_top or self.use_sides else None
        return max(limit_btm, limit_top) if limit_btm and limit_top else None

    def _cuts_threshold(self, margin : float):
        '''Subdivision level at which faces collapse because of margins'''
        threshold = 0
        for c in range(self.props.cuts):
            cut = c+1
            threshold += (.5**cut)
            if margin <= threshold:
                return cut
        return None
    
    def _cells_info(self) -> Tuple[Union[int, np.zeros]]:
        cells = self.props.cells[0]*self.props.cells[1]
        return self._cells_count(cells, self.limit)
        
    def _cells_count(self, faces : int, limit : float
                                            ) -> Tuple[Union[int, np.array]]:
        '''Cells number after subdivisions'''
        faces = int(faces)
        subdivided = int(faces)
        total_indices = 0
        cuts = limit if limit else self.props.cuts+1
        index_ranges = np.zeros([cuts,2], dtype='i')
        stopped = False
        for i in range(self.props.cuts):
            to_subdiv = int(subdivided*self.props.distribution)
            if i+1 == limit:
                index_ranges[i][0] = int(total_indices)
                index_ranges[i][1] = total_indices+subdivided-to_subdiv
                faces-=to_subdiv
                stopped = True
                break
            index_ranges[i][0] = int(total_indices)
            index_ranges[i][1] = total_indices+subdivided-to_subdiv
            total_indices+=(subdivided-to_subdiv)
            subdivided = to_subdiv*4
            faces-=to_subdiv
            faces+=subdivided
        if not stopped:
            to_subdiv = int(subdivided*self.props.distribution)
            index_ranges[self.props.cuts][0] = int(total_indices)
            index_ranges[self.props.cuts][1] = total_indices+subdivided
        return faces, index_ranges
    
    def _fc(self) -> int:
        '''Total faces number'''
        btm = self.cc if self.use_btm and self.props.btm else 0
        top = self.cc if self.use_top and self.props.top else 0
        sides = self.cc*4 if self.use_sides and self.props.sides else 0
        return btm + top + sides
    
    def _ir_btm(self) -> np.ndarray:
        '''Bottom faces indicies ranges for each subdivision level'''
        return self.cells_info[1].copy() if self.use_btm or self.use_sides \
                                                                    else None
    
    def _ir_top(self) -> np.ndarray:
        '''Top faces indicies ranges for each subdivision level'''
        if self.use_top or self.use_sides:
            start_index = self.cc if self.use_btm or self.use_sides else 0
            return self.cells_info[1].copy() + start_index
        else:
            return None

class GRDBRK_FaceData:
    '''Get Pydata (Vertices, Faces data) for Grid Breaker Mesh'''
    
    def __init__(self, props, sc : Scene, vl : ViewLayer):
        self.props = props
        self.vl = vl
        self.sc = sc
        self.fc = GRDBRK_FaceCount(props)
        self.cc_double = self._cc_double()
        
        # first calculate top and bottom faces
        self.faces_centers = self._face_centers()
        self.set_heights()
        
        # set their sizes taking margins into account
        self.margins = self._margins()
        self.faces_radiuses = self._faces_radiuses()
        
        self.x_radius = None
        self.y_radius = None
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.y_end = None
        self.seed = int(self.props.seed)
        self.set_bounds()
        self.set_faces_centers()
        self.vertices = self._vertices()
        
        self.faces = self._faces()
        
        self.materials_store()
        self.store_materials()
        self.materials = self._materials()
        
    def _cc_double(self) -> int:
        '''Number of cells - doubled or not'''
        return self.fc.cc*self.fc.double
        
    def _face_centers(self) -> np.zeros:
        '''Numpy zeros array for Faces Centers coordinates'''
        return np.zeros([self.cc_double , 3], dtype='f')
        
    def set_heights(self) -> None:
        '''
        Calculate top and bottom faces' heights taking into account
        z-position, height, random height and random height center offset 
        '''
        ps = self.props
        cc = self.fc.cc
        
        positions = np.full(self.cc_double, ps.position[2])
        
        if self.fc.double==2 or (self.props.height and (self.fc.use_top 
                                                        or self.fc.use_btm)):
            center = 1-ps.height_center if ps.height_invert else ps.height_center
            
            centers = np.full(self.cc_double, center)
            heights = np.full(self.cc_double, ps.height)
            randoms = np.zeros(self.cc_double)
            randoms[:cc] = self._heights_randoms()   # random amount
            if self.fc.double==2:
                randoms[cc:]=randoms[:cc]            
            heights_randomed = heights-heights*randoms            
            if ps.height_invert:
                heights_randomed = heights-heights_randomed
            if self.fc.double==2:
                remainder_left = heights_randomed[:cc] * centers[:cc]
                remainder_right = heights_randomed[cc:] * (1-centers[:cc])
            
                pos_btm = positions[:cc] + centers[:cc]*heights[:cc] - remainder_left
                pos_top = positions[cc:] + centers[cc:]*heights[cc:] + remainder_right
                
                self.faces_centers[:cc, 2] = pos_btm
                self.faces_centers[cc:, 2] = pos_top
            elif self.fc.use_top:
                remainder = heights_randomed[:] * (1-centers[:])
                pos_top = positions[:] + centers[:]*heights[:] + remainder
                self.faces_centers[:, 2] = pos_top
            elif self.fc.use_btm:
                remainder = heights_randomed[:cc] * centers[:cc]
                pos_btm = positions[:cc] + centers[:cc]*heights[:cc] - remainder                
                self.faces_centers[:, 2] = pos_btm
            else:
                raise ValueError('Setting height is impossible')
        else:
            self.faces_centers[:, 2] = positions
        
    def _heights_randoms(self) -> np.array:
        '''np.arrays filled with cells' random height values'''
        assert self.props.height != 0
        np.random.seed(self.props.height_seed)
        randoms = np.random.rand(self.fc.cc)   # random numbers
        heights = np.full(self.fc.cc, self.props.height_random)
        return randoms * heights
    
    def _margins(self) -> np.ndarray:
        '''Return np.ndarray with faces' margins in absolute metrics'''
        margins = np.zeros([self.cc_double, 2], dtype='f')
        cc = self.fc.cc
        margin_btm = self._margin(self.props.margin_btm)
        if self.fc.use_top or self.fc.use_sides:
            margin_top = self._margin(self.props.margin_top) if \
                            self.props.split_margins else margin_btm
        if self.fc.double==2:
            margins[:cc, 0].fill(margin_btm[0])
            margins[:cc, 1].fill(margin_btm[1])
            margins[cc:, 0].fill(margin_top[0])
            margins[cc:, 1].fill(margin_top[1])
        elif self.fc.use_top:
            margins[:cc, 0].fill(margin_top[0])
            margins[:cc, 1].fill(margin_top[1])
        else:
            margins[:cc, 0].fill(margin_btm[0])
            margins[:cc, 1].fill(margin_btm[1])
        return margins
    
    def _margin(self, margin : float) -> Tuple[float]:
        '''Calculate absolute margin metrics'''
        x_size = self.props.size[0] / self.props.cells[0]
        y_size = self.props.size[1] / self.props.cells[1]
        x = x_size - x_size * margin
        y = y_size - y_size * margin
        return (x, y)
    
    def _faces_radiuses(self) -> np.ndarray:
        '''Return nd.array with faces' radiuses in absolute metrics'''
        fr = np.zeros([self.cc_double , 2], dtype='f')
        x_size = self.props.size[0]/self.props.cells[0]
        y_size = self.props.size[1]/self.props.cells[1]
        
        if self.fc.use_sides or (self.fc.use_top and self.fc.use_btm):
            for ir_b, ir_t in zip(self.fc.ir_btm, self.fc.ir_top):
                x_size /= 2
                y_size /= 2
                if ir_b[0]==ir_b[1]:
                    continue
                x_btm = x_size - self.margins[ir_b[0]:ir_b[1], 0] / 2
                y_btm = y_size - self.margins[ir_b[0]:ir_b[1], 1] / 2
                x_top = x_size - self.margins[ir_t[0]:ir_t[1], 0] / 2
                y_top = y_size - self.margins[ir_t[0]:ir_t[1], 1] / 2
                if x_btm[0] < 0:
                    if x_top[0] < 0:
                        break
                    else:
                        fr[ir_t[0]:ir_t[1], 0] = x_top
                        fr[ir_t[0]:ir_t[1], 1] = y_top
                elif x_top[0] < 0:
                    if x_btm[0] < 0:
                        break
                    else:
                        fr[ir_b[0]:ir_b[1], 0] = x_btm
                        fr[ir_b[0]:ir_b[1], 1] = y_btm
                else:
                    fr[ir_b[0]:ir_b[1], 0] = x_btm
                    fr[ir_b[0]:ir_b[1], 1] = y_btm
                    fr[ir_t[0]:ir_t[1], 0] = x_top
                    fr[ir_t[0]:ir_t[1], 1] = y_top
        else:
            irs = self.fc.ir_top if self.fc.use_top else self.fc.ir_btm
            for ir in irs:
                x_size /= 2
                y_size /= 2
                if ir[0]==ir[1]:
                    continue
                x_top = x_size - self.margins[ir[0]:ir[1], 0] / 2
                y_top = y_size - self.margins[ir[0]:ir[1], 1] / 2
                if x_top[0] < 0:
                    break
                else:
                    fr[ir[0]:ir[1], 0] = x_top
                    fr[ir[0]:ir[1], 1] = y_top
        return fr
        
    def set_bounds(self) -> None:
        '''Set Gridbreaker mesh's start & end coordinates on x & y axises'''
        self.x_radius = self.props.size[0] / 2
        self.y_radius = self.props.size[1] / 2
        self.x_start = self.props.position[0] - self.x_radius
        self.x_end = self.props.position[0] + self.x_radius
        self.y_start = self.props.position[1] - self.y_radius
        self.y_end = self.props.position[1] + self.y_radius
        
    def set_faces_centers(self) -> None:
        '''Fill face_centers np.ndarray with absolute x & y coordinates'''
        ir = self.fc.ir_top if self.fc.use_top and self.fc.double==1 else \
                                                            self.fc.ir_btm
        cells_centers = self._cells_centers()
        size_x = self.props.size[0]/self.props.cells[0]/2
        size_y = self.props.size[1]/self.props.cells[1]/2
        cell_size = np.array([size_x, size_y])
        self._subdivide(ir, cells_centers, cell_size, 1)
        if self.fc.double==2:
            cc = self.fc.cc
            self.faces_centers[cc:,0:2] = self.faces_centers[:cc,0:2]
    
    def _cells_centers(self) -> np.ndarray:
        '''Return np.ndarray with the main cells' centers X & Y coordinates'''
        line_x = self._line(0, self.x_start, self.x_end)
        line_y = self._line(1, self.y_start, self.y_end)
        cells_x = self.props.cells[0]
        cells_y = self.props.cells[1]
        cells_faces = np.zeros([cells_x * cells_y, 2])
        num = 0
        for x in range(cells_x):
            for y in range(cells_y):
                cells_faces[num] = (line_x[x],line_y[y])
                num+=1
        return cells_faces
        
    def _line(self, index : int, start : float, end : float) -> np.array:
        '''Return 1D numpy array with main cells's centers X & Y coordinates'''
        cell_radius = self.props.size[index]/self.props.cells[index]/2
        start += cell_radius
        step = cell_radius * 2
        return np.arange(start, end, step)
    
    def _subdivide(self, ir : np.ndarray, cells : np.ndarray,
                            cell_size : np.array, cuts : int) -> None:
        '''
        Recursively subdivide filtered faces
        Fill self.face_centers np.ndarray with the faces' X & Y coordinates
        '''
        all_cuts = self.fc.limit if self.fc.limit else self.props.cuts
        if cuts <= all_cuts:
            filtered = self._filter_faces(cells)
            to_subdiv = filtered[0]
            to_append = filtered[1]
            self.faces_centers[ir[cuts-1][0]:ir[cuts-1][1], 0:2] = to_append
            cell_size /= 2
            subdivided = self._subdiv(to_subdiv, cell_size)
            self._subdivide(ir, subdivided, cell_size, cuts+1)
        elif not self.fc.limit:
            self.faces_centers[ir[cuts-1][0]:ir[cuts-1][1], 0:2] = cells
                                            
    def _filter_faces(self, faces : np.ndarray) -> Tuple[np.ndarray]:
        '''
        Filter faces for subdividing
        Return tuple with Filtered and Remaining faces centers np.ndarrays
        '''
        faces_total = len(faces)
        faces_num = int(faces_total*self.props.distribution)
        faces_indices = list(range(faces_total))
        result_indices = set()
        while faces_num:
            faces_num-=1
            faces_total-=1
            self.seed+=2
            random.seed(self.seed)
            r = random.randint(0, faces_total)
            assert faces_indices[r] not in result_indices
            result_indices.add(faces_indices.pop(r))
        all_indices = set(faces_indices)
        remaining_indices = all_indices-result_indices
        filtered = np.zeros([len(result_indices),2])
        remaining = np.zeros([len(remaining_indices),2])
        for i, ind in enumerate(result_indices):
            filtered[i] = faces[ind]
        for i, ind in enumerate(remaining_indices):
            remaining[i] = faces[ind]
        return filtered, remaining
            
        
    def _subdiv(self, to_subdiv : np.ndarray, cell_size : np.array
                                                        ) -> np.ndarray:
        '''Subdivide faces and set new faces centers (for recursion)'''
        faces_count = len(to_subdiv)
        
        shape = np.zeros([faces_count*4, 2])
        subdivided = shape.copy()
        subdivided[0::4] = to_subdiv
        subdivided[1::4] = to_subdiv
        subdivided[2::4] = to_subdiv
        subdivided[3::4] = to_subdiv
        
        sizes = shape.copy()
        sizes[:,0].fill(cell_size[0])
        sizes[:,1].fill(cell_size[1])
        
        vectors = shape.copy()
        vectors[0::4,0].fill(-1)
        vectors[0::4,1].fill(-1)
        vectors[1::4,0].fill(-1)
        vectors[1::4,1].fill(1)
        vectors[2::4,0].fill(1)
        vectors[2::4,1].fill(1)
        vectors[3::4,0].fill(1)
        vectors[3::4,1].fill(-1)
        
        return subdivided+vectors*sizes
            
    def _vertices(self) -> np.ndarray:
        '''Calculate vertices coordinates and return as np.ndarray'''
        fc = self.faces_centers
        fr = self.faces_radiuses
        verts = np.zeros([self.cc_double*4 , 3])
        
        faces_centers = verts.copy()
        faces_centers[0::4] = fc
        faces_centers[1::4] = fc
        faces_centers[2::4] = fc
        faces_centers[3::4] = fc
        
        faces_radiuses = verts.copy()
        faces_radiuses[0::4, 0:2] = fr
        faces_radiuses[1::4, 0:2] = fr
        faces_radiuses[2::4, 0:2] = fr
        faces_radiuses[3::4, 0:2] = fr  
              
        vectors = verts.copy()
        vectors[0::4,0].fill(-1)
        vectors[0::4,1].fill(-1)
        vectors[1::4,0].fill(-1)
        vectors[1::4,1].fill(1)
        vectors[2::4,0].fill(1)
        vectors[2::4,1].fill(1)
        vectors[3::4,0].fill(1)
        vectors[3::4,1].fill(-1)
        
        return faces_centers+faces_radiuses*vectors
        
    def _faces(self):
        ps = self.props
        cc = self.fc.cc   
        fc = self.fc.fc
        faces = np.zeros([fc, 4], dtype='i') 
        if self.fc.use_sides:
            vc = cc*8
            if self.fc.use_top and self.fc.use_btm:
                sd_start = cc*2
                verts_inds = np.arange(0, vc, 1)
                faces[:cc,0] = verts_inds[0:cc*4:4]
                faces[:cc,1] = verts_inds[1:cc*4:4]
                faces[:cc,2] = verts_inds[2:cc*4:4]
                faces[:cc,3] = verts_inds[3:cc*4:4]
                faces[cc:cc*2, 0] = verts_inds[cc*4+3:cc*8:4]
                faces[cc:cc*2, 1] = verts_inds[cc*4+2:cc*8:4]
                faces[cc:cc*2, 2] = verts_inds[cc*4+1:cc*8:4]
                faces[cc:cc*2, 3] = verts_inds[cc*4+0:cc*8:4]
            elif self.fc.use_btm:
                sd_start = cc
                verts_inds = np.arange(0, vc, 1)
                faces[:cc,0] = verts_inds[0:cc*4:4]
                faces[:cc,1] = verts_inds[1:cc*4:4]
                faces[:cc,2] = verts_inds[2:cc*4:4]
                faces[:cc,3] = verts_inds[3:cc*4:4]
            elif self.fc.use_top:
                sd_start = cc
                verts_inds = np.arange(0, vc, 1)
                faces[:cc,0] = verts_inds[cc*4+3:cc*8:4]
                faces[:cc,1] = verts_inds[cc*4+2:cc*8:4]
                faces[:cc,2] = verts_inds[cc*4+1:cc*8:4]
                faces[:cc,3] = verts_inds[cc*4+0:cc*8:4]
            else:
                sd_start = 0
                verts_inds = np.arange(0, vc, 1)
            vs = cc*4
            faces[sd_start+0::4,0] = verts_inds[vs::4]
            faces[sd_start+0::4,1] = verts_inds[vs+1::4]
            faces[sd_start+0::4,2] = verts_inds[1:vs:4]
            faces[sd_start+0::4,3] = verts_inds[:vs:4]
            faces[sd_start+1::4,0] = verts_inds[vs+1::4]
            faces[sd_start+1::4,1] = verts_inds[vs+2::4]
            faces[sd_start+1::4,2] = verts_inds[2:vs:4]
            faces[sd_start+1::4,3] = verts_inds[1:vs:4]
            faces[sd_start+2::4,0] = verts_inds[vs+2::4]
            faces[sd_start+2::4,1] = verts_inds[vs+3::4]
            faces[sd_start+2::4,2] = verts_inds[3:vs:4]
            faces[sd_start+2::4,3] = verts_inds[2:vs:4]
            faces[sd_start+3::4,0] = verts_inds[vs+3::4]
            faces[sd_start+3::4,1] = verts_inds[vs::4]
            faces[sd_start+3::4,2] = verts_inds[:vs:4]
            faces[sd_start+3::4,3] = verts_inds[3:vs:4]                
        else:
            vc = fc*4
            verts_inds = np.arange(0, vc, 1)
            faces[:,0] = verts_inds[3::4]
            faces[:,1] = verts_inds[2::4]
            faces[:,2] = verts_inds[1::4]
            faces[:,3] = verts_inds[0::4]
        return [tuple(f) for f in faces]
        
    def materials_store(self) -> None:
        ps = self.props
        if (ps.materials_split or ps.mat_use_list) and not ps.mat_used:
            self._mat_store()
            ps.mat_used = True
    
    def store_materials(self) -> None:
        ps = self.props
        if ps.materials_split or ps.mat_use_list:
            return
        if ps.mat_used:
            ps.mat_used = False
            return
        self._mat_store()
            
    def _mat_store(self):
        ps = self.props
        sc = bpy.context.scene
        ps.mat_real.clear()
        materials = set([ms.material for ms in ps.obj.material_slots])
        if not materials:
            return
        for _ in range(len(materials)):
            prop = f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_real"
            active = f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_real_active"
            GRDBRK_SlotAdd(self.sc, self.vl, prop, active)
        for mr, m in zip(ps.mat_real, materials):
            mr.material = m
    
    def _materials(self):
        ps = self.props
        slots = len(ps.obj.material_slots)
        fc = self.fc.fc
        cc = self.fc.cc
        mats = np.zeros(fc, dtype=int)
        mats_btm = np.zeros(cc, dtype=int)
        mats_top = np.zeros(cc, dtype=int)
        mats_sides = np.zeros(cc*4, dtype=int)
        if ps.materials_split:
            if self.fc.use_btm:
                if len(ps.mat_btm):
                    mat_inds = self._mat_inds([m.material for m in ps.mat_btm])
                    np.random.seed(ps.mat_btm_seed)
                    mats_btm = np.random.choice(mat_inds, cc) if mat_inds else \
                                                        np.zeros(cc, dtype=int)
                    mats[:cc] = mats_btm                    
            if self.fc.use_top:
                if len(ps.mat_top):
                    mat_inds = self._mat_inds([m.material for m in ps.mat_top])
                    np.random.seed(ps.mat_top_seed)
                    mats_top = np.random.choice(mat_inds, cc) if mat_inds else \
                                                        np.zeros(cc, dtype=int)
                    if ps.btm:
                        mats[cc:cc*2] = mats_top
                    else:
                        mats[:cc] = mats_top
            if self.fc.use_sides:
                if len(ps.mat_sides):
                    mat_inds = self._mat_inds([m.material for m in
                                                        ps.mat_sides])
                    np.random.seed(ps.mat_sides_seed)
                    mats_sides = np.random.choice(mat_inds, cc) if mat_inds \
                                                    else np.zeros(cc, dtype=int)
                    if ps.btm:
                        if ps.top:
                            mats[cc*2+0::4] = mats_sides
                            mats[cc*2+1::4] = mats_sides
                            mats[cc*2+2::4] = mats_sides
                            mats[cc*2+3::4] = mats_sides
                        else:
                            mats[cc+0::4] = mats_sides
                            mats[cc+1::4] = mats_sides
                            mats[cc+2::4] = mats_sides
                            mats[cc+3::4] = mats_sides
                    elif ps.top:
                        mats[cc+0::4] = mats_sides
                        mats[cc+1::4] = mats_sides
                        mats[cc+2::4] = mats_sides
                        mats[cc+3::4] = mats_sides                        
                    else:
                        mats[0::4] = mats_sides
                        mats[1::4] = mats_sides
                        mats[2::4] = mats_sides
                        mats[3::4] = mats_sides
                    
        elif ps.materials_random:
            np.random.seed(ps.mat_btm_seed)
            if ps.mat_use_list:
                mat_inds = self._mat_inds([m.material for m in ps.mat_main])
                randoms = np.random.choice(mat_inds, fc) if mat_inds else \
                                                        np.zeros(fc, dtype=int)
            elif len(ps.mat_real):
                mat_inds = self._mat_inds([m.material for m in ps.mat_real])
                randoms = np.random.choice(mat_inds, fc) if mat_inds else \
                                                        np.zeros(fc, dtype=int)
            else:
                randoms = np.random.randint(0, high=slots, size=fc, dtype=int) \
                                         if slots else np.zeros(fc, dtype=int)
            mats[:] = randoms
        else:
            np.random.seed(ps.mat_btm_seed)
            if ps.mat_use_list:
                mat_inds = self._mat_inds([m.material for m in ps.mat_main])
                randoms = np.random.choice(mat_inds, cc) if mat_inds else \
                                                        np.zeros(cc, dtype=int)
            elif len(ps.mat_real):
                mat_inds = self._mat_inds([m.material for m in ps.mat_real])
                randoms = np.random.choice(mat_inds, cc) if mat_inds else \
                                                        np.zeros(cc, dtype=int)
            else:
                randoms = np.random.randint(0, high=slots, size=cc, dtype=int) \
                                         if slots else np.zeros(cc, dtype=int)
            if ps.sides:
                if ps.btm:
                    mats[:cc] = randoms
                    if ps.top:
                        if ps.height:
                            mats[cc:cc*2] = randoms
                            mats[cc*2+0::4] = mats[cc*2+1::4] = mats[cc*2+2::4]\
                                                    = mats[cc*2+3::4] = randoms
                    elif ps.height:
                        mats[cc+0::4] = mats[cc+1::4] = mats[cc+2::4] = \
                                                        mats[cc+3::4] = randoms
                elif ps.top:
                    mats[:cc] = randoms
                    if ps.height:
                        mats[cc+0::4] = mats[cc+1::4] = mats[cc+2::4] = \
                                                        mats[cc+3::4] = randoms
                else:
                    mats[0::4] = mats[1::4] = mats[2::4] = mats[3::4] = randoms
                    
            elif ps.btm:
                mats[:cc] = randoms
                if ps.top and ps.height:
                    mats[cc:] = randoms
            else:
                mats[:cc] = randoms
        return mats
    
    def _mat_inds(self, mat_list : List[Material]) -> List[int]:
        ob = self.props.obj
        obj_mats = ob.data.materials
        none_ms = [m for m in ob.material_slots if not m.material]
        part_mats = []
        for m in mat_list:
            if not m:
                if not none_ms:
                    none_ms = self._new_slot()
                    part_mats.append(len(ob.material_slots)-1)
                continue
            if m.name not in obj_mats:
                ms = self._new_slot()
                ms.material = m
                part_mats.append(len(ob.material_slots)-1)
        obj_mats = self.props.obj.data.materials
        part_mats = []
        for i, m in enumerate(obj_mats):
            if m in mat_list:
                part_mats.append(i)
        return part_mats
    
    def _new_slot(self) -> MaterialSlot:
        self.props.obj.active_material_index = len(self.props.obj.material_slots)-1
        bpy.ops.object.material_slot_add()
        return self.props.obj.material_slots[len(self.props.obj.material_slots)-1]
    
# --------------------------------- New Base Mesh ------------------------------ GRID BREAKER

class GRDBRK_Gridbreaker:
            
    def __init__(self, sc : Scene, vl : ViewLayer):
        self.sc = sc
        self.vl = vl
        self.cancelled = False
        self.props = self.sc.gridbreaker[self.sc.grdbrk_active]
        if not (self.props.obj and self.props.obj.name in self.sc.objects):
            GRDBRK_New(sc, vl)
            return #because GRDBRK_New will launch GRDBRK_Gridbreaker once again
        self.obj = self.props.obj
        self.set_active()
        self.mesh = self.obj.data
        self.mesh.clear_geometry()
        self.props = self.obj.grdbrk
        if all((not self.props.top, not self.props.sides, not self.props.btm)):
            self.cancelled = True
            return
        self.pydata = GRDBRK_FaceData(self.props, self.sc, self.vl)
        self.mesh_set()
        self.mesh_data_set()
        self.set_collection()
        GRDBRK_ClearMaterials(self.sc, self.vl)
    
    def set_active(self):
        '''Set active GB object as View Layer's active object'''
        if self.obj.name in self.vl.objects:
            self.vl.objects.active = self.obj
            for ob in self.sc.objects:
                ob.select_set(False)
            self.obj.select_set(True)
    
    def mesh_set(self) -> None:
        self.mesh.from_pydata(
                vertices=self.pydata.vertices,
                edges=[],
                faces=self.pydata.faces )        
        
    def mesh_data_set(self):
        faces = self.obj.data.polygons
        fc = len(faces)
        smooth = np.ones(fc, dtype = bool) if self.props.shade_smooth \
                                    else np.zeros(fc, dtype = bool)
        faces.foreach_set('material_index', self.pydata.materials)
        faces.foreach_set('use_smooth', smooth)        
                
    def merge(self):
        '''
        NOT IMPLEMENTED. UNSTABLE.
        Merge is a good idea but it needs manual algorithm
        '''
        mod_name='GRDBRK_Merge'
        mod = self.obj.modifiers.new(name=mod_name, type='WELD')
        mod_name = mod.name
        while not self.obj.modifiers[0] == mod:
            bpy.ops.object.modifier_move_up(modifier=mod_name)
        bpy.ops.object.modifier_apply(modifier=mod_name)
    
    def set_collection(self):
        col = self.sc.grdbrk.collection if self.sc.grdbrk.collection_use \
                                        else self.props.collection
        if col is not None and self.obj.name not in col.objects:
            GRDBRK_Support.move_ob_to_col(self.obj, col)
    
# --------------------------------- New Base Mesh ------------------------------ NEW BASE MESH

class GRDBRK_New:
    
    def __init__(self, sc : Scene, vl : ViewLayer):
        self.sc = sc
        self.vl = vl
        self.props = self.sc.gridbreaker[self.sc.grdbrk_active]
        self.ob_name = self._ob_name()
        self.mesh = self._mesh()
        self.obj = self._obj()
        self.obj_set()
        self.col = self._col()
        self.col_set()
        GRDBRK_Gridbreaker(sc, vl)
    
    def _ob_name(self) -> str:
        basename = "Grid Breaker"
        obname = str(basename)
        num = 0
        while obname in bpy.data.objects:
            num += 1
            obname = f'{basename} {num:02d}'
        return obname
    
    def _mesh(self) -> Mesh:
        return bpy.data.meshes.new(self.ob_name)
    
    def _obj(self) -> Object:
        return bpy.data.objects.new(self.ob_name, self.mesh)
    
    def _col(self) -> Collection:
        if self.props.obj.grdbrk.collection:
            return self.props.obj.grdbrk.collection
        elif self.vl.active_layer_collection is not None:
            try:
                col = bpy.data.collections[self.vl.active_layer_collection.name]
            except KeyError:
                col = self.sc.collection
            return col
        else:
            return self.sc.collection
        
    def obj_set(self) -> None:
        self.props.obj = self.props.obj.grdbrk.obj = self.obj
        self.props.obj.data.use_auto_smooth=True
        
    def col_set(self) -> None:
        self.col.objects.link(self.obj)

#  ------------------------------ Slot Add/Remove  ----------------------------- SLOT ADD/REMOVE

class PropPathParse():
    '''
    To be used in SlotAdd/SlotRemove Operators
    with multiple CollectionProperty() hierarchies
    '''
    
    def prop_list(self, prop : str) -> List[str]:
        '''Convert prop string into list of proper props strings'''
        splitted = prop.split('.')
        result = []
        short_name=""
        append_to = 'main'
        for s in splitted:
            if "'[" in s or '"[' in s:
                short_name+=s
                append_to = 'short'
            elif append_to == 'short':
                short_name+=s
                if "]'" in s or ']"' in s:
                    append_to='main'
                    result.append(short_name)
                    short_name=""
            elif "[" in s and "]" in s:
                split_index = s.split('[')
                result.append(split_index[0])
                result.append('...'+split_index[-1][:-1])
            else:
                result.append(s)
        return result                
    
    def get_prop(self, source : Union[Object, Scene], prop : str) -> BlenderRNA:
        prop_list = self.prop_list(prop)
        for p in prop_list:
            if p.startswith('...'):
                source = source[int(p[3:])]
            else:
                source = getattr(source, p)
        return source
    
    def set_prop(self, source : Union[Object, Scene], prop : str,
                    value : Union[int, float, Vector]) -> None:
        prop_list = self.prop_list(prop)
        if len(prop_list) == 0:
            return
        elif len(prop_list) == 1:
            setattr(source, prop, value)
        else:
            num = 0
            while num < len(prop_list)-1:
                if prop_list[num].startswith('...'):
                    source = source[int(prop_list[num][3:])]
                else:
                    source = getattr(source, prop_list[num])
                num += 1
            setattr(source, prop_list[-1], value)
            return source

class GRDBRK_SlotAdd(PropPathParse):
    
    def __init__(self, sc : Scene, vl : ViewLayer, prop: str, active : str):
        props = self.get_prop(sc, prop)
        props.add()
        index = len(props)-1
        self.set_prop(sc, active, index)
        props[index].index = index
        if prop == 'gridbreaker':
            GRDBRK_New(sc, vl)
        sc.update_tag()
    
class GRDBRK_SlotRemove(PropPathParse):
    
    def __init__(self, sc : Scene, prop: str, active : str):
        index = self.get_prop(sc, active)
        props = self.get_prop(sc, prop)
        if prop == 'gridbreaker'and props[index].obj:
            bpy.data.objects.remove(props[index].obj)
        props.remove(index)
        for i, ar in enumerate(props):
            ar.index = i
        new_index = index-1 if index else 0
        self.set_prop(sc, active, new_index)
        sc.update_tag()

# ------------------------------- Materials Clear ----------------------------- MATERIALS CLEAR
    
class GRDBRK_ClearMaterials:
    
    def __init__(self, sc : Scene, vl : ViewLayer):
        self.sc = sc
        self.vl = vl
        self.props = self.sc.gridbreaker[self.sc.grdbrk_active]
        self.ob = self.props.obj
        if not self.obj_in_vl():
            wtype='ERROR'
            msg=f'{self.ob.name} object is not in the active View Layer \
({self.vl.name})\nSome properties may be set unproperly'
            bpy.ops.grdbrk.warning('INVOKE_DEFAULT', type=wtype, msg=msg)
            return
        self.ensure_active()
        self.props = self.props.obj.grdbrk
        self.used_materials = self._used_materials()
        self.remove_unused()
        
    def obj_in_vl(self) -> bool:
        '''Check if object is in view_layer'''
        return self.ob.name in self.vl.objects
    
    def ensure_active(self) -> None:
        if not self.vl.objects.active == self.ob:
            self.vl.objects.active = self.ob                
    
    def _used_materials(self) -> List[Material]:
        ob = self.props.obj
        materials = set()
        if self.props.materials_split:
            if self.props.btm:
                for m in self.props.mat_btm:
                    materials.add(m.material)
            if self.props.top:
                for m in self.props.mat_top:
                    materials.add(m.material)
            if self.props.sides:
                for m in self.props.mat_sides:
                    materials.add(m.material)
        elif self.props.mat_use_list:
            materials = set([m.material for m in self.props.mat_main])
        else:
            materials = set([m.material for m in self.props.mat_real])
        return materials
    
    def remove_unused(self):
        unused_slots_inds = []
        for i, ms in enumerate(self.ob.material_slots):
            if ms.material not in self.used_materials:
                unused_slots_inds.append(i)
        for i in reversed(unused_slots_inds):
            self.ob.active_material_index = i
            bpy.ops.object.material_slot_remove()

# ------------------------------ Fix Missing Object ---------------------------- OBJ FIX MISSING

class GRDBRK_OT_FixMissingObj(Operator):
    bl_label = "Fix missing object"
    bl_idname = "grdbrk.fix_missing"
    bl_options = {'UNDO'}
    
    def execute(self, context):
        self.sc = context.scene
        self.vl = context.view_layer
        wtype='ERROR'
        msg="Deleting Grid Breaker objects manually may lead to unstable work \
and crashes.\nPlease, use the minus '-' button in the add-on's settings \
instead.\n"
        bpy.ops.grdbrk.warning('INVOKE_DEFAULT',type=wtype, msg=msg)
        sc = context.scene
        missing_indices = [gb.index for gb in sc.gridbreaker if gb.obj is None
                                            or gb.obj.name not in sc.objects]
        for i in reversed(missing_indices):
            while sc.grdbrk_active != i:
                sc.grdbrk_active = i
                sc.update_tag()
            GRDBRK_SlotRemove(self.sc, "gridbreaker", "grdbrk_active")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return self.execute(context)
    
class GRDBRK_OT_Apply(Operator):
    bl_label = "Apply Geometry"
    bl_idname = "grdbrk.apply"
    bl_description = "Unlink geometry from add-on, Remove active slot"
    bl_options = {"UNDO"}
    
    def duplicate(self) -> Object:
        mesh = self.ob.data.copy()
        mesh.name = self.ob_name
        ob = self.ob.copy()
        ob.name = self.ob_name
        ob.data = mesh
        for c in bpy.data.collections:
            if self.ob.name in c.objects:
                c.objects.link(ob)
        for sc in bpy.data.scenes:
            if self.ob.name in sc.collection.objects:
                sc.collection.objects.link(ob)
        return ob
    
    def execute(self, context) -> set:
        self.sc = context.scene
        self.vl = context.view_layer
        self.props = self.sc.gridbreaker[self.sc.grdbrk_active]
        self.ob = self.props.obj
        self.ob_name = str(self.ob.name)
        ob = self.duplicate()
        GRDBRK_SlotRemove(self.sc, 'gridbreaker', 'grdbrk_active')
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = ob
        ob.select_set(True)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return self.execute(context)

class GRDBRK_OT_Warning(Operator):
    bl_idname = "grdbrk.warning"
    bl_label = "Warning!"
    type: StringProperty(default="ERROR")
    msg : StringProperty(default="")
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        return {'FINISHED'}
    
    def modal(self, context, event):
        if event:
            self.report({self.type}, self.msg)
        return {'FINISHED'}
        
    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

#  -------------------------- Duplicate Grid Breaker  -------------------------- OBJ DUPLICATE

class GRDBRK_OT_Duplicate(Operator):
    bl_label = "Duplicate"
    bl_idname = "grdbrk.duplicate"
    bl_description = "Duplicate Active Grid Breaker"
    bl_options = {"UNDO"}
    
    def _ob_name(self) -> str:
        obname = str(self.ob.name)
        if (len(obname)>8 and obname[-8:-3] == '.copy') or \
        (obname.endswith('.copy')):
            num = 1
            newname = self._get_name(obname, num)
        else:
            num = 0
            newname=f'{obname}.copy'
            if newname in bpy.data.objects:
                obname = str(newname)
        while newname in bpy.data.objects:
            num += 1
            newname = self._get_name(obname, num)
                
        return newname
    
    def _get_name(self, obname : str, num : int) -> str:
        return f'{obname[:-2]}{num:02d}' if \
                all([c.isdigit() for c in obname[-2:]]) else \
                f'{obname} {num:02d}'   
                
    def copy_obj(self):
        self.sc.grdbrk.live_update = False
        ob = self.props.obj.copy()
        for c in bpy.data.collections:
            if self.props.obj.name in c.objects:
                c.objects.link(ob)
        for sc in bpy.data.scenes:
            if self.props.obj.name in sc.collection.objects:
                sc.collection.objects.link(ob)
        bpy.data.objects.remove(self.new_props.obj)
        self.new_props.obj = ob
        self.new_props.obj.data = self.props.obj.data.copy()
        self.new_props.obj.name = self.new_props.obj.data.name = self.ob_name
        if ob.animation_data and ob.animation_data.action:
            self.new_props.obj.animation_data.action = ob.animation_data.action.copy()
        
    def _add_slots(self, prop : str, active : str, slots : int):
        for _ in range(slots):
            n_prop = f"gridbreaker[{self.sc.grdbrk_active}].{prop}"
            n_active = f"gridbreaker[{self.sc.grdbrk_active}].{active}"
            GRDBRK_SlotAdd(self.sc, self.vl,n_prop, n_active)               
    
    def col_copy(self, old : CollectionProperty, new : CollectionProperty, 
                prop : str, active : str) -> None:
        self._add_slots(prop, active, len(old))
        for o, n in zip(old, new):
            for p in dir(o):
                if p.startswith(self.excepts):
                    continue
                setattr(n, p, getattr(o,p))
    
    def duplicate(self) -> None:
        for p in dir(self.props):
            if p.startswith(self.excepts):
                continue
            try:
                setattr(self.new_props, p, getattr(self.props, p))
            except AttributeError:
                old_c_prop = getattr(self.props, p)
                new_c_prop = getattr(self.new_props, p)
                active = [pr for pr in dir(self.props) if pr.startswith(p)
                            and pr.endswith('active')][0]
                self.col_copy(old_c_prop, new_c_prop, p, active)
        self.sc.grdbrk.live_update = True
        self.props.mat_used = True
        GRDBRK_Gridbreaker(self.sc, self.vl)
    
    def execute(self, context) -> set:
        self.sc = context.scene
        self.vl = context.view_layer
        self.excepts = ('__', 'bl_', 'index', 'rna', 'name', 'obj')
        self.props = self.sc.gridbreaker[int(self.sc.grdbrk_active)]
        self.ob = self.props.obj
        self.ob_name = self._ob_name()
        GRDBRK_SlotAdd(self.sc, self.vl, "gridbreaker", "grdbrk_active")
        self.new_props = self.sc.gridbreaker[int(self.sc.grdbrk_active)]
        self.copy_obj()
        self.duplicate()
        return {'FINISHED'}
    
    def invoke(self, context, event) -> callable:
        return self.execute(context)
    
# ---------------------------------- Copy Object ------------------------------- OBJ COPY
    
class GRDBRK_OT_Copy(Operator):
    bl_label = "Copy"
    bl_idname = "grdbrk.copy"
    bl_description = "Copy Active Grid Breaker Settings"
    bl_options = {"UNDO"}
    
    def copy_settings(self):
        ps = self.props
        for p in dir(ps):
            if p.startswith(self.excepts):
                continue 
            elif p in self.mat_cols:
                setattr(self.op, p, [])
                for ms in getattr(ps,p):
                    for m in dir(ms):
                        if m.startswith(self.excepts):
                            continue
                        else:
                            prop = getattr(ms, m)
                            mat_list = getattr(self.op, p)
                            mat_list.append(prop)
            else:
                setattr(self.op, p, getattr(ps, p))
    
    def execute(self, context) -> set:
        self.sc = context.scene
        self.excepts = ('__', 'bl_', 'index', 'rna', 'name', 'id_')
        self.mat_cols = ('mat_main', 'mat_btm', 'mat_top', 'mat_sides',
                                                            'mat_real')
        self.props = self.sc.gridbreaker[int(self.sc.grdbrk_active)].obj.grdbrk
        self.op = bpy.types.GRDBRK_OT_copy
        self.copy_settings()
        return {'FINISHED'}
    
    def invoke(self, context, event) -> callable:
        return self.execute(context)
    
# --------------------------------- Paste Object ------------------------------ OBJ PASTE
    
class GRDBRK_OT_Paste(Operator):
    bl_label = "Paste"
    bl_idname = "grdbrk.paste"
    bl_description = "Paste Grid Breaker Settings to Active"
    bl_options = {"UNDO"}
    
    def paste_settings(self) -> None:
        ps = self.props
        self.sc.grdbrk.live_update=False
        self.props.mat_real.clear()
        self.props.mat_main.clear()
        self.props.mat_btm.clear()
        self.props.mat_top.clear()
        self.props.mat_sides.clear()
        for p in dir(self.op):
            if p.startswith(self.excepts):
                continue 
            elif p in self.mat_cols:
                m_prop = f"gridbreaker[{self.sc.grdbrk_active}].obj.grdbrk.{p}"
                m_active = f"gridbreaker[{self.sc.grdbrk_active}].obj.grdbrk.{p+'_active'}"
                while len(getattr(self.op, p)) > len(getattr(ps, p)):
                    GRDBRK_SlotAdd(self.sc, self.vl, m_prop, m_active)
                while len(getattr(self.op, p)) < len(getattr(ps, p)):
                    GRDBRK_SlotRemove(self.sc, self.vl, m_prop, m_active)
                for m, ms in zip(getattr(self.op, p), getattr(ps, p)):
                    ms.material = m
            else:
                setattr(ps, p, getattr(self.op, p))
        ps.mat_used = True
        self.sc.grdbrk.live_update=True
        GRDBRK_Gridbreaker(self.sc, self.vl)
    
    def execute(self, context) -> set:
        self.sc = context.scene
        self.vl = context.view_layer
        self.excepts = ('__', 'bl_', 'index', 'rna', 'name', 'obj', 'id_')
        self.mat_cols = ('mat_main', 'mat_btm', 'mat_top', 'mat_sides',
                                                            'mat_real')
        self.op = bpy.types.GRDBRK_OT_copy
        self.props = self.sc.gridbreaker[int(self.sc.grdbrk_active)].obj.grdbrk
        self.paste_settings()
        return {'FINISHED'}
    
    def invoke(self, context, event) -> callable:
        return self.execute(context)
    
# ------------------------------- Render Animation ----------------------------- RENDER ANIMATION
    
class GRDBRK_OT_RenderAnimation(Operator):
    bl_idname = 'grdbrk.render'
    bl_label = 'Render Animation'
    rendering = False
    render_started = False
    render_cancelled = False
    render_finished = False
    timer = None
    render_pre = None
    render_cancel = None
    render_complete = None
    frame = None
    frames = None
    frame_change = None
    render_path = None
    sc = None
    wm = None
    win = None
    handlers_store = []
    
    @classmethod
    def poll(self,context):
        ps = context.scene.grdbrk
        return ps.animatable and ps.frame_change_id
        
    def structure(self):            
        self.render_path = self.sc.render.filepath
        self.frame = self.sc.frame_current
        self.frames = self._frames()
        self.frame_change = Pointer(int(self.sc.grdbrk.frame_change_id))
        assert callable(self.frame_change)
        self.pre_render_handlers_clear()
        self.pre_render_handlers_append()
        self.rendering = False
        self.render_started = False
        self.render_cancelled = False
        self.render_finished = True
        self.timer_add()
        
    def pre_render_handlers_clear(self):
        '''Remove regular limiting functions from render handlers'''
        r_init = Pointer(int(self.sc.grdbrk.render_init_id))
        r_complete = Pointer(int(self.sc.grdbrk.render_complete_id))
        
        # just to keep links to the original limiting functions in the memory
        # so that they are not cleaned up by the garbage collector:
        self.handlers_store = [r_init, r_complete]
        
        while r_init in RenderInit:
            RenderInit.remove(r_init)
        while r_complete in RenderComplete:
            RenderComplete.remove(r_complete)
        while r_complete in RenderCancel:
            RenderCancel.remove(r_complete)
        
    def pre_render_handlers_append(self):
        '''
        Append internal functions to render handlers
        Store them as class call's attributes
        '''
                
        def render_pre(self):
            self.rendering = True
            self.render_finished = False
            
        def render_cancel(self):
            self.rendering = False
            if self.render_started:
                self.render_started = False
            self.render_cancelled = True
            self.render_finished = True
            
        def render_complete(self):
            self.rendering = False
            if self.render_started:
                self.render_started = False
            self.render_finished = True
            
        self.render_pre = lambda x: render_pre(self)
        self.render_cancel = lambda x: render_cancel(self)
        self.render_complete = lambda x: render_complete(self)
        RenderPre.append(self.render_pre)
        RenderCancel.append(self.render_cancel)
        RenderComplete.append(self.render_complete)
        
    def post_render_handlers_clear(self):
        '''Remove internal functions from render handlers'''
        while self.render_pre in RenderPre:
            RenderPre.remove(self.render_pre)
        while self.render_cancel in RenderCancel:
            RenderCancel.remove(self.render_cancel)
        while self.render_complete in RenderComplete:
            RenderComplete.remove(self.render_complete)
            
    def post_render_handlers_append(self):
        '''Append regular limiting functions to render handlers'''
        RenderInit.append(Pointer(int(self.sc.grdbrk.render_init_id)))
        RenderComplete.append(Pointer(int(self.sc.grdbrk.render_complete_id)))
        RenderCancel.append(Pointer(int(self.sc.grdbrk.render_complete_id)))
        
    def fix_write_still(self):
        '''
        Sets render.render()'s write_sill parameter to False
        This is done via adding this to depsgraph_update_post handler
        Because if called from the current operator write_still becomes True
        once the current operator is finished
        '''
        def fix_write_still(self, context):
            try:
                op = bpy.context.window_manager.operator_properties_last(
                                                                "render.render")
            except:
                return
            if op.write_still == True:
                op.write_still=False
                bpy.context.scene.update_tag()
                while fix_write_still in DepsgraphUpdate:
                    DepsgraphUpdate.remove(fix_write_still)
        
        self.wm.operator_properties_last("render.render").write_still=False    
        DepsgraphUpdate.append(fix_write_still)
    
    def timer_add(self, tick : float = .1):
        self.timer = self.wm.event_timer_add(time_step=tick, window = self.win)
        
    def timer_remove(self):
        try:
            for _ in range(3):
                self.wm.event_timer_remove(self.timer)
        except Exception as e:
            print(f'Exception {e} passed in GRDBRK_OT_RenderAnimation while\
                    atempt to delete')
    
    def _frames(self) -> List[int]:
        return list(range(self.sc.frame_start, self.sc.frame_end+1))
    
    def set_path(self, frame):
        self.sc.render.filepath = self.render_path+f'{frame:04d}'
        
    def fr_change_off(self):
        assert self.frame_change == Pointer(int(self.sc.grdbrk.frame_change_id))
        for f in FrameChange:
            if f == self.frame_change:
                while f in FrameChange:
                    FrameChange.remove(f)
    
    def fr_change_on(self):
        frame_change = self.frame_change = \
                                    Pointer(int(self.sc.grdbrk.frame_change_id))
        for f in FrameChange:
            if f == self.frame_change:
                return
        FrameChange.append(self.frame_change)
    
    def set_new_frame(self):
        frame = self.frames.pop(0)
        self.set_path(frame)
        self.fr_change_on()
        self.sc.frame_set(frame)
        self.fr_change_off()
        self.sc.update_tag()
        
    def render_new_frame(self):
        self.render_started=True
        bpy.ops.render.render('INVOKE_DEFAULT', animation=False, write_still=True)
        
    def cleanup(self, context):
        self.sc.render.filepath = self.render_path
        self.sc.frame_set(self.frame)
        self.timer_remove()
        self.fr_change_off() # in case cancelled
        self.fr_change_on()
        self.post_render_handlers_clear()
        self.post_render_handlers_append()
        self.fix_write_still()
        self.sc.update_tag()
    
    def invoke(self, context, event):
        return self.execute(context)
    
    def execute(self, context):
        self.wm = context.window_manager
        self.win = context.window
        self.sc = context.scene
        self.structure()
        self.wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        if event.type == 'ESC':
            print('Render Escaped')
            self.cleanup(context)
            return {'FINISHED'}
        elif event.type == 'TIMER':
            if self.render_cancelled:
                print('Render Cancelled')
                self.cleanup(context)
                return {'FINISHED'}
            elif self.render_started and not self.rendering:
            # force render launch if not started
                self.render_new_frame()
            elif self.render_finished:
                if len(self.frames):
                    self.set_new_frame()
                    self.render_new_frame()
                else:
                    self.cleanup(context)
                    return {'FINISHED'}           
        return {'PASS_THROUGH'}

class GRDBRK_OT_SlotAdd(Operator):
    '''Add new Grid Breaker'''
    bl_idname = "grdbrk.slot_add"
    bl_label = "Add Slot"
    bl_options = {"UNDO"}
    prop : StringProperty(default = "gridbreaker")
    active : StringProperty(default="grdbrk_active")
    
    def execute(self, context):
        bpy.ops.ed.undo_push()
        GRDBRK_SlotAdd(context.scene, context.view_layer, self.prop,
                                                        self.active)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return self.execute(context)
    
class GRDBRK_OT_SlotRemove(Operator):
    '''Remove active Grid Breaker'''
    bl_idname = "grdbrk.slot_remove"
    bl_label = "Remove Slot"
    bl_options = {"UNDO"}
    prop : StringProperty(default = "gridbreaker")
    active : StringProperty(default="grdbrk_active")
    
    def execute(self, context):
        bpy.ops.ed.undo_push()
        GRDBRK_SlotRemove(context.scene, self.prop, self.active)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return self.execute(context)
    
# ---------------------- UILists for Collection Properties  -------------------- UI LISTS

class GRDBRK_UIList:
    '''to be inherited by UILists'''
    
    def draw_item(self, _context, layout, _data, item, icon, _active_data,
                                                _active_propname, _index):
        commons = _context.scene.grdbrk
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if item.obj is None or not item.obj.name in _context.scene.objects:
                row = layout.row(align=True)
                op = row.operator("grdbrk.fix_missing", text="",icon='ERROR',
                                                                depress=True)
                op = row.operator("grdbrk.fix_missing", text="WARNING!",
                                                                depress=True)
                op = row.operator("grdbrk.fix_missing", text="",icon='ERROR',
                                                                depress=True)
                return
            layout.prop(item.obj, "name", text="", emboss=False, icon_value=icon)
            row = layout.row(align=True)
            if commons.show_select:
                row.prop(item.obj, "hide_select", text="", emboss=False)
            if commons.show_eye:
                row.prop(item.obj.grdbrk, "hide_eye", text="",emboss=False,
                        icon='HIDE_ON' if item.obj.grdbrk.hide_eye else "HIDE_OFF")
            if commons.show_viewport:
                row.prop(item.obj, "hide_viewport", text="", emboss=False)
            if commons.show_render:
                row.prop(item.obj, "hide_render", text="", emboss=False)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon="FILE_VOLUME")

class GRDBRK_UL_list(UIList, GRDBRK_UIList):
    '''
    Just inherits from 2 base classes
    Allows to use 1 BASE_UIList's draw() method for multiple CollectionProperty()
    '''
    pass

class GRDBRK_Materials:
    
    def draw_item(self, _context, layout, _data, item, icon, _active_data,
                                                _active_propname, _index):
        slot = item
        ma = slot.material
        
        layout.context_pointer_set("id", ma)
        
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if ma:
                row = layout.row(align=True)
                row.prop(ma, "name", text="", emboss=False, icon_value=icon)
#                row.prop(item, "amount", text="")                              # for later versions
            else:
                layout.label(text="", icon_value=icon)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)
            
class GRDBRK_UL_MaterialsReal(UIList, GRDBRK_Materials):
    pass
            
class GRDBRK_UL_MaterialsMain(UIList, GRDBRK_Materials):
    pass
            
class GRDBRK_UL_MaterialsTop(UIList, GRDBRK_Materials):
    pass

class GRDBRK_UL_MaterialsSides(UIList, GRDBRK_Materials):
    pass

class GRDBRK_UL_MaterialsBtm(UIList, GRDBRK_Materials):
    pass

# ----------------------------- Update Functions  ------------------------------ UPDATE
    
def GRDBRK_upd(self, context):
    sc = bpy.context.scene
    vl = bpy.context.view_layer
    props = sc.gridbreaker[sc.grdbrk_active]
    if sc.grdbrk.live_update and props.obj:
        GRDBRK_Gridbreaker(sc, vl)
            
def GRDBRK_col_mode_upd(self, context):
    '''
    Updater for GRDBRK_Common collection_use checkbox
    Using a single collection for all GRDBRK objects
    '''
    sc = context.scene
    vl = context.view_layer
    props = sc.gridbreaker[sc.grdbrk_active].obj.grdbrk
    commons = sc.grdbrk
    if commons.collection_use:
        if props.collection is not None:
            col = commons.collection = props.collection
        elif commons.collection is not None:
            col = commons.collection
        elif vl.active_layer_collection is not None:
            try:
                col = bpy.data.collections[vl.active_layer_collection.name]
            except KeyError:
                col = context.scene.collection
        else:
             col = context.scene.collection
        for gb in sc.gridbreaker:
            if gb.obj is not None:
                GRDBRK_Support.move_ob_to_col(gb.obj, col)
    else:
        for gb in sc.gridbreaker:
            if gb.obj is not None and gb.obj.grdbrk.collection is not None:
                col = gb.obj.grdbrk.collection
            elif vl.active_layer_collection is not None:
                try:
                    col = bpy.data.collections[
                                vl.active_layer_collection.name]
                except KeyError:
                    col = context.scene.collection
            else:
                 col = context.scene.collection            
            GRDBRK_Support.move_ob_to_col(gb.obj,col)
            
def GRDBRK_col_upd(self, context):
    '''
    Updater for GRDBRK_Common Collection pointer
    Whether to use a single or individual collection
    '''
    sc = context.scene
    vl = context.view_layer
    commons = sc.grdbrk
    if commons.collection is not None:
        col = commons.collection
    elif vl.active_layer_collection is not None:
            try:
                col = bpy.data.collections[vl.active_layer_collection.name]
            except KeyError:
                col = sc.collection
    else:
        col = sc.collection
    for gb in sc.gridbreaker:
        GRDBRK_Support.move_ob_to_col(gb.obj, col)

def GRDBRK_eye_hide(self, context):
    '''Updater for Eye button (controlling object's Hide Viewport)'''
    sc = context.scene
    for gb in sc.gridbreaker:
        gb.obj.hide_set(gb.obj.grdbrk.hide_eye)
    
def GRDBRK_live_upd(self, context):
    '''Updater for Live Update button'''
    sc = bpy.context.scene
    vl = bpy.context.view_layer
    if sc.grdbrk.live_update:
        GRDBRK_Gridbreaker(sc, vl)

def GRDBRK_frame_change(sc : Scene, vl : ViewLayer):
    '''Main animation function for frame_change handler'''
    animated_indices = []
    for gb in sc.gridbreaker:
        ob = gb.obj
        if ob.animation_data and ob.animation_data.action and len([fc for fc in
        ob.animation_data.action.fcurves if fc.data_path.startswith('grdbrk')]):
            animated_indices.append(gb.index)
   
    active_obj = vl.objects.active
    active_gb = int(sc.grdbrk_active)
    for i in animated_indices:
        while not sc.grdbrk_active == i:
            sc.grdbrk_active = i
        GRDBRK_Gridbreaker(sc, vl)
    sc.grdbrk_active = active_gb
    bpy.ops.object.select_all(action='DESELECT')
    vl.objects.active=active_obj
    if active_obj is not None:
        vl.objects.active.select_set(True)

def GRDBRK_render_init(sc : Scene):
    '''For render_init handler. To prevent crashes on original render'''
    assert sc.grdbrk.render_init_id != ""
    assert sc.grdbrk.frame_change_id != ""
    assert sc.grdbrk.animatable
    frame_change = Pointer(int(sc.grdbrk.frame_change_id))
    while frame_change in FrameChange:
        FrameChange.remove(frame_change)
    
def GRDBRK_render_complete(sc : Scene, vl : ViewLayer):
    '''
    For render_complete and render_cancel handlers.
    Set back frame change handler after original original render is finished
    and GB animation is on
    '''
    assert sc.grdbrk.animatable
    frame_change = lambda scene : GRDBRK_frame_change(scene, vl)
    sc.grdbrk.frame_change_id = str(id(frame_change))
    FrameChange.append(Pointer(int(sc.grdbrk.frame_change_id)))

@persistent
def GRDBRK_animatable(self, context):
    '''Updater for Animatable button'''
    sc = context.scene
    anim = lambda scene : GRDBRK_frame_change(scene, context.view_layer)
    complete = lambda scene : GRDBRK_render_complete(scene, context.view_layer)
    r_init = lambda scene : GRDBRK_render_init(scene)
    if sc.grdbrk.animatable:
        sc.grdbrk.frame_change_id = str(id(anim))
        sc.grdbrk.render_complete_id = str(id(complete))
        sc.grdbrk.render_init_id = str(id(r_init))
        FrameChange.append(Pointer(int(sc.grdbrk.frame_change_id)))
        RenderCancel.append(Pointer(int(sc.grdbrk.render_complete_id)))
        RenderComplete.append(Pointer(int(sc.grdbrk.render_complete_id)))
        RenderInit.append(Pointer(int(sc.grdbrk.render_init_id)))
    else:
        if not sc.grdbrk.frame_change_id or not sc.grdbrk.render_complete_id or\
        not sc.grdbrk.render_init_id:
            return
        anim = Pointer(int(sc.grdbrk.frame_change_id))
        complete = Pointer(int(sc.grdbrk.render_complete_id))
        r_init = Pointer(int(sc.grdbrk.render_init_id))
        while anim in FrameChange:
            FrameChange.remove(anim)
        while complete in RenderCancel:
            RenderCancel.remove(complete)
        while complete in RenderComplete:
            RenderComplete.remove(complete)
        while r_init in RenderInit:
            RenderInit.remove(r_init)
            
# -------------------------- Collection PropertyGroup  ------------------------- PROPERTY GROUPS
        
class GRDBRK_Common(PropertyGroup):
    live_update     : BoolProperty(default=True, options={'HIDDEN'},
                    update=GRDBRK_live_upd)
    animatable      : BoolProperty(default=False, options={'HIDDEN'},
                    name="Animatable", update=GRDBRK_animatable,
                    description="Take into account add-on settings' animation\
 while Render and Playback.\n\nWARNING!\nAnimation of the Grid Breaker add-on\
 settings may significantly\nslow down performance, lead to unstable work\
 and crashes.\nUse at your own risk")
    frame_change_id : StringProperty(default="",
                    description="pointer to the memory address of the \
frame change handler's function")
    render_init_id  : StringProperty(default="",
                    description="pointer to the memory address of the \
frame change handler's function")
    render_complete_id : StringProperty(default="",
                    description="pointer to the memory address of the \
frame change handler's function")
    lock_interface  : BoolProperty(default=False, options={'HIDDEN'})
    active          : IntProperty(default=0, options={'HIDDEN'},
                    description="Store Active GB index while animation")
    rendering       : BoolProperty(default=False, options={'HIDDEN'})
    collection_use  : BoolProperty(name='Use for all objects', default=False,
                    options={'HIDDEN'}, update=GRDBRK_col_mode_upd,
                    description="Use a single collection for all GB objects")
    collection      : PointerProperty(type=Collection, update=GRDBRK_col_upd,
                    description="Collection for all GB objects\n\
If not specified, Scene's active collection is used", name="Collection",
                    options={'HIDDEN'})
    show_select     : BoolProperty(default=False, options={'HIDDEN'},
                    description="Show Hide Select status preview")
    show_eye        : BoolProperty(default=False, options={'HIDDEN'},
                    description="Show Hide Viewport status preview")
    show_viewport   : BoolProperty(default=True, options={'HIDDEN'},
                    description="Show Hide Viewports status preview")
    show_render     : BoolProperty(default=True, options={'HIDDEN'},
                    description="Show Hide Render status preview")
    warning         : BoolProperty(default=True, options={'HIDDEN'},
                    name ="WARNING!",
                    description="Deleting Grid Breaker bjects manually may lead \
to unstable work and crashes.\nUse minus '-' button instead")
        
class GRDBRK_Material(PropertyGroup):
    name            : StringProperty(default="Material")
    index           : IntProperty(default=0)
    material        : PointerProperty(type=Material, update=GRDBRK_upd,
                    options={'HIDDEN'})
#    amount          : FloatProperty(default=.5,min=0, max=1, subtype='FACTOR', # for later versions
#                    update=GRDBRK_upd, options={'HIDDEN'})                     # for later versions
        
class GRDBRK_MaterialReal(PropertyGroup):
    name            : StringProperty(default="Material")
    index           : IntProperty(default=0)
    material        : PointerProperty(type=Material, options={'HIDDEN'})

class GRDBRK_Props(PropertyGroup):
    name            : StringProperty(default="Grid Breaker")
    index           : IntProperty(default=0)
    obj             : PointerProperty(type=Object)
    type            : EnumProperty(items={
                        ("SQUARES", "Squares", "Squares"),
                    }, default="SQUARES", description="Subdivision Patterns",
                    options={'HIDDEN'})
                    
class GRDBRK_ObjProps(PropertyGroup):
    obj             : PointerProperty(type=Object)
    hide_eye        : BoolProperty(default=False, update=GRDBRK_eye_hide,
                    options={'HIDDEN'})
    collection      : PointerProperty(type=Collection, update=GRDBRK_upd,
                    description="Choose Collection for Grid Breaker object\n\
If not specified, Scene's active collection is used", name= "Collection",
                    options={'HIDDEN'})
    top             : BoolProperty(default=True, update=GRDBRK_upd,
                    description="Top Planes")
    btm             : BoolProperty(default=True, update=GRDBRK_upd,
                    description="Bottom Planes")
    sides           : BoolProperty(default=True, update=GRDBRK_upd,
                    description="Side Planes")
    shade_smooth    : BoolProperty(default=True, update=GRDBRK_upd,
                    description="Smooth Shading")
    size            : FloatVectorProperty(size=2, default=(2,2), min=0,
                    update=GRDBRK_upd, subtype='TRANSLATION')
    height          : FloatProperty(default=0, min=0, update=GRDBRK_upd,
                    subtype='DISTANCE')
    position        : FloatVectorProperty(default=(0,0,0), update=GRDBRK_upd,
                    subtype='TRANSLATION')                    
    cells           : IntVectorProperty(size=2, default=(4,4), min=1,
                    update=GRDBRK_upd)
    cuts            : IntProperty(default=3, min=0, soft_max=5,
                    update=GRDBRK_upd)
    seed            : IntProperty(default=1, min=0, update=GRDBRK_upd)
    distribution    : FloatProperty(subtype='FACTOR', min=0, max=1, default=.5,
                    update=GRDBRK_upd)
    height_seed     : IntProperty(default=1, min=0, update=GRDBRK_upd)
    height_center   : FloatProperty(default=0, min=0, max=1,
                    subtype='FACTOR', update=GRDBRK_upd)
    height_random   : FloatProperty(subtype='FACTOR', min=0, max=1, default=0,
                    update=GRDBRK_upd)
    height_invert   : BoolProperty(default=False,
                    update=GRDBRK_upd)
    split_margins   : BoolProperty(default=False,
                    update=GRDBRK_upd)
#    merge           : BoolProperty(default=False) # for later versions
    margin_top      : FloatProperty(subtype='FACTOR', min=0, soft_max=1,
                    default=.95, update=GRDBRK_upd)
    margin_btm      : FloatProperty(subtype='FACTOR', min=0, soft_max=1,
                    default=.95, update=GRDBRK_upd)
    materials_split : BoolProperty(default=False, update=GRDBRK_upd,
                    description="Set individual materials for Bottom Top and \
Sides faces")
    materials_random: BoolProperty(default=False,
                    update=GRDBRK_upd, description="Random material for each face")
    mat_use_list    : BoolProperty(default=False, update=GRDBRK_upd,
                    description="Specify materials list")
    mat_used        : BoolProperty(default=False, options={'HIDDEN'},
                    description="Check if materials've been switched to any list")                
    mat_real        : CollectionProperty(type=GRDBRK_MaterialReal,
                    options={'HIDDEN'})
    mat_real_active : IntProperty(default=0,
                    options={'HIDDEN'})
    mat_main        : CollectionProperty(type=GRDBRK_Material,
                    options={'HIDDEN'})
    mat_main_active : IntProperty(default=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    mat_btm         : CollectionProperty(type=GRDBRK_Material,
                    options={'HIDDEN'})
    mat_btm_active  : IntProperty(default=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    mat_btm_seed    : IntProperty(default=1, min=0, update=GRDBRK_upd)
    mat_sides       : CollectionProperty(type=GRDBRK_Material,
                    options={'HIDDEN'})
    mat_sides_active: IntProperty(default=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    mat_sides_seed  : IntProperty(default=1, min=0, update=GRDBRK_upd)
    mat_top         : CollectionProperty(type=GRDBRK_Material,
                    options={'HIDDEN'})
    mat_top_active  : IntProperty(default=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    mat_top_seed    : IntProperty(default=1, min=0, update=GRDBRK_upd)
    
# -------------------------------- UI Panels  ---------------------------------- PANELS

class GRDBRK_Panels:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category="Grid Break"

class GRDBRK_PT_Panel(Panel,GRDBRK_Panels):
    bl_label = 'Gridbreaker'
    
    def set_active(self, context):
        sc = context.scene
        ob = context.object
        gr = sc.gridbreaker
        active = sc.grdbrk_active
        obs = [p.obj for p in gr if p.obj is not None]
        if ob in obs:
            for i, o in enumerate(obs):
                if o == ob:
                    if active != i:
                        sc.grdbrk_active = i
    
    def _lcol(self, vl : ViewLayer, col : Collection) -> LayerCollection:
        '''return layer collection for collection'''
        def find_col(lcol : LayerCollection, col : Collection):
            if lcol.name == col.name:
                return lcol
            elif lcol.children:
                for lc in lcol.children:
                    result = find_col(lc, col)
                    if result:
                        return result
        return find_col(vl.layer_collection, col)
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene
        commons = sc.grdbrk
        mainrow = layout.row(align=True)
        col = mainrow.column()
        
        row = col.row()
        rcol = row.column()
        rcol.prop(commons, "live_update", text="Live",
                                        icon='FILE_REFRESH')
            
        rcol = row.column()
        lrow = rcol.row(align = True)
        lrow.alignment='RIGHT'
        lrow.prop(commons, "show_select", text="", icon='RESTRICT_SELECT_OFF')
        lrow.prop(commons, "show_eye", text="", icon='HIDE_OFF')
        lrow.prop(commons, "show_viewport", text="", icon='RESTRICT_VIEW_OFF')
        lrow.prop(commons, "show_render", text="", icon="RESTRICT_RENDER_OFF")
        lrow.separator()
            
        row = col.row(align=True)
        row.template_list("GRDBRK_UL_list", "name",
                            context.scene,      "gridbreaker",
                            context.scene,      "grdbrk_active",
                            rows=5 if len(sc.gridbreaker) else 1)
                        
        ops = mainrow.column(align=True)
        ops.prop(commons, "animatable", text="", 
                icon='DECORATE_KEYFRAME' if commons.animatable else "KEYFRAME")
        ops.separator()
        op_add = ops.operator("grdbrk.slot_add", text = "", icon="ADD")
        op_add.prop =       "gridbreaker"
        op_add.active =     "grdbrk_active"
        op_rem = ops.operator("grdbrk.slot_remove", text = "", icon="REMOVE")
        op_rem.prop =       "gridbreaker"
        op_rem.active =     "grdbrk_active"
        if not len(sc.gridbreaker):
            return
        
        props = sc.gridbreaker[sc.grdbrk_active]
        if props.obj.name not in sc.objects:
            return
        
        props = props.obj.grdbrk      
        self.set_active(context)
        ops.separator()
        ops.operator("grdbrk.duplicate", text="", icon='DUPLICATE')
        ops.separator()
        ops.operator("grdbrk.copy", text="", icon='COPYDOWN')
        oprow = ops.row()
        oprow.enabled = hasattr(bpy.types.GRDBRK_OT_copy, "mat_main")
        oprow.operator("grdbrk.paste", text="", icon='PASTEDOWN')
        ops.separator()
        ops.operator("grdbrk.apply", text="", icon='EVENT_RETURN')
        
        
        if commons.animatable:
            col = layout.column(align=True)
            col.operator("grdbrk.render", text="Render Animation",
                                                        icon='RENDER_ANIMATION')        
        collection = commons.collection if commons.collection_use \
                                                        else props.collection
        col = layout.column(align=True)
        box = col.box()
        brow = box.row(align=True)
        brow.prop(commons, "collection_use", text = "",
                icon = 'OUTLINER_OB_POINTCLOUD' if commons.collection_use
                else 'LAYER_ACTIVE')
        brow.prop(commons if commons.collection_use else props, "collection",
                                                                    text="")
        if collection:
            brow.separator()
            brow.prop(collection, "hide_select", text = '', emboss=False)
            lcol = self._lcol(context.view_layer, collection)
            if lcol:
                brow.prop(lcol, "hide_viewport", text='',emboss=False)
            brow.prop(collection, "hide_viewport", text = '', emboss=False)
            brow.prop(collection, "hide_render", text = '', emboss=False)

class GRDBRK_ColPanels:        
    bl_parent_id="GRDBRK_PT_Panel"
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(self, context):
        sc = context.scene
        if not len(sc.gridbreaker):
            return False
        props = sc.gridbreaker[sc.grdbrk_active]
        return props.obj is not None and props.obj.name in sc.objects
        
class GRDBRK_PT_Geometry(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Geometry'
    bl_options = set()
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene       
        props = sc.gridbreaker[sc.grdbrk_active]
        row = layout.row(align=True)
        row.prop(props, "type", text='Type')
        
class GRDBRK_PT_Transforms(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Transforms'
    bl_parent_id = 'GRDBRK_PT_Geometry'
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene        
        props = sc.gridbreaker[sc.grdbrk_active]
        
        if not props.obj:
            return
        elif props.obj and props.obj.name not in sc.objects:
            bpy.data.objects.remove(props.obj)
            
        col = layout.column(align=True)
        col.prop(props.obj.grdbrk, "position", text="Position")
        
class GRDBRK_PT_Grid(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Grid'
    bl_parent_id = 'GRDBRK_PT_Geometry'
    bl_options = set()
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene        
        props = sc.gridbreaker[sc.grdbrk_active].obj.grdbrk
        row = layout.row(align=True, heading='Parts:')
        row.prop(props, "top", text="Top")
        row.prop(props, "sides", text="Sides")
        row.prop(props, "btm", text="Base")
        col = layout.column(align=True)
        row = col.row(align=True,heading='Size:')
        row.prop(props, "size", index=0, text="")
        row.prop(props, "size", index=1, text="")
        row = col.row()
        row.prop(props, "cells", text='Cells')
        
        col.separator()
        col.prop(props, "cuts", text='Subdivisions')
        col.prop(props, "seed", text='Seed')
        col.prop(props, "distribution", text='Distribution')
        col.separator()
        row = col.row(heading='Margins:')
        row.prop(props, "split_margins", text='Split')
        if props.split_margins:
            col.prop(props, "margin_btm", text='Bottom')
            col.prop(props, "margin_top", text='Top')
        else:
            col.prop(props, "margin_btm", text='Margin')
        col = layout.column(align=True, heading='Randomize Height:')
        col.prop(props, "height", text='Height')
        col.prop(props, "height_random", text='Random Height')
        col.prop(props, "height_center", text='Height Center Offset')
        col.prop(props, "height_seed", text='Seed')
        col.prop(props, "height_invert", text='Invert')

class GRDBRK_PT_Materials(Panel, GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Materials'
    
    def draw(self, context):
        pass
        
class GRDBRK_PT_Shading(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Shading'
    bl_parent_id = 'GRDBRK_PT_Materials'
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene
        obj = sc.gridbreaker[sc.grdbrk_active].obj
        props = obj.grdbrk
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "shade_smooth", text="Shade Smooth")
        row.prop(obj.data, "use_auto_smooth")
        if obj.data.use_auto_smooth:
            row = box.row(align=True)
            row.prop(obj.data, "auto_smooth_angle")
    
class GRDBRK_PT_MaterialsCommon(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Distribution'
    bl_parent_id = 'GRDBRK_PT_Materials'
    
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene       
        props = sc.gridbreaker[sc.grdbrk_active].obj.grdbrk
        row = layout.row()
        row.prop(props, "materials_split", text="Split")
        if not props.materials_split:
            row.prop(props, "materials_random", text="Faces")
            row.prop(props, "mat_use_list", text="List")
            row = layout.row(align=True)
            if not props.mat_use_list:
                row.prop(props, "mat_btm_seed", text='Seed')
        
class MaterialsPanels:
    bl_parent_id = 'GRDBRK_PT_MaterialsCommon'
    bl_options=set()
    
    @classmethod
    def poll(self, context):
        sc = context.scene
        if not len(sc.gridbreaker):
            return False
        elif sc.gridbreaker[sc.grdbrk_active].obj is None:
            return False
        props = sc.gridbreaker[sc.grdbrk_active].obj.grdbrk
        return props.materials_split and getattr(props, self.geometry_part)
    
    def uilists(self, context : Context, ul_class : str, prop : str,
                prop_active : str, path : str, path_active : str, seed : str):
        layout = self.layout
        sc = context.scene       
        props = sc.gridbreaker[sc.grdbrk_active].obj.grdbrk
        col = layout.column(align=True)
        col.prop(props, seed, text='Seed')
        col.separator()
        row = col.row(align=True)
        row.template_list(ul_class, "", props, prop, props, prop_active,
                            rows=2 if len(sc.gridbreaker) else 1)
        ops = row.column(align=True)
        op_add = ops.operator("grdbrk.slot_add", text = "", icon="ADD")
        op_add.prop = path
        op_add.active = path_active
        op_rem = ops.operator("grdbrk.slot_remove", text = "", icon="REMOVE")
        op_rem.prop = path
        op_rem.active = path_active
        if len(getattr(props, prop)):
            tm_props = getattr(props, prop)[getattr(props, prop_active)]
            tm_col = layout.column(align=True)
            tm_col.prop(tm_props, "material", text = "")
            
class GRDBRK_PT_MaterialsMain(Panel, GRDBRK_Panels, MaterialsPanels):
    bl_label = 'Materials list'
    
    @classmethod
    def poll(self, context):
        sc = context.scene
        props = sc.gridbreaker[sc.grdbrk_active].obj.grdbrk
        return props.mat_use_list and not props.materials_split
    
    def draw(self, context):
        sc = context.scene
        self.uilists(context,
            "GRDBRK_UL_MaterialsMain",
            "mat_main",
            "mat_main_active",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_main",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_main_active",
            "mat_btm_seed"
        )
        
class GRDBRK_PT_MaterialsTop(Panel, GRDBRK_Panels, MaterialsPanels):
    bl_label = 'Top'
    geometry_part = 'top'
    
    def draw(self, context):
        sc = context.scene
        self.uilists(context,
            "GRDBRK_UL_MaterialsTop",
            "mat_top",
            "mat_top_active",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_top",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_top_active",
            "mat_top_seed"
        )
        
class GRDBRK_PT_MaterialsSides(Panel, GRDBRK_Panels, MaterialsPanels):
    bl_label = 'Sides'
    geometry_part = 'sides'
    
    def draw(self, context):
        sc = context.scene
        self.uilists(context,
            "GRDBRK_UL_MaterialsSides",
            "mat_sides",
            "mat_sides_active",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_sides",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_sides_active",
            "mat_sides_seed"
        )        

class GRDBRK_PT_MaterialsBtm(Panel, GRDBRK_Panels, MaterialsPanels):
    bl_label = 'Bottom'
    geometry_part = 'btm'
    
    def draw(self, context):
        sc = context.scene
        self.uilists(context,
            "GRDBRK_UL_MaterialsBtm",
            "mat_btm",
            "mat_btm_active",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_btm",
            f"gridbreaker[{sc.grdbrk_active}].obj.grdbrk.mat_btm_active",
            "mat_btm_seed"
        )
            
def GRDBRK_index_upd(self, context):
    '''Set active Grid Breaker when its object is active in Scene'''
    sc = context.scene
    vl = context.view_layer
    props = sc.gridbreaker[sc.grdbrk_active]
    if props.obj is not None and props.obj != vl.objects.active:
        bpy.ops.object.select_all(action='DESELECT')
        if props.obj and props.obj.name in sc.objects:
            vl.objects.active = props.obj
            props.obj.select_set(True)

# --------------------------------- Register ----------------------------------- REGISTER

classes = [
    GRDBRK_Common,
    GRDBRK_Material,
    GRDBRK_MaterialReal,
    GRDBRK_Props,
    GRDBRK_ObjProps,
    GRDBRK_OT_Warning,
    GRDBRK_OT_Apply,
    GRDBRK_OT_Duplicate,
    GRDBRK_OT_Copy,
    GRDBRK_OT_Paste,
    GRDBRK_OT_FixMissingObj,
    GRDBRK_OT_RenderAnimation,
    GRDBRK_OT_SlotAdd,
    GRDBRK_OT_SlotRemove,
    GRDBRK_UL_list,
    GRDBRK_UL_MaterialsReal,
    GRDBRK_UL_MaterialsMain,
    GRDBRK_UL_MaterialsTop,
    GRDBRK_UL_MaterialsSides,
    GRDBRK_UL_MaterialsBtm,
    GRDBRK_PT_Panel,
    GRDBRK_PT_Geometry,
    GRDBRK_PT_Transforms,
    GRDBRK_PT_Grid,
    GRDBRK_PT_Materials,
    GRDBRK_PT_Shading,
    GRDBRK_PT_MaterialsCommon,
    GRDBRK_PT_MaterialsMain,
    GRDBRK_PT_MaterialsTop,
    GRDBRK_PT_MaterialsSides,
    GRDBRK_PT_MaterialsBtm,
]

@persistent
def on_load(self, context):
    if on_load in DepsgraphUpdate:
        if bpy.context.scene.grdbrk.animatable:
            GRDBRK_animatable(self, bpy.context)
        while on_load in DepsgraphUpdate:
            DepsgraphUpdate.remove(on_load)
    elif on_load in LoadPost:
        if bpy.context.scene.grdbrk.animatable:
            GRDBRK_animatable(self, bpy.context)

def register():
    for cl in classes:
        register_class(cl)
    Scene.gridbreaker = CollectionProperty(type=GRDBRK_Props)
    Scene.grdbrk_active = IntProperty(default=0, update=GRDBRK_index_upd)
    Scene.grdbrk = PointerProperty(type=GRDBRK_Common)
    Object.grdbrk = PointerProperty(type=GRDBRK_ObjProps)
    DepsgraphUpdate.append(on_load)
    LoadPost.append(on_load)
    
def unregister():
    for cl in reversed(classes):
        try:
            unregister_class(cl)
        except Exception as e :
            print(f"could not unregister {cl} because of {e}")

# ----------------------------------- Test ------------------------------------- TEST

if __name__ == "__main__":
    register()
