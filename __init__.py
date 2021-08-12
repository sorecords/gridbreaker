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

bl_info = {
    "name": "Gridbreaker",
    "author": "Andrey Sokolov",
    "version": (1, 0, 0),
    "blender": (2, 93, 0),
    "location": "View 3D > N-Panel > Grid Break",
    "description": "Generate grids",
    "warning": "",
    "wiki_url": "https://github.com/sorecords/gridbreaker/blob/main/README.md",
    "tracker_url": "https://github.com/sorecords/gridbreaker/issues",
    "category": "Mesh"
}

'''GRID BREAKER'''

import bpy
from bpy.types import Context, Operator, Panel, PropertyGroup, Scene, \
    Object, BlenderRNA, UIList, Collection, Mesh, VertexGroup, Material, \
    SolidifyModifier
from bpy.props import *
from bpy.utils import register_class, unregister_class
from typing import List, Tuple, Union, Dict
from mathutils import Vector
from random import randint, seed
from concurrent.futures import ThreadPoolExecutor # will be changed
import numpy as np # for the future development

# ----------------------- Grid Breaker internal classes ------------------------

class GRDBRK_Vert:
    
    def __init__(self, index : int, co : Vector):
        self.index = index
        assert type(co) == Vector
        self.co = co
    
    def __gt__(self, other : int) -> bool:
        return self.index > other
    
    def __ge__(self, other : int) -> bool:
        return self.index >= other
    
    def __lt__(self, other : int) -> bool:
        return self.index < other
    
    def __le__(self, other : int) -> bool:
        return self.index <= other
    
    def __eq__(self, other : int) -> bool:
        return self.index == other
    
    def __ne__(self, other : int) -> bool:
        return self.index != other
    
    def __hash__(self):
        return hash(str(self))

class GRDBRK_Face:
    
    def __init__(self, index : int, center : Vector, verts_proxy : List[Vector],
                 verts_real : List[GRDBRK_Vert], margin : float,
                 use_margin : bool = True, flip_normal : bool = False):
        assert type(center) == Vector
        assert len(verts_proxy)
        assert type(verts_proxy[0]) == Vector
        assert len(verts_real)
        assert type(verts_real[0].co) == Vector
        self.index = index
        self.center = center
        self.verts_proxy = verts_proxy      # List[Vector]
        self.verts_real = verts_real        # List[GRDBRK_Vert]
        self.margin = margin
        self.use_margin = use_margin
        self.flip_normal = flip_normal
        self._material=None
    
    @property
    def material(self) -> int:
        return self._material
    
    @material.setter
    def material(self, material : int) -> None:
        self._material = material
    
    def get_co(self, co : Vector) -> Vector:
        return self.center+(co-self.center)*self.margin
    
    def fix_co(self, vert : GRDBRK_Vert, index : int) -> None:
        vert.co = self.get_co(self.verts_proxy[index])
        
    def set_margins(self) -> None:
        '''
        margin is a float-factor between 0 and 1
        when 1 verts_real coordinates == verts.proxy
        when 0 verts real coordinates == center 
        '''
        for i, v in enumerate(self.verts_real):
            self.fix_co(v, i)
    
    def __gt__(self, other : int) -> bool:
        return self.index > other
    
    def __ge__(self, other : int) -> bool:
        return self.index >= other
    
    def __lt__(self, other : int) -> bool:
        return self.index < other
    
    def __le__(self, other : int) -> bool:
        return self.index <= other
    
    def __eq__(self, other : int) -> bool:
        return self.index == other
    
    def __ne__(self, other : int) -> bool:
        return self.index != other

class GRDBRK_Pydata:
    '''Main Grid Break Internal Mesh Operations: Create, Setup, etc.'''
    
    def __init__(self, props : CollectionProperty):
        self.props = props
        self.obj = props.obj
        self.mesh = self.obj.data
        self.size = props.size
        self.cells = props.cells
        self.margin = props.margin
        self.faces = []
        self.faces_splitted = []
        self.verts = []
        self.distribution = props.distribution  # how many faces on each iteration are affected
        self.seed = float(props.seed)           # random seed
        self.cuts = int(props.cuts)             # recursion depth
#        NOT IMPLEMENTED CONCEPT:
#            - store all vertex and faces info into numpy ndarrays
#            - don't create faces which are not going to be visible
#              because of the margins
#        self.cuts_threshold = self._cuts_threshold() # if < self.cuts : delete faces, no further cuts
#        self.face_num = self._face_num()
                
        self.get_bounds()
        self.split_cells()
        self.subdivide(self.faces, self.cuts)
        if self.check_materials():
            self.set_faces_materials()
        self.primary_faces = list(self.faces)
        solidified = self.solidify(self.faces, self.props.height, self.props.random_height,
                        self.props.height_seed)
        self.set_margins()
    
    def _cuts_threshold(self):
        '''
        NOT IMPLEMENTED CONCEPT
        If at some cuts level face size becomes <= 0 because margin value is too small:
        return the number of cut at wich it happens
        (faces of this level of cut should be deleted instead of splitting)
        '''
        threshold = 0
        for c in range(self.cuts):
            cut = c+1
            threshold += (.5**cut)
            if self.margin <= threshold:
                return cut
        return None
        
    def _face_num(self) -> None:
        '''
        NOT IMPLEMENTED CONCEPT
        Predictably calculate face number for Numpy ndarray to store Verts and Faces info to
        '''
        btm_faces = self.cells[0]*self.cells[1]
        faces_to_split = int(btm_faces*self.distribution) # number of faces to split on each cut
        assert faces_to_split <= btm_faces
        
        if self.cuts_threshold:
            for f in range(self.cuts_threshold):
                if f == (self.cuts_threshold-1):
                    btm_faces -= faces_to_split
                else:
                    btm_faces += (faces_to_split*3)
        else:
            for _ in range(self.cuts): 
                btm_faces += (faces_to_split*3) # because we change one existing face and add 3 new        
            
    def get_center(self, verts : List[Vector]) -> Vector:
        '''Get Grid Breaker face center coordinates'''
        center = Vector((0,0,0))
        for v in verts:
            center+=v
        return center/len(verts)
    
    def vert_new(self, v_co : Vector) -> GRDBRK_Vert:
        '''Create new Grid Break Vertex from coordinates'''
        v = GRDBRK_Vert(len(self.verts), Vector(v_co))
        self.verts.append(v)
        return v
    
    def check_materials(self):
        return True if len([m for m in self.props.materials
                if m.material is not None]) else False
    
    def set_faces_materials(self):
        r_seed = int(self.props.materials_seed)
        for f in self.faces:
            r_seed+=1
            seed(r_seed)
            f.material = randint(0, len(self.props.materials)-1)
        
    def face_new(self, verts_co : List[Vector], margin : float) -> GRDBRK_Face:
        '''Create new Grid Breaker Face'''
        verts_co = list([Vector(v) for v in verts_co])
        f_index = len(self.faces)
        verts = [self.vert_new(v) for v in verts_co]
        center = self.get_center(verts_co)
        face = GRDBRK_Face(f_index, center, verts_co, verts, margin)
        self.faces.append(face)
        return face
    
    def face_change_verts(self, face : GRDBRK_Face, new_co : List[Vector],
                                                margin : float) -> GRDBRK_Face:
        '''Set new Grid Breaker Face vertex coordinates'''
        assert len(face.verts_real) == len(face.verts_proxy) == len(new_co)
        for i in range(len(new_co)):
            face.verts_proxy[i] = Vector(new_co[i])
            face.verts_real[i].co = Vector(new_co[i])
        face.center = self.get_center(face.verts_proxy)
        face.margin=margin
        return face
    
    def fs_v_line(self, verts : List[Vector], cuts : int = 1) -> List[Vector]:
        '''Face split - Vecrtices - new cutted vertices co'''
        assert len(verts) == 2
        new_cs = []
        if cuts == 0:
            return verts
        vert_vector = (verts[1]-verts[0])/cuts
        v = verts[0]
        for _ in range(cuts):
            v=Vector(v+vert_vector)
            new_cs.append(v)
        return new_cs

    def fs_all_verts(self, verts : List[Vector], cut_x : int,
                                            cut_y : int) -> List[List[Vector]]:
        '''Face split - Vertices - all vertices co'''
        row_btm = [verts[0]] + self.fs_v_line([verts[0],verts[3]], cut_y)
        row_top = [verts[1]] + self.fs_v_line([verts[1],verts[2]], cut_y)
        verts_cs = []
        for rb, rt in zip(row_top, row_btm):
            col = [rb] + self.fs_v_line([rb, rt], cut_x)
            verts_cs.append(col)
        return verts_cs
    
    def fs_verts_from_cs(self, verts_co : List[Vector]) -> List[GRDBRK_Vert]:
        '''Get new GB verts coordinates in the middle of the source coordinates'''
        return [self.vert_new(c) for c in verts_co]
    
    def face_split(self, face : GRDBRK_Face, cut_x : int = 1, cut_y : int = 1,
                    margin : float =0) -> List[GRDBRK_Face]:
        '''
        Split face to new faces
        sx = how many faces on X axis
        sy = how many new faces on Y axis
        '''
        faces_new = []
        new_verts_co = self.fs_all_verts(face.verts_proxy, cut_x=cut_x,
                                                            cut_y=cut_y)
        source_used = False
        for i, row in enumerate(new_verts_co):
            if i+2 > len(new_verts_co):
                continue
            for j, col in enumerate(row):
                if j+2 > len(row):
                    break
                vcs0 = new_verts_co[i][j]
                vcs1 = new_verts_co[i+1][j]
                vcs2 = new_verts_co[i+1][j+1]
                vcs3 = new_verts_co[i][j+1]
                
                verts_cs = [vcs0, vcs1, vcs2, vcs3]
                
                if not source_used:
                    self.face_change_verts(face, verts_cs, margin)
                    source_used = True
                    faces_new.append(face)
                else:
                    f = self.face_new(verts_cs, margin)
                    faces_new.append(f)
        return faces_new
            
    def filter_faces(self, faces : List[GRDBRK_Face]) -> List[GRDBRK_Face]:
        '''filter faces for splitting'''
        faces_total = len(faces)
        faces_to_affect = int(faces_total*self.distribution)
        result_indices = set()
        for _ in range(faces_to_affect):
            seed(self.seed)
            r = randint(0, faces_total-1)
            while r in result_indices:
                self.seed+=1
                seed(self.seed)
                r = randint(0, faces_total-1)
            result_indices.add(r)
        faces = [faces[i] for i in result_indices]
        return faces
    
    def get_bounds(self) -> None: # STEP 1
        '''set single face bounds as pydata'''
        x = self.size[0]/2
        y = self.size[1]/2
        
        vcs0 = Vector((-x,-y, 0))
        vcs1 = Vector((-x, y, 0))
        vcs2 = Vector(( x, y, 0))
        vcs3 = Vector(( x,-y, 0))
        
        verts_cs = [vcs0, vcs1, vcs2, vcs3]
        self.face_new(verts_cs, self.margin)
        
    def split_cells(self) -> None: # STEP 2
        self.face_split(self.faces[0], cut_x=self.cells[0], cut_y=self.cells[1],
                                                            margin=self.margin)
    
    def subdivide(self, faces : List[GRDBRK_Face], cuts : int) -> None: #STEP 3
        if cuts <= 0:
            return
        affected_faces = self.filter_faces(faces)
        margin = max(0, (1-(1-self.margin)*(2*(self.cuts-cuts+1))))
        splitted_faces = []

        for f in affected_faces:
            faces_new = self.face_split(f, cut_x=2, cut_y=2, margin=margin)
            splitted_faces+=faces_new
        self.subdivide(splitted_faces, cuts-1)
            
    def random_float(self, start: float , end : float, r_seed : int) -> float:
        start = int(start*100)
        end = int(end*100)
        seed(r_seed)
        return randint(start, end)/100
    
    def duplicate_faces(self, faces : List[GRDBRK_Face]) -> Tuple[List[GRDBRK_Face]]:
        '''Duplicate faces, return 2 lists: with source and new faces'''
        src_faces = list(faces)
        trg_faces = []
             
        for f in src_faces:
            nf = self.face_new(f.verts_proxy, f.margin)
            nf.material = f.material
            f.flip_normal = True
            trg_faces.append(nf)
        return src_faces, trg_faces
    
    def set_height(self, f : GRDBRK_Face, height, rand_height, r_seed) -> None:
        r_factor = self.random_float(0, rand_height, r_seed)
        f_height = height - height*r_factor
        for i in range(len(f.verts_proxy)):
            f.verts_proxy[i] = Vector(f.verts_proxy[i]+Vector((0,0,f_height)))
            f.verts_real[i].co = Vector(f.verts_real[i].co+Vector((0,0,f_height)))
        f.center+=Vector((0,0,f_height))
          
    def offset_height(self, faces : List[GRDBRK_Face], height : float,\
                rand_height : float, r_seed : int) -> List[GRDBRK_Face]:
        r_seed = int(r_seed)
        seeds = range(r_seed, len(faces)+r_seed, 1)
        
        with ThreadPoolExecutor() as executor:
            executor.map(lambda f, rs:
                    self.set_height(f, height, rand_height, rs), faces, seeds)
        
        return faces
         
    def bridge_faces(self, faces : Tuple[GRDBRK_Face],
                                        br_faces : List[GRDBRK_Face]) -> None:
        '''Rim sides faces'''
        vb = faces[0].verts_real
        vt = faces[1].verts_real
        for i, b in enumerate(vb):
            index = i-len(vb)
            nv = [vb[index+1], vt[index+1], vt[index], b]
            nvprx = [v.co for v in nv]
            # face to be created manually without using self.face_new()
            # to avoid creating additional vertices
            face = GRDBRK_Face(
                    index=len(self.faces),
                    center=self.get_center(nvprx),
                    verts_proxy=nvprx,
                    verts_real=nv,
                    margin=1,
                    use_margin=False
            )
            face.material = faces[0].material
            self.faces.append(face)
            br_faces.append(face)
         
    def bridge_face_lists(self, face_lists : Tuple[List[GRDBRK_Face]]
                                        ) -> Tuple[List[GRDBRK_Face]]:
        '''
        Link pairs of faces with face bridges
        return tuple with 3 faces lists:
        btm, top, bridge
        '''
        br_faces = []
        b = face_lists[0]
        t = face_lists[1]
        assert len(b) == len(t)
        pairs = [(b[i],t[i]) for i in range(len(b))]
        
        with ThreadPoolExecutor() as executor:
            executor.map(lambda p : self.bridge_faces(p, br_faces), pairs)
            
        return face_lists[0], face_lists[1], br_faces
    
    def solidify(self, faces : List[GRDBRK_Face], height : float,
                rand_height : float, r_seed : int) -> Tuple[List[GRDBRK_Face]]:
        if height == 0:
            all_faces = [faces,[],[]]
            return all_faces
        if self.props.top:
            if self.props.btm:
                face_lists = self.duplicate_faces(faces)
            else:
                face_lists = [[],faces]
            self.offset_height(face_lists[1], height, rand_height, r_seed)
        else:
            face_lists = [faces,[]]
        
        if self.props.sides:
            if len(face_lists[1]):
                all_faces = self.bridge_face_lists(face_lists)
            else:
                raise NotImplementedError
        else:
            all_faces = [face_lists[0], face_lists[1], []]
        return all_faces        
        
    def set_margins(self) -> None: # STEP 4
        def setmrg(f : GRDBRK_Face):
            if f.use_margin:
                f.set_margins()
        
        with ThreadPoolExecutor() as executor:
            executor.map(setmrg, self.faces)
                
# ------------------------------- Main Operators -------------------------------

class GRDBRK_OT_gridbreaker(Operator):
    bl_label = "Grid Breaker"
    bl_idname = "grdbrk.update"
    bl_options = {"UNDO"}
    
    def set_object_materials(self):
        for m in self.props.materials:
            self.obj.data.materials.append(m.material)
        for f, fgr in zip(self.obj.data.polygons, self.pydata.faces):
            f.material_index = fgr.material if fgr.material else 0
            
        
    def get_pydata(self):
        self.pydata = GRDBRK_Pydata(self.props)
        verts = []
        edges = []
        faces = []
        for f in self.pydata.faces:
            face_verts = []
            for v in f.verts_real:
                face_verts.append(v.index)
                verts.append(v.co)
            faces.append(tuple(reversed(face_verts)))
        return { "verts":verts, "edges":edges, "faces":faces }
    
    def mesh_data_set(self) -> None:
        self.mesh.from_pydata(
                vertices=self.mesh_data["verts"],
                edges=[],
                faces=self.mesh_data["faces"]
        )
        return
    
    def flip_normals(self) -> None:
        for f in self.pydata.faces:
            if f.flip_normal:
                self.mesh.polygons[f.index].flip()
    
    def set_collection(self):
        col = self.props.collection
        if col and self.obj.name not in col.objects:
            for c in bpy.data.collections:
                if self.obj.name in c.objects:
                    c.objects.unlink(self.obj)
            if self.obj.name in self.sc.collection.objects:
                self.sc.collection.objects.unlink(self.obj)
            col.objects.link(self.obj)
    
    def execute(self, context):
        self.sc = context.scene
        self.props = self.sc.gridbreaker[self.sc.grdbrk_active]
        self.obj = self.props.obj
        if self.obj.name != self.props.obj_name:
            self.obj.name = self.props.obj_name
        self.old_mesh = self.obj.data
        self.old_mesh.name+="_tmp"
        self.mesh = bpy.data.meshes.new(self.obj.name)
        self.mesh.name = self.obj.name
        self.obj.data = self.mesh
        if all((not self.props.top, not self.props.sides, not self.props.btm)):
            return {'CANCELLED'}
        if (not self.props.top or not self.props.btm) and self.props.sides:
            # TODO : Remove after generating Sides without Top and Btm is implemented
            self.props.sides=False
        self.pydata = None
        self.mesh_data = self.get_pydata()
        self.mesh_data_set()
        if all((self.props.top, self.props.sides)):
            self.flip_normals()
        self.set_object_materials()
        bpy.data.meshes.remove(self.old_mesh)
        self.set_collection()
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return self.execute(context)

class GRDBRK_OT_New(Operator):
    bl_label = "New Base Form"
    bl_idname = "grdbrk.new_base_mesh"
    bl_options = {"UNDO"}
    
    def get_pydata(self) -> Dict[str, Union[List[Tuple[float]], List[Tuple[int]]]]:
        verts = self.get_verts()
        faces = self.get_faces()
        return {"verts" : verts, "faces" : faces}
    
    def mesh_data_set(self) -> None:
        self.mesh.from_pydata(vertices=self.pydata["verts"],edges=[],
                                faces=self.pydata["faces"])
        return
    
    def mesh_new(self) -> Mesh:
        return bpy.data.meshes.new(self.props.obj_name)
    
    def get_verts(self) -> List[Tuple[float]]:
        x = self.props.size[0]/2
        y = self.props.size[1]/2
        verts = [
            (-x, -y, 0), # -1,-1, 0
            (-x, y, 0),  # -1, 1, 0
            (x, y, 0),   #  1, 1, 0
            (x, -y, 0)   #  1,-1, 0
        ]
        return verts
    
    def get_faces(self) -> List[Tuple[int]]:
        return [(3,2,1,0)]
    
    def obj_new(self) -> Object:
        return bpy.data.objects.new(self.props.obj_name, self.mesh)
    
    def col_get(self) -> Collection:
        if self.props.collection:
            return self.props.collection
        elif self.vl.active_layer_collection is not None:
            try:
                col = bpy.data.collections[self.vl.active_layer_collection.name]
            except KeyError:
                col = self.sc.collection
            return col
        else:
            return self.sc.collection
        
    def col_set(self) -> None:
        self.col.objects.link(self.obj)
        
    def obj_set(self) -> None:
        self.props.obj = self.obj
    
    def execute(self, context):
        self.sc = context.scene
        self.vl = context.view_layer
        self.props = self.sc.gridbreaker[self.sc.grdbrk_active]
        self.mesh = self.mesh_new()
        self.pydata = self.get_pydata()
        self.mesh_data_set()
        self.obj = self.obj_new()
        self.col = self.col_get()
        self.col_set()
        self.obj_set()
        bpy.ops.grdbrk.update('INVOKE_DEFAULT')
        return {'FINISHED'}

#  ----------------------------- UILists Operators  ----------------------------

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

class GRDBRK_UL_SlotAdd(Operator, PropPathParse):
    '''Add new String System'''
    bl_idname = "grdbrk.slot_add"
    bl_label = "Add Slot"
    prop : StringProperty(default = "gridbreaker")
    active : StringProperty(default="grdbrk_active")
    
    def invoke(self, context, event):
        bpy.ops.ed.undo_push()
        sc = context.scene
        props = self.get_prop(sc, self.prop)
        props.add()
        index = len(props)-1
        self.set_prop(sc, self.active, index)
        props[index].index = index
        if self.prop == 'gridbreaker':
            bpy.ops.grdbrk.new_base_mesh()
        return {'FINISHED'}
    
class GRDBRK_UL_SlotRemove(Operator, PropPathParse):
    '''Remove active String System'''
    bl_idname = "grdbrk.slot_remove"
    bl_label = "Remove Slot"
    prop : StringProperty(default = "gridbreaker")
    active : StringProperty(default="grdbrk_active")
    
    def invoke(self, context, event):
        bpy.ops.ed.undo_push()
        sc = context.scene
        index = self.get_prop(sc, self.active)
        props = self.get_prop(sc, self.prop)
        if self.prop == 'gridbreaker':
            bpy.data.objects.remove(props[index].obj)
        props.remove(index)
        for i, ar in enumerate(props):
            ar.index = i
        self.set_prop(sc, self.active, index-1)
        return {'FINISHED'}
    
# ---------------------- UILists for Collection Properties  --------------------

class GRDBRK_UIList:
    '''to be inherited by UILists'''
    
    def draw_item(self, _context, layout, _data, item, icon, _active_data,
                                                _active_propname, _index):        
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.prop(item, "name", text="", emboss=False, icon_value=icon)
            layout.label(text="", icon="MESH_GRID")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon="MESH_GRID")

class GRDBRK_UL_list(UIList, GRDBRK_UIList):
    '''
    Just inherits from 2 base classes
    Allows to use 1 BASE_UIList's draw() method for multiple CollectionProperty()
    '''
    pass

class GRDBRK_UL_Materials(UIList):
    
    def draw_item(self, _context, layout, _data, item, icon, _active_data,
                                                _active_propname, _index):
        slot = item
        ma = slot.material
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if ma:
                layout.prop(ma, "name", text="", emboss=False, icon_value=icon)
            else:
                layout.prop(slot, "material", text="", emboss=False, icon_value=icon)
                layout.label(text="", icon_value=icon)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)

# -------------------------- Collection PropertyGroup  -------------------------
    
def GRDBRK_upd(self, context):
    sc = bpy.context.scene
    props = sc.gridbreaker[sc.grdbrk_active]
    if props.live_update and props.obj:
        bpy.ops.grdbrk.update('INVOKE_DEFAULT')
        
class GRDBRK_Material(PropertyGroup):
    name            : StringProperty(default="Material")
    index           : IntProperty(default=0)
    material        : PointerProperty(type=Material, update=GRDBRK_upd,
        options={'HIDDEN'})

class GRDBRK_Props(PropertyGroup):
    name            : StringProperty(default="Grid Breaker")
    index           : IntProperty(default=0)
    obj_name        : StringProperty(default="Grid Breaker", update=GRDBRK_upd,
                    description="Grid Breaker object name", options={'HIDDEN'})
    collection      : PointerProperty(type=Collection, update=GRDBRK_upd,
                    description="Choose Collection for Grid Breaker object\n\
If not specified, Scene's active collection is used", options={'HIDDEN'})
    size            : FloatVectorProperty(size=2, default=(2,2), min=1,
                    update=GRDBRK_upd, options={'HIDDEN'})
    live_update     : BoolProperty(default=True, options={'HIDDEN'})
    top             : BoolProperty(default=True, update=GRDBRK_upd,
                    description="Top Planes", options={'HIDDEN'})
    btm             : BoolProperty(default=True, update=GRDBRK_upd,
                    description="Bottom Planes", options={'HIDDEN'})
    sides           : BoolProperty(default=True, update=GRDBRK_upd,
        description="Currently can be generated only together with both Top and \
Bottom Planes", options={'HIDDEN'})
    obj             : PointerProperty(type=Object, options={'HIDDEN'})
    cells           : IntVectorProperty(size=2, default=(4,4), min=0,
                    update=GRDBRK_upd, options={'HIDDEN'})
    cuts            : IntProperty(default=3, min=0, soft_max=5,
                    update=GRDBRK_upd, options={'HIDDEN'})
    seed            : IntProperty(default=1, min=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    distribution    : FloatProperty(subtype='FACTOR', min=0, max=1, default=.5,
                    update=GRDBRK_upd, options={'HIDDEN'})
    height_seed     : IntProperty(default=1, min=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    height          : FloatProperty(min=0, default=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    random_height   : FloatProperty(subtype='FACTOR', min=0, max=1, default=0,
                    update=GRDBRK_upd, options={'HIDDEN'})
    margin          : FloatProperty(subtype='FACTOR', min=0, max=1, default=.95,
                    update=GRDBRK_upd, options={'HIDDEN'})
    materials_seed  : IntProperty(default=1, min=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    materials       : CollectionProperty(type=GRDBRK_Material,
                    options={'HIDDEN'})
    materials_active: IntProperty(default=0, update=GRDBRK_upd,
                    options={'HIDDEN'})
    
# -------------------------------- UI Panels  ----------------------------------

class GRDBRK_Panels:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category="Grid Break"

class GRDBRK_PT_Panel(Panel,GRDBRK_Panels):
    bl_label = 'Gridbreaker'
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene
        col = layout.column(align=True)
        row = col.row(align=True)
        row.template_list("GRDBRK_UL_list", "name",
                            context.scene,      "gridbreaker",
                            context.scene,      "grdbrk_active",
                            rows=3 if len(sc.gridbreaker) else 1)
        ops = row.column(align=True)
        op_add = ops.operator("grdbrk.slot_add", text = "", icon="ADD")
        op_add.prop =       "gridbreaker"
        op_add.active =     "grdbrk_active"
        op_rem = ops.operator("grdbrk.slot_remove", text = "", icon="REMOVE")
        op_rem.prop =       "gridbreaker"
        op_rem.active =     "grdbrk_active"
        if not len(sc.gridbreaker):
            return
        props = sc.gridbreaker[sc.grdbrk_active]
        row = col.row()
        row.prop(props, "live_update", text = "live update")
        if not props.live_update:
            row.operator("grdbrk.update", text="",icon="FILE_REFRESH")
        
        col.prop(props, "obj_name", text="name")
        col.prop(props, "collection")
        
class GRDBRK_ColPanels:        
    bl_parent_id="GRDBRK_PT_Panel"
    bl_options = {"DEFAULT_CLOSED"}
    
    @classmethod
    def poll(self, context):
        sc = context.scene
        if not len(sc.gridbreaker):
            return False
        props = sc.gridbreaker[sc.grdbrk_active]
        return props.obj is not None
        
class GRDBRK_PT_Geometry(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Geometry'
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene       
        props = sc.gridbreaker[sc.grdbrk_active]
        row = layout.row(align=True)
        row.prop(props, "top", text="Top")
        row.prop(props, "btm", text="Bottom")
        row.prop(props, "sides", text="Sides")

class GRDBRK_PT_Grid(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Grid Settings'
    bl_options = set()
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene        
        props = sc.gridbreaker[sc.grdbrk_active]
        col = layout.column(align=True)
        row = col.row()
        row.prop(props, "size")
        
        if not props.obj:
            col.operator("grdbrk.new_base_mesh")
            return
        elif props.obj and props.obj.name not in sc.objects:
            bpy.data.objects.remove(props.obj)       
        
        row = col.row()
        row.prop(props, "cells")
        col.separator()
        col.separator()
        col.prop(props, "seed")
        col.separator()
        col.prop(props, "margin")
        col.separator()
        col.prop(props, "cuts")
        col.prop(props, "distribution")
        col.separator()
        col.prop(props, "height_seed")
        col.prop(props, "height")
        col.prop(props, "random_height")

class GRDBRK_PT_Materials(Panel,GRDBRK_Panels, GRDBRK_ColPanels):
    bl_label = 'Materials'
    
    def draw(self, context):
        layout = self.layout
        sc = context.scene       
        props = sc.gridbreaker[sc.grdbrk_active]
        col = layout.column(align=True)
        col.prop(props, "materials_seed")
        row = col.row(align=True)
        row.template_list("GRDBRK_UL_Materials", "",
                            props,      "materials",
                            props,      "materials_active",
                            rows=2 if len(sc.gridbreaker) else 1)
        ops = row.column(align=True)
        op_add = ops.operator("grdbrk.slot_add", text = "", icon="ADD")
        op_add.prop =       f"gridbreaker[{sc.grdbrk_active}].materials"
        op_add.active =     f"gridbreaker[{sc.grdbrk_active}].materials_active"
        op_rem = ops.operator("grdbrk.slot_remove", text = "", icon="REMOVE")
        op_rem.prop =       f"gridbreaker[{sc.grdbrk_active}].materials"
        op_rem.active =     f"gridbreaker[{sc.grdbrk_active}].materials_active"
        if len(props.materials):
            tm_props = props.materials[props.materials_active]
            tm_col = layout.column(align=True)
            tm_col.prop(tm_props, "material", text = "")
            
# --------------------------------- Register -----------------------------------

classes = [
    GRDBRK_Material,
    GRDBRK_Props,
    GRDBRK_OT_gridbreaker,
    GRDBRK_OT_New,
    GRDBRK_UL_SlotAdd,
    GRDBRK_UL_SlotRemove,
    GRDBRK_UL_Materials,
    GRDBRK_UL_list,
    GRDBRK_PT_Panel,
    GRDBRK_PT_Geometry,
    GRDBRK_PT_Grid,
    GRDBRK_PT_Materials,
]

def register():
    for cl in classes:
        register_class(cl)
    Scene.gridbreaker = CollectionProperty(type=GRDBRK_Props)
    Scene.grdbrk_active = IntProperty(default=0)
    
#    TODO: for True Motion Blur compatibility:
#    if not GRDBRK_upd in bpy.app.handlers.frame_change_pre:
#        bpy.app.handlers.frame_change_pre.append(GRDBRK_upd)
    
def unregister():
    del Scene.gridbreaker
    del Scene.grdbrk_active
    for cl in reversed(classes):
        unregister_class(cl)

# ----------------------------------- Test -------------------------------------

if __name__ == "__main__":
    register()
