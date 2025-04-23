# when configuring spawn objects 
# make sure Y is facing UP by performing a 90 deg rotation along X 
import numpy as np
import bpy
import codecs
import json
import os

def read_np(path: str) -> None:
    """
        Open a file containg a serialized numpy array within
        Load the numpy array and return 
    """
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    py_arr = json.loads(obj_text) # python arr
    return np.array(py_arr)

class TrackLine():
    """
        This class renders a blender curve from a set of coordinates
        It also stores references to the object   
        
        https://blender.stackexchange.com/questions/180148/create-curve-from-numpy-array-using-python/180184#180184     
    """
    
    def __init__(self, mesh_name: str, coordinates: np.ndarray, weights: np.ndarray = None, closed: bool = True) -> None:
        self.coordinates = coordinates
        self.mesh_name = mesh_name
        self.weight = weights if weights else np.ones(len(self.coordinates))
        self.closed = closed
        
        self.spline = None
        self.obj = None
        
    def create_spline(self) -> None:
        # create a curve in blender
        curve_data = bpy.data.curves.new(name=self.mesh_name, type='CURVE')
        curve_data.dimensions = '3D'
        
        # add a poly spline and create len(coordinates) points
        spline = curve_data.splines.new(type='POLY')            
        spline.points.add(len(self.coordinates) - 1)
        
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        z = np.zeros(len(self.coordinates))
     
        # add the points
        spline.points.foreach_set("co", self._flatten(x,y,z, self.weight))
        
        if self.closed:
            spline.use_cyclic_u = self.closed
            spline.use_cyclic_v= self.closed

        self.curve_data = curve_data
        
    def link_obj(self) -> None:
        self.obj = bpy.data.objects.new(name=self.mesh_name, object_data=self.curve_data)
        bpy.context.collection.objects.link(self.obj)
            
    def _flatten(self, *args) -> np.ndarray:
        """
            Flattens the arguments into a singuler numpy array of size sum(args.size)
        """
        c = np.empty(sum(arg.size for arg in args))
        l = len(args)
        for i, arg in enumerate(args):
            c[i::l] = arg
        return c
                
coordinates = read_np(r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\test_track.json")
name = "test_mesh"

track_width = 12
segment_length = 0.25
kerb_width = 1

kerb_resize = track_width / 0.66
track = TrackLine(name, coordinates, closed=True)
track.create_spline()
track.link_obj()

#bpy.ops.mesh.primitive_plane_add(size=track_width, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
bpy.ops.object.editmode_toggle()


# THIS
bpy.ops.object.modifier_add(type='CURVE')
bpy.context.object.modifiers["Curve"].object = bpy.data.objects[name] # replace with name

# need to move object to first index in track
# toggle again
bpy.ops.object.editmode_toggle()


# export to blend file location
basedir = os.path.dirname(bpy.data.filepath)

if not basedir:
    raise Exception("Blend file is not saved")

view_layer = bpy.context.view_layer

obj_active = view_layer.objects.active
selection = bpy.context.selected_objects

bpy.ops.object.select_all(action='DESELECT')

# save fbx
for obj in selection:

    obj.select_set(True)

    # some exporters only use the active object
    view_layer.objects.active = obj

    name = bpy.path.clean_name(obj.name)
    fn = os.path.join(basedir, name)

    bpy.ops.export_scene.fbx(filepath=fn + ".fbx", use_selection=True)

    # Can be used for multiple formats
    # bpy.ops.export_scene.x3d(filepath=fn + ".x3d", use_selection=True)

    obj.select_set(False)

    print("written:", fn)


view_layer.objects.active = obj_active

for obj in selection:
    obj.select_set(True)