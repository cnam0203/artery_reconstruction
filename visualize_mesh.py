import nibabel as nib
from skimage import measure
import numpy as np
import trimesh

def visualize_artery_mesh(segment_data, voxel_sizes, index, path, all=False):
    scene = trimesh.scene.scene.Scene()
    
    if all:
        mask = np.isin(segment_data, index)
        copy_data = np.copy(segment_data)
        copy_data[mask == False] = 0
        
        verts, faces, normals, values = measure.marching_cubes(copy_data, level=0.5, spacing=voxel_sizes)
        faces = np.flip(faces, axis=1)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=3, implicit_time_integration=False, volume_constraint=True, laplacian_operator=None)
        
        scene.add_geometry(mesh)
        
    else:
        for i in index:
            mask = np.isin(segment_data, [i])
            copy_data = np.copy(segment_data)
            copy_data[mask == False] = 0
            
            verts, faces, normals, values = measure.marching_cubes(copy_data, level=0.5, spacing=voxel_sizes)
            faces = np.flip(faces, axis=1)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=3, implicit_time_integration=False, volume_constraint=True, laplacian_operator=None)

            scene.add_geometry(mesh)
    
    scene.export('output_mesh_gap.stl')
    
    