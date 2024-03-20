import nibabel as nib
from skimage import measure
import open3d as o3d
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
    
    
    # verts_offset = 0  # To keep track of the total number of vertices
        # all_verts = []
        # all_faces = []
        
        # for i in index:
        #     mask = np.isin(segment_data, [i])
        #     copy_data = np.copy(segment_data)
        #     copy_data[mask == False] = 0
            
        #     verts, faces, normals, values = measure.marching_cubes(copy_data, level=0.5, spacing=voxel_sizes)
        #     # faces = np.flip(faces, axis=1)
            
        #     # Adjust faces to index the correct vertices
        #     faces += verts_offset
            
        #     all_verts.append(verts)
        #     all_faces.append(faces)
            
        #     # Update verts_offset for the next iteration
        #     verts_offset += verts.shape[0]

        # # Concatenate all vertices and faces
        # all_verts = np.concatenate(all_verts)
        # all_faces = np.concatenate(all_faces)

        # # Create an Open3D TriangleMesh from the concatenated vertices and faces
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        # mesh.triangles = o3d.utility.Vector3iVector(all_faces)
        # mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])

        # # Perform laplacian smoothing
        # mesh = mesh.filter_smooth_laplacian(3, 0.5)
        # mesh.compute_triangle_normals()

        # # Export the mesh to an STL file
        # o3d.io.write_triangle_mesh('output_mesh.stl', mesh)
    
 
    
    