import os
import torch
import trimesh
import argparse
import numpy as np

from varen import VAREN


def main():
    parser = argparse.ArgumentParser(description="Export VAREN meshes to a specified folder.")
    parser.add_argument('--model_path', type=str, default='models/varen', help='Location of saved model files.')
    parser.add_argument('--output_path', type=str, default="", help="Folder to save the exported mesh files. Default is the current directory.")
    parser.add_argument('--save-meshes', action='store_true', help="Option to save meshes.")
    args = parser.parse_args()

    output_path = args.output_path

    # Check if the output folder exists
    if output_path and not os.path.exists(output_path):
        create_folder = input(f"The folder '{output_path}' does not exist. Would you like to create it? (y/n): ").strip().lower()
        if create_folder == 'y':
            os.makedirs(output_path)
            print(f"Created folder: {output_path}")
        else:
            print("Folder not created. Exiting.")
            return
        

    model_path = args.model_path

    varen_base = VAREN(model_path, use_muscle_deformations=False)
    varen_ext = VAREN(model_path, use_muscle_deformations=True)
    NUM_JOINTS = varen_ext.NUM_JOINTS

    pose = (torch.rand(1, NUM_JOINTS * 3) - 0.5) * 0.3
    shape = torch.rand(1, 39) 
    transl = torch.rand(1,3)
    global_orient = torch.rand(1,3)

    MESH_COLOUR = (torch.rand(3) * 255).int()

    # Process base model
    model_output_base = varen_base(body_pose=pose, betas=shape, transl=transl, global_orient=global_orient)
    vertices_base = model_output_base.vertices.squeeze().detach().numpy()
    joints_base = model_output_base.joints.squeeze().detach().numpy()
    faces_base = varen_base.faces

    mesh_base = trimesh.Trimesh(vertices_base, faces_base, face_colors=MESH_COLOUR)
    joints_pcd_base = trimesh.points.PointCloud(joints_base, size=0.01)

    # Process extended model
    model_output_ext = varen_ext(body_pose=pose, betas=shape, transl=transl, global_orient=global_orient)
    vertices_ext = model_output_ext.vertices.squeeze().detach().numpy()
    joints_ext = model_output_ext.joints.squeeze().detach().numpy()
    faces_ext = varen_ext.faces

    mesh_ext = trimesh.Trimesh(vertices_ext, faces_ext, face_colors=(MESH_COLOUR-255)*-1)
    joints_pcd_ext = trimesh.points.PointCloud(joints_ext, size=0.01)

    #mesh_ext.visual.face_colors[:] = np.array([138, 42, 173, 150]) # purple
    #normals = (mesh_ext.face_normals + 1) / 2 
    #colours = (normals * 255).astype(np.uint8)
    #colours = np.hstack((colours, np.full((colours.shape[0], 1), 200)))
    #colors = trimesh.visual.interpolate(values=mesh_ext.face_normals, color_map='viridis')
    
    #mesh_ext.visual.face_colors = colours
    joints_pcd_ext.colors = np.array([0, 0, 0, 255])
    joints_pcd_base.colors = np.array([0, 0, 0, 255])
    # Create and show scene
    scene = trimesh.Scene([mesh_ext, joints_pcd_ext])
    scene.show()

    # Export meshes to files
    if args.save_meshes:
        base_file_path = os.path.join(output_path, 'VAREN_base.ply') if output_path else 'VAREN_base.ply'
        full_file_path = os.path.join(output_path, 'VAREN_full.ply') if output_path else 'VAREN_full.ply'
    
        trimesh.exchange.export.export_mesh(mesh_base, base_file_path)
        trimesh.exchange.export.export_mesh(mesh_ext, full_file_path)
        print(f"Saved 'VAREN_base_base.ply' Meshes to /{output_path}")
        print(f"Saved 'VAREN_base_full.ply' Meshes to /{output_path}")


# add arguments for output folder path

if __name__ == "__main__":

    main()
