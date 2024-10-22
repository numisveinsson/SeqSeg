import os
import glob
import pyvista as pv
import numpy as np
import sys
# add the folder above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import vtk_functions as vf


def calculate_mesh_centroid(mesh):
    # Calculate the centroid of the mesh
    center = mesh.center_of_mass()
    return np.array(center)


def visualize_and_save_combined_image(mesh_files, output_file):
    # Load the meshes and store them in a list
    meshes = [pv.read(mesh_file) for mesh_file in mesh_files]

    # Calculate the centroids of the meshes
    centroids = [calculate_mesh_centroid(mesh) for mesh in meshes]

    # Find the maximum extent along the x-axis for proper spacing
    max_extent = max([mesh.GetBounds()[1] for mesh in meshes])
    max_extent *= 1.5  # Adjust the spacing factor as needed

    # Create a single plotter to display all the meshes
    plotter = pv.Plotter()

    # Loop over the meshes and add them to the plotter
    # with appropriate translations
    for i, (mesh, centroid) in enumerate(zip(meshes, centroids)):
        # Calculate the translation needed to space the meshes evenly
        translation = [i * max_extent, 0, 0]
        translation -= centroid  # Adjust to the centroid of the mesh
        mesh.translate(translation)
        plotter.add_mesh(mesh, color='blue')

    # Set up the plotter properties
    plotter.camera_position = 'xy'
    plotter.background_color = 'white'

    # Save the plotter view to an image file
    plotter.show(screenshot=output_file)


def visualize_and_save_screenshot(mesh_file, output_folder, add_name,
                                  check_region=False, cmap_name='coolwarm'):
    # Load the mesh
    mesh = pv.read(mesh_file)

    for position in ['xz', 'yz']:
        # Create a plotter
        plotter = pv.Plotter()

        if check_region:
            mesh_vtp = vf.read_geo(mesh_file).GetOutput()
            arrays = vf.collect_arrays(mesh_vtp.GetCellData())
            # if arrays empty, try point data
            if not arrays:
                arrays = vf.collect_arrays(mesh_vtp.GetPointData())
            if 'RegionId' in arrays.keys():
                region = arrays['RegionId']
            elif 'Scalars_' in arrays.keys():
                region = arrays['Scalars_']
            else:
                region = None

            if region is not None:
                cmap_name = 'coolwarm'
                min_value = 0
                max_value = 9
                if region.max() <= 1:
                    min_value = -6
                    max_value = 3
                # Add the mesh with scalar coloring
                plotter.add_mesh(mesh, scalars=region,
                                 cmap=cmap_name,
                                 clim=[min_value, max_value],
                                 reset_camera=True,
                                 show_scalar_bar=False)
            else:
                # check names
                names = ['174', '176', '188']
                for name in names:
                    if name in mesh_file:
                        # flip the position
                        if position == 'xz':
                            # flip the mesh
                            # Flip the mesh along the x-axis by creating a transformation matrix
                            flip_x_matrix = np.array([
                                [-1,  0,  0,  0],  # Flip along the x-axis
                                [ 0,  1,  0,  0], 
                                [ 0,  0,  1,  0], 
                                [ 0,  0,  0,  1]
                            ])
                            mesh.transform(flip_x_matrix)
                        else:
                            # Flip the mesh along the y-axis by creating a transformation matrix
                            flip_y_matrix = np.array([
                                [ 1,  0,  0,  0], 
                                [ 0, -1,  0,  0],  # Flip along the y-axis
                                [ 0,  0,  1,  0], 
                                [ 0,  0,  0,  1]
                            ])
                            mesh.transform(flip_y_matrix)
                # Add the mesh without scalar coloring
                plotter.add_mesh(mesh, color='white', reset_camera=True)

        else:
            # Add the mesh to the plotter
            plotter.add_mesh(mesh, color='white', reset_camera=True)

        # Set up the camera and style (adjust as needed)
        plotter.camera_position = position

        # zoom in
        plotter.camera.zoom(0.9)
        # set background to white
        plotter.background_color = 'white'
        # set axes off
        plotter.show_axes = False
        # Save screenshot as SVG
        screenshot_file = os.path.join(output_folder, os.path.splitext(os.path.basename(mesh_file))[0] + '_' + add_name + '_' + position + '.png')
        plotter.show(screenshot=screenshot_file)

        # Close the plotter
        plotter.close()


if __name__ == "__main__":

    check_region = True

    main_dir = '/Users/numisveins/Downloads/preds_new_aortas/'
    main_dir = '/Users/numisveins/Documents/data_seqseg_paper/pred_aortas_june24_3/'
    main_dir = '/Users/numisveins/Documents/data_combo_paper/ct_data/vascular_segs/vascular_segs_mha/'
    main_dir = '/Users/numisveins/Documents/data_combo_paper/figures/ct/'
    # main_dir = '/Users/numisveins/Documents/data_seqseg_paper/pred_mic_aortas_june24/results/'
    # main_dir = '//Users/numisveins/Documents/MICCAI_Challenge23_Aorta_Tree_Data/'
    add_name = ''
    dir_list = os.listdir(main_dir)
    dir_list.sort()
    dir_list = [x for x in dir_list if not x.startswith('.')]
    # dir_list = [x for x in dir_list if 'pred' in x and 'old' not in x]

    for i in range(len(dir_list)):

        meshes_directory = main_dir + dir_list[i]  # + '/postprocessed/'

        # Specify the output folder for screenshots
        output_folder = meshes_directory+'/imgs/'

        # Ensure the output folder exists, create it if necessary
        os.makedirs(output_folder, exist_ok=True)

        # Loop over all .vtp files in the specified directory
        mesh_files = glob.glob(os.path.join(meshes_directory, '*.vtp'))

        # Specify the output file for the final image
        output_file = os.path.join(output_folder, 'render.png')
        # visualize_and_save_combined_image(mesh_files, output_file)

        # Visualize the meshes and save screenshots
        for mesh_file in mesh_files:
            visualize_and_save_screenshot(mesh_file, output_folder, add_name, check_region)

        print("Images saved successfully.")