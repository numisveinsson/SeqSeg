import os
import glob
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid

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

    # Loop over the meshes and add them to the plotter with appropriate translations
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

def visualize_and_save_screenshot(mesh_file, output_folder, add_name):
    # Load the mesh
    mesh = pv.read(mesh_file)

    for position in ['xz', 'yz']:
        # Create a plotter
        plotter = pv.Plotter()

        # Add the mesh to the plotter
        plotter.add_mesh(mesh, color='red', reset_camera=True)

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

    main_dir = '/Users/numisveins/Downloads/preds_new_aortas/'
    main_dir = '/Users/numisveins/Documents/data_seqseg_paper/pred_aortas_june24_3/'
    # main_dir = '/Users/numisveins/Documents/data_seqseg_paper/pred_mic_aortas_june24/results/'
    # main_dir = '//Users/numisveins/Documents/MICCAI_Challenge23_Aorta_Tree_Data/'
    add_name = ''
    dir_list = os.listdir(main_dir)
    dir_list.sort()
    dir_list = [x for x in dir_list if 'pred' in x and 'old' not in x]

    for i in range(len(dir_list)):

        meshes_directory = main_dir + dir_list[i] + '/postprocessed/'

        # Specify the output folder for screenshots
        output_folder = meshes_directory+'/imgs/'

        # Ensure the output folder exists, create it if necessary
        os.makedirs(output_folder, exist_ok=True)

        # Loop over all .vtp files in the specified directory
        mesh_files = glob.glob(os.path.join(meshes_directory, '*.vtp'))

        # Specify the output file for the final image
        output_file = os.path.join(output_folder, 'screenshots.png')
        #visualize_and_save_combined_image(mesh_files, output_file)

        # Visualize the meshes and save screenshots
        for mesh_file in mesh_files:
            visualize_and_save_screenshot(mesh_file, output_folder, add_name)

        print("Screenshots saved successfully.")