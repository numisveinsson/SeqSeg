import os
import pyvista as pv
import glob
import numpy as np
import matplotlib.pyplot as plt

def calculate_mesh_centroid(mesh):
    # Calculate the centroid of the mesh
    center = mesh.center_of_mass()
    return np.array(center)

def visualize_and_save_combined_image(parent_directory, output_file, spacing_x, spacing_z):
    # Get a list of all subdirectories in the parent directory
    folder_paths = [os.path.join(parent_directory, folder) for folder in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, folder))]

    # Sort folder paths alphabetically
    folder_paths.sort()

    folder_paths = [x for x in folder_paths if 'pred' in x and 'old' not in x]
    import pdb; pdb.set_trace()
    # Create a list to store meshes and centroids for each folder
    all_meshes = []
    all_centroids = []

    # Iterate over folders
    for folder_path in folder_paths:
        # Get the subfolder path 'postprocessed' in each folder
        postprocessed_path = os.path.join(folder_path, 'postprocessed')

        # Get a list of all .vtp files in the subfolder
        mesh_files = glob.glob(os.path.join(postprocessed_path, '*.vtp'))

        # Sort mesh files alphabetically
        mesh_files.sort()

        # Load the meshes and store them in a list
        meshes = [pv.read(mesh_file) for mesh_file in mesh_files]

        # Calculate the centroids of the meshes
        centroids = [calculate_mesh_centroid(mesh) for mesh in meshes]

        # Append meshes and centroids to the lists
        all_meshes.append(meshes)
        all_centroids.append(centroids)

    # Create a single plotter to display all the meshes
    plotter = pv.Plotter()

    # Loop over the folders, meshes, and centroids to add them to the plotter
    for i, (meshes, centroids) in enumerate(zip(all_meshes, all_centroids)):
        # Calculate the translation needed to bring the center of bounds to (0, 0, 0)
        bounds_centers = [mesh.GetBounds() for mesh in meshes]

        # Calculate the mean of the bounds centers
        mean_centers = np.mean(bounds_centers, axis=0)

        # Calculate the translation needed to bring the mean of bounds centers to (0, 0, 0)
        translation_center = -mean_centers

        # Translate all meshes to bring their bounds center to (0, 0, 0)
        for mesh in meshes:
            mesh.translate(translation_center)

        # Calculate the translation needed to space the columns evenly in x and avoid overlap in z
        max_extent_x = max([mesh.GetBounds()[1] - mesh.GetBounds()[0] for mesh in meshes]) * 1.5
        max_extent_z = max([mesh.GetBounds()[5] - mesh.GetBounds()[4] for mesh in meshes]) * 1.5

        # Calculate the desired position for the current row
        desired_position = np.array([i * spacing_x, 0, i * spacing_z])

        # Loop over the meshes in the current folder
        for mesh, centroid in zip(meshes, centroids):
            # Calculate the translation based on the difference between the desired and current positions
            translation = desired_position - centroid
            mesh.translate(translation)
            
            # Add meshes to the plotter
            plotter.add_mesh(mesh, color='blue')

    # Set up the plotter properties
    plotter.camera_position = 'xz'
    plotter.background_color = 'white'

    # Save the plotter view to an image file
    plotter.show(screenshot=output_file)

if __name__ == "__main__":
    # Specify the parent directory containing multiple folders
    parent_directory = '/Users/numisveins/Downloads/preds_new_aortas/'

    # Specify the output file for the combined image
    output_file = '/Users/numisveins/Downloads/preds_new_aortas/combined.png'

    # Get all subfolders
    subfolders = [f.path for f in os.scandir(parent_directory) if f.is_dir()]

    # Sort subfolders alphabetically
    subfolders.sort()

    subfolders = [x for x in subfolders if 'pred' in x and 'old' not in x]

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Loop through each subfolder
    for row, folder in enumerate(subfolders):
        # Find all mesh files in the 'postprocessed' folder
        mesh_files = glob.glob(os.path.join(folder, 'postprocessed', '*.vtp'))
        
        # Loop through each mesh file in the subfolder
        for col, mesh_file in enumerate(mesh_files):
            # Load the mesh using PyVista
            mesh = pv.read(mesh_file)

            # Change the origin of the coordinate system to 0, 0, 0
            mesh.origin = np.array([0, 0, 0])
            
            # Translate the mesh so that its center is on y=0
            translation = np.array([0, -mesh.center[1], 0])
            mesh.translate(translation)
            
            # Position the mesh in space based on row and column
            translation = np.array([col, 0, row]) * 20.0  # You may adjust the spacing factor
            
            # Add the mesh to the plotter with translation
            plotter.add_mesh(mesh.copy().translate(translation), color='blue')
    # Set up the plotter properties
    plotter.camera_position = 'xz'
    plotter.background_color = 'white'

    # Remove the axes
    plotter.show_axes = False

    # Show the plot
    plotter.show()