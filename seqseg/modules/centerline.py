# TODO: Implement centerline calculation using pathfinding and exploration
import sys
import os
import time
import math
import vtk
import SimpleITK as sitk
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
# from numba import jit
sys.path.insert(0, './')
from seqseg.modules.sitk_functions import create_new, distance_map_from_seg
from seqseg.modules.vtk_functions import (calc_caps, evaluate_surface, 
                                   points2polydata, write_geo, appendPolyData)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def calculate_centerline(segmentation, caps, initial_radius=1):
    """
    Function to calculate the centerline of a segmentation
    using pathfinding and exploration to connect caps of the surface.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    caps : list of np.array
        List of np.array with the coordinates of the caps of the surface mesh.
        Assume first cap is the inlet and the rest are outlets.

    Returns
    -------
    centerline : vtkPolyData
    """

    # Initialize centerline
    source = caps[0]
    targets = caps[1:]

    # Calculate centerline by connecting caps
    path = pathfinding(segmentation, source, targets, initial_radius)

    return path


def set_new(segmentation):
    """
    Function to set all values of a segmentation to 0.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation to set to 0.

    Returns
    -------
    explored : sitk image
    """
    explored = create_new(segmentation)

    return explored


def add_explored(segmentation, explored, point, radius):
    """
    Function to add space around a point to the explored segmentation.

    We define a sphere of radius around the point and set all values
    within that sphere to 1. We treat this as a mask of explored space.

    We then find intersections between the segmentation and new sphere
    and add those to the explored segmentation.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    explored : sitk image
        Explored segmentation.
    point : np.array
        Coordinates of the point to add.
    radius : float
        Radius of the point to add.

    Returns
    -------
    explored : sitk image
        Updated explored segmentation.
    num_voxels : int
        Number of voxels added.
    """
    # Create sphere around point
    sphere = create_new(segmentation)  # Create new segmentation sitk image
    sphere = sphere * 0  # Set all values to 0

    # Get locations of all non-zero values in segmentation,
    # N x 3 array of indices
    locations = np.argwhere(sitk.GetArrayFromImage(
                            segmentation).transpose(2, 1, 0) > 0)
    # Use filter to get the physical coordinates of the non-zero values
    # N x 3 array of physical coordinates
    physical_locations = np.array([np.array(
                                  segmentation.TransformIndexToPhysicalPoint(
                                      location.tolist()))
                                  for location in locations])

    # Set values within sphere to 1 by calculating distance to point
    distances = np.linalg.norm(physical_locations - point, axis=1)
    # Points within sphere
    points_in_sphere = physical_locations[distances < radius]
    # Transform physical points to index
    indices = [segmentation.TransformPhysicalPointToIndex(point)
               for point in points_in_sphere]
    # Set values within sphere to 1
    for index in indices:
        sphere[index] = 1
    # for location in physical_locations:
    #     if np.linalg.norm(location - point) < radius:
    #         index = segmentation.TransformPhysicalPointToIndex(location)
    #         sphere[index] = 1

    # Find intersection between sphere and segmentation
    intersection_filter = sitk.AndImageFilter()
    intersection = intersection_filter.Execute(segmentation, sphere)
    num_voxels = sitk.GetArrayFromImage(intersection).sum()

    # Add intersection to explored
    explored = explored + intersection

    return explored, num_voxels


def find_first_branch(segmentation, distance_map, explored,
                      current, current_radius, targets):
    """
    Function to find the first branch of the path.

    We take steps based on the highest reward for each step
    until we reach a target.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    explored : sitk image
        Explored segmentation.
    current : np.array
        Coordinates of the current point.
    current_radius : float
        Radius of the current point.
    targets : list of np.array
        List of target points.

    Returns
    -------
    points : list of np.array
        Updated list of points.
    explored : sitk image
        Updated explored segmentation.
    targets : list of np.array
        Updated list of target points.
    """

    points = [current]
    while not reached_targets(current, targets, current_radius):
        # Get candidates for next point
        candidates = get_candidates(current, current_radius)
        # Remove candidates that are already explored
        candidates = [candidate for candidate in candidates
                      if sitk.GetArrayFromImage(explored).transpose(2, 1, 0)
                      [segmentation.TransformPhysicalPointToIndex(candidate)]
                      == 0]
        # Get rewards for each candidate
        # rewards = get_rewards_volume(segmentation, explored,
        #           candidates, current_radius)
        rewards = get_rewards_distance(distance_map, candidates)
        # Select candidate with highest reward
        next_index = np.argmax(rewards)
        next_point = candidates[next_index]
        # Add next point to path
        points.append(next_point)
        # Update current point
        current = next_point
        # Add next point to explored
        explored, _ = add_explored(segmentation, explored,
                                   current, current_radius)

    # Remove target from list and add the target as the last point
    target_index = np.argwhere([reached_target(current, target, current_radius)
                                for target in targets])[0]
    target = targets.pop(target_index.item())
    points.append(target)

    return points, explored, targets


def pathfinding(segmentation, source, targets, radius):
    """
    Function to calculate the path between two points on a surface mesh.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    source : np.array
        Coordinates of the source point.
    target : list of np.array
        Coordinates of the target points.
    radius : float
        Radius of the source point.

    Returns
    -------
    path : vtkPolyData
    """
    # Create distance map
    distance_map = distance_map_from_seg(segmentation)

    # Initialize explored segmentation
    explored = set_new(segmentation)
    explored, _ = add_explored(segmentation, explored, source, radius)
    sitk.WriteImage(explored,
                    '/Users/numisveins/Downloads/debug_centerline/explored.mha'
                    )

    # Initialize vtk polydata for path (lines) and points
    path = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    path.SetPoints(points)
    lines = vtk.vtkCellArray()
    path.SetLines(lines)
    radii = vtk.vtkDoubleArray()
    radii.SetName("Radius")

    # Initialize path
    current = source         # Current point
    current_radius = radius  # Current radius
    points.InsertNextPoint(current)
    lines.InsertNextCell(1)
    lines.InsertCellPoint(0)
    radii.InsertNextValue(current_radius)

    # Explore first branch, first is treated as inlet to outlet
    points_list, explored, targets = find_first_branch(segmentation,
                                                       distance_map,
                                                       explored,
                                                       current,
                                                       current_radius,
                                                       targets)
    from seqseg.modules.vtk_functions import points2polydata, write_geo
    polydata_point = points2polydata(points_list)
    pfn = os.path.join('/Users/numisveins/Downloads/debug_centerline/',
                       'first_branch.vtp')
    write_geo(pfn, polydata_point)
    for point in points_list[1:]:
        points.InsertNextPoint(point)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(points.GetNumberOfPoints()-2)
        lines.InsertCellPoint(points.GetNumberOfPoints()-1)
        radii.InsertNextValue(current_radius)
        current = point

    # Explore the rest of the branches
    while len(targets) > 0:
        # Get candidates for next point
        candidates = get_candidates_branches(points_list, current_radius)
        # Remove candidates that are already explored
        candidates = [candidate for candidate in candidates
                      if sitk.GetArrayFromImage(explored).transpose(2, 1, 0)
                      [segmentation.TransformPhysicalPointToIndex(candidate)]
                      == 0]
        # Get rewards for each candidate
        # rewards = get_rewards_volume(segmentation, explored,
        #                              candidates, current_radius)
        rewards = get_rewards_distance(distance_map, candidates)
        # Select candidate with highest reward
        next_index = np.argmax(rewards)
        next_point = candidates[next_index]
        # Add next point to path
        points.InsertNextPoint(next_point)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(points.GetNumberOfPoints()-2)
        lines.InsertCellPoint(points.GetNumberOfPoints()-1)
        radii.InsertNextValue(current_radius)
        # Update current point
        current = next_point
        # Add next point to explored

    return path


def reached_target(current, target, radius):
    """
    Function to check if the current point is close enough to the target point.

    Parameters
    ----------
    current : np.array
        Coordinates of the current point.
    target : np.array
        Coordinates of the target point.
    radius : float
        Radius of the target point.

    Returns
    -------
    reached : bool
    """
    reached = np.linalg.norm(current - target) < radius
    return reached


def reached_targets(current, targets, radius):
    """
    Function to check if the current point is close enough
    to any of the target points.

    Parameters
    ----------
    current : np.array
        Coordinates of the current point.
    targets : list of np.array
        Coordinates of the target points.
    radius : float
        Radius of the target points.

    Returns
    -------
    reached : bool
    """
    reached = False
    for target in targets:
        if reached_target(current, target, radius):
            reached = True
            break
    return reached


def get_candidates(current, current_radius, degrees=30):
    """
    Function to get the candidates for the next point to move to.
    We calculate vectors departing from the current point in all directions
    with degrees of separation.

    This is in 3D space, so the vectors are on the unit sphere.

    Example:
    current = (0, 0, 0)
    current_radius = 1
    degrees = 90
    This will return 6 candidates:
    [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    Parameters
    ----------
    current : np.array
        Coordinates of the current point.
    current_radius : float
        Radius of the current point.

    Returns
    -------
    candidates : list of np.array
    """
    candidates = []
    for i in range(0, 360, degrees):
        for j in range(0, 360, degrees):
            x = current_radius * np.sin(np.radians(i)) * np.cos(np.radians(j))
            y = current_radius * np.sin(np.radians(i)) * np.sin(np.radians(j))
            z = current_radius * np.cos(np.radians(i))
            candidates.append(np.array([x, y, z]) + current)
    return candidates


def get_candidates_branches(points_list, current_radius, degrees=30):
    """
    Function to get the candidates for the next point to move to.
    Here we already have a list of points and we want to explore the branches
    diverging from these points.

    For each point we calculate the tangent between the current point
    and the next point
    and calculate the candidates in the plane perpendicular to the tangent.

    The candidates are x number of degrees apart.

    Parameters
    ----------
    points_list : list of np.array
        List of points to explore from.
    current_radius : float
        Radius of the current point.
    degrees : int
        Degrees of separation between candidates.

    Returns
    -------
    candidates : list of np.array
    """
    # Multiply radius by a factor to go beyond the current branch
    current_radius *= 1.5

    candidates = []
    for i in range(1, len(points_list)):
        tangent = points_list[i] - points_list[i-1]
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([1, 0, 0])
        if np.dot(tangent, normal) > 0.9:
            normal = np.array([0, 1, 0])
        binormal = np.cross(tangent, normal)
        normal = np.cross(binormal, tangent)
        for j in range(0, 360, degrees):
            x = current_radius * np.cos(np.radians(j)) * normal
            y = current_radius * np.sin(np.radians(j)) * binormal
            candidates.append(x + y + points_list[i])
    return candidates


def get_rewards_volume(segmentation, explored, candidates, current_radius):
    """
    Function to get the rewards for each candidate point.
    The reward is calculated as the new volume exposed by the candidate point.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    candidates : list of np.array
        List of candidate points.

    Returns
    -------
    rewards : list of float
    """
    rewards = []
    for candidate in candidates:
        _, num_voxels = add_explored(segmentation, explored,
                                     candidate, current_radius)
        rewards.append(num_voxels)

    return rewards


def get_rewards_distance(distance_map, candidates):
    """
    Function to get the rewards for each candidate point.
    The reward is calculated as the distance value of the candidate point.
    Lower distance values are better so return the negative distance value.

    Parameters
    ----------
    distance_map : sitk image
        Distance map of the vessel.
    candidates : list of np.array
        List of candidate points.

    Returns
    -------
    rewards : list of float
    """
    rewards = []
    for candidate in candidates:
        index = distance_map.TransformPhysicalPointToIndex(candidate)
        rewards.append(-distance_map[index])

    return rewards


def initialize_seeds_random(segmentation, num_points=10):
    """
    Function to initialize seed points randomly inside the segmentation.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.

    Returns
    -------
    seeds : list of np.array
    """
    # Get physical size of the segmentation
    size = segmentation.GetSize()
    print(f"Size: {size}")
    bounds_min = np.array(segmentation
                          .TransformIndexToPhysicalPoint([0, 0, 0]))
    bounds_max = np.array(segmentation
                          .TransformIndexToPhysicalPoint(size))

    # Initialize seed points randomly
    seeds = []
    while len(seeds) < num_points:
        point = np.random.uniform(bounds_min, bounds_max).tolist()
        index = segmentation.TransformPhysicalPointToIndex(point)
        # print(f"Index: {index}")
        if (index[0] > 0 and index[1] > 0
           and index[2] > 0 and index[0] < size[0]
           and index[1] < size[1] and index[2] < size[2]):
            if segmentation[index] > 0:
                seeds.append(np.array(point))

    return seeds


def initialize_seeds_pixels(segmentation, every=1, value=0, negative=False):
    """
    Function to initialize seed points at the center of
    the pixels of the segmentation.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.

    Returns
    -------
    seeds : list of np.array
    """

    # Initialize seed points at the center of the pixels
    # Get locations of all non-zero values in segmentation
    if not negative:
        # N x 3 array of indices
        locations = np.argwhere(sitk.GetArrayFromImage(segmentation)
                                .transpose(2, 1, 0) > value)
    else:
        # N x 3 array of indices
        locations = np.argwhere(sitk.GetArrayFromImage(segmentation)
                                .transpose(2, 1, 0) < value)
    # Use filter to get the physical coordinates of the non-zero values
    physical_locations = np.array([np.array(
        segmentation.TransformIndexToPhysicalPoint(location.tolist()))
        for location in locations])  # N x 3 array of physical coordinates

    # keep every nth point
    physical_locations = physical_locations[::every]

    return physical_locations


def frangi_filter(input_image):
    """
    Function to apply the Frangi vesselness filter to an image.

    Parameters
    ----------
    input_image : sitk image
        Image to apply the filter to.

    Returns
    -------
    result : sitk image
    """
    # Set the parameters for the objectness measure
    sigmas = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 40]

    # Set the parameters for the objectness measure
    # according to specific input,
    # for example the vessels in the fundus image are dark,
    # so brightObject=False
    images = [
        sitk.ObjectnessMeasure(
            sitk.SmoothingRecursiveGaussian(input_image, s),
            alpha=0.5,
            beta=0.5,
            gamma=5.0,
            scaleObjectnessMeasure=True,  # changed from default value
            objectDimension=1,
            brightObject=True,  # changed from default value
        )
        for s in sigmas
    ]
    # # change data type to float32
    # images = [sitk.Cast(image, sitk.sitkFloat32) for image in images]

    # return the image with the max objectness value
    # result = sitk.Compose(images)

    return images[0]


def gradient_matrix(x):
    """
    Calculate the gradient matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.ndim,) + x.shape
       where the array[i, ...] corresponds to the gradient of
       x along the ith dimension.
    """
    gradients = np.gradient(x)
    gradients = np.array(gradients)[:, :, :]
    return gradients


def hessian_matrix(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for lim, grad_kl in enumerate(tmp_grad):
            hessian[k, lim, :, :] = grad_kl
    # make x_grad same shape as hessian
    x_grad = np.array(x_grad)[:, :, :]
    return hessian, x_grad


def calculate_principal_direction(matrix):
    """
    Function to calculate the principal direction of a matrix.

    Parameters
    ----------
    matrix : np.array
        Matrix to calculate the principal direction of.

    Returns
    -------
    principal_direction : np.array
    """
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # print(f"Eigenvalues: {eigenvalues}")
    # print(f"Condition number: {np.linalg.cond(matrix)}")
    # Get index of maximum eigenvalue
    max_index = np.argmin(eigenvalues)
    # print(f"Max Eig: {eigenvalues[max_index]:.4f},
    # Min Eig: {eigenvalues.min():.4f}")
    # Get principal direction
    principal_direction = (eigenvectors[max_index] /
                           np.linalg.norm(eigenvectors[max_index]))

    return principal_direction


def calc_inner_gradient_principal_direction(gradient_vector, hessian_matrix):
    """
    Function to calculate the inner gradient of the principal direction.

    Parameters
    ----------
    gradient_vector : np.array
        Gradient vector.
    hessian_matrix : np.array
        Hessian matrix.

    Returns
    -------
    inner_gradient : np.array
    """
    eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)
    # Normalize eigenvectors
    for i in range(len(eigenvalues)):
        eigenvectors[i] = eigenvectors[i] / np.linalg.norm(eigenvectors[i])

    inner = np.dot(gradient_vector/np.linalg.norm(gradient_vector),
                   eigenvectors)

    return inner, eigenvalues


def take_steps_gradient(distance_map_np, seeds, seg_img, tol=1e-2,
                        max_iter=100, step_size=0.5):
    """
    Function to take steps in the direction of the gradient at the seed points.
    We calculate the principal direction of the gradient and move the
    seed points in that direction. This is repeated until convergence.

    We use the Hessian to calculate the curvature of the distance map.
    Then we take steps in the direction where the curvature is the highest.
    This is done by calculating the principal direction of the Hessian.

    Parameters
    ----------
    distance_map_np : np.array
        Distance map of the vessel.
    seeds : list of np.array
        Seed points.
    seg_img : sitk image
        Segmentation of the vessel.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    final_seeds : list of np.array
    """
    use_hessian = False

    # Get spacings and origin
    spacings = np.array(seg_img.GetSpacing())
    print(f"Spacings: {spacings}")

    # Calculate bounds of the distance map
    bounds_min = seg_img.TransformIndexToPhysicalPoint([0, 0, 0])
    bounds_max = seg_img.TransformIndexToPhysicalPoint(seg_img.GetSize())

    # Initialize seed points
    final_seeds = []
    current_seeds = seeds
    from seqseg.modules.vtk_functions import points2polydata, write_geo
    polydata_point = points2polydata(current_seeds)
    pfn = os.path.join('/Users/numisveins/Downloads/debug_centerline/',
                       'initial_seeds.vtp')
    write_geo(pfn, polydata_point)

    # Calculate hessian of distance map
    hessian, gradient = hessian_matrix(distance_map_np)

    # Get hessian around seed points
    hessian_seeds = []
    gradient_seeds = []
    for seed in current_seeds:
        index = np.array(seg_img
                         .TransformPhysicalPointToIndex(seed)).astype(int)
        hessian_seeds.append(hessian[:, :, index[0], index[1], index[2]])
        gradient_seeds.append(gradient[:, index[0], index[1], index[2]])

    # Calculate principal direction of hessian
    principal_directions = [calculate_principal_direction(hessian_seed)
                            for hessian_seed in hessian_seeds]

    # Take steps in the direction of the principal direction
    for i in range(max_iter):
        new_seeds = []
        for j, seed in enumerate(current_seeds):
            inners, eigs = calc_inner_gradient_principal_direction(
                gradient_seeds[j], hessian_seeds[j])
            # print(f"Inner product: {inners}")
            # print(f"Eigenvalues: {eigs}")
            # print(f"""{i} Grad norm magnitude:
            #       {np.linalg.norm(gradient_seeds[j]):.4f}""")
            # print(f""" Inner product: {np.dot(principal_directions[j],
            #                                   gradient_seeds[j]/np.linalg.norm(
            #                                       gradient_seeds[j])):.4f}""")
            if use_hessian:  # Use hessian
                new_seed = seed + step_size * principal_directions[j]
            else:  # Use gradient
                new_seed = seed - step_size * gradient_seeds[j]

            if (new_seed[0] < bounds_min[0] or new_seed[1] < bounds_min[1]
               or new_seed[2] < bounds_min[2] or new_seed[0] > bounds_max[0]
               or new_seed[1] > bounds_max[1] or new_seed[2] > bounds_max[2]):
                continue
            # End if gradient is close to 0
            # print(f"Gradient norm: {np.linalg.norm(gradient_seeds[j])}")
            if np.linalg.norm(gradient_seeds[j]) < 0.02:
                final_seeds.append(new_seed)
                continue

            # End if new seed is close to old seed
            if np.linalg.norm(new_seed - seed) < tol:
                final_seeds.append(new_seed)
                continue
            new_seeds.append(new_seed)

        current_seeds = new_seeds
        polydata_point = points2polydata(current_seeds)
        pfn = os.path.join('/Users/numisveins/Downloads/debug_centerline/',
                           f'seeds_{i}.vtp')
        write_geo(pfn, polydata_point)

        # Update hessian around seed points
        hessian_seeds = []
        gradient_seeds = []
        for seed in current_seeds:
            index = np.array(seg_img
                             .TransformPhysicalPointToIndex(seed)).astype(int)
            if index[0] < 0:
                index[0] = 0
            if index[1] < 0:
                index[1] = 0
            if index[2] < 0:
                index[2] = 0
            if index[0] >= seg_img.GetSize()[0]:
                index[0] = seg_img.GetSize()[0] - 1
            if index[1] >= seg_img.GetSize()[1]:
                index[1] = seg_img.GetSize()[1] - 1
            if index[2] >= seg_img.GetSize()[2]:
                index[2] = seg_img.GetSize()[2] - 1

            hessian_seeds.append(hessian[:, :, index[0], index[1], index[2]])
            gradient_seeds.append(gradient[:, index[0], index[1], index[2]])

        # Calculate principal direction of hessian
        principal_directions = [calculate_principal_direction(hessian_seed)
                                for hessian_seed in hessian_seeds]

    final_seeds += current_seeds

    return final_seeds


def calculate_centerline_gradient(segmentation):
    """
    Function to calculate the centerline of a segmentation
    using gradient descent. We do this by:

    1. Creating a distance map from the segmentation.
    2. Initialize seed points randomly inside the segmentation.
    3. Calculate the gradient of the distance map at the seed points.
    4. Calculate the principal direction of the gradient.
    5. Move the seed points in the principal direction.
    6. Repeat steps 3 to 5 until convergence.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.

    Returns
    -------
    centerline : vtkPolyData
    """

    # Create distance map
    distance_map = distance_map_from_seg(segmentation)
    distance_map_np = sitk.GetArrayFromImage(distance_map).transpose(2, 1, 0)

    # Initialize seed points
    # seeds = initialize_seeds_random(segmentation, num_points=100)
    # seeds = initialize_seeds_pixels(segmentation, every=100)
    seeds = initialize_seeds_pixels(distance_map, every=50, value=-0.01,
                                    negative=True)

    # Calculate final location of seed points
    final_seeds = take_steps_gradient(distance_map_np, seeds, segmentation,
                                      tol=1e-3, max_iter=200, step_size=0.5)

    # Create vtk polydata for points
    points = vtk.vtkPoints()
    for seed in final_seeds:
        points.InsertNextPoint(seed)

    # Create vtk polydata for lines
    from seqseg.modules.vtk_functions import points2polydata
    centerline = points2polydata(final_seeds)

    return centerline


def fast_marching_method_seg_dist(segmentation, distance_map, point):
    """
    Function to calculate the fast marching method from a seed point.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    point : np.array
        Seed point.

    Returns
    -------
    output : sitk image of distances computed from the seed point.
    """
    # Index of seed point
    index = segmentation.TransformPhysicalPointToIndex(point.tolist())
    index = [94, 124, 3]
    print(f"""Max: {sitk.GetArrayFromImage(segmentation).max()},
          Min: {sitk.GetArrayFromImage(segmentation).min()}""")
    # Invert the distance map so positive values are inside the vessel
    distance_map = distance_map * -1
    print(f"""Max: {sitk.GetArrayFromImage(distance_map).max()},
          Min: {sitk.GetArrayFromImage(distance_map).min()}""")
    # Now make all values that are 0 in the segmentation to 0
    # in the distance map
    # distance_map = sitk.Mask(distance_map, segmentation)
    print(f"""Max: {sitk.GetArrayFromImage(distance_map).max()},
          Min: {sitk.GetArrayFromImage(distance_map).min()}""")
    # Now rescale to 10-255
    distance_map = sitk.RescaleIntensity(distance_map, 1, 255)
    print(f"""Max: {sitk.GetArrayFromImage(distance_map).max()},
          Min: {sitk.GetArrayFromImage(distance_map).min()}""")
    # Create fast marching filter
    fast_marching = sitk.FastMarchingImageFilter()
    fast_marching.AddTrialPoint((index[0], index[1], index[2]))
    # fast_marching.SetStoppingValue(1)
    path = fast_marching.Execute(distance_map)
    # Keep only segmentation values
    path = sitk.Mask(path, segmentation)
    # rescale segmentation values to 1-255
    new_segmentation = sitk.Cast(segmentation, sitk.sitkFloat32)
    new_segmentation = new_segmentation * distance_map
    sitk.WriteImage(new_segmentation,
                    '/Users/numisveins/Downloads/debug_centerline'
                    + '/new_segmentation_0.mha')
    new_segmentation = sitk.RescaleIntensity(segmentation, 0.1, 1)
    sitk.WriteImage(new_segmentation,
                    '/Users/numisveins/Downloads/debug_centerline'
                    + '/new_segmentation.mha')
    print(f"""Max: {sitk.GetArrayFromImage(new_segmentation).max()},
          Min: {sitk.GetArrayFromImage(new_segmentation).min()}""")
    # Create fast marching filter
    out = fast_marching.Execute(new_segmentation)
    out = sitk.Mask(path, segmentation)

    return out


def fast_marching_method(speed_image, point, stopping_value=1000):
    """
    Function to calculate the fast marching method from a seed point.

    Parameters
    ----------
    speed_image : sitk image
        Speed image.
    point : list of int
        Seed point.

    Returns
    -------
    output : sitk image of distances computed from the seed point.
    """
    # Create fast marching filter
    fast_marching = sitk.FastMarchingImageFilter()
    fast_marching.AddTrialPoint(point)
    fast_marching.SetStoppingValue(stopping_value)
    output = fast_marching.Execute(speed_image)

    return output


def upwind_fast_marching_method(speed_image, point):
    """
    Function to calculate the upwind fast marching method from a seed point.

    Parameters
    ----------
    speed_image : sitk image
        Speed image.
    point : np.array
        Seed point.

    Returns
    -------
    out: sitk image of gradient computed from the seed point.
    """
    # Index of seed point
    index = speed_image.TransformPhysicalPointToIndex(point.tolist())
    index = [94, 124, 3]
    # Create fast marching filter
    fast_marching = sitk.FastMarchingUpwindGradientImageFilter()
    fast_marching.AddTrialPoint((index[0], index[1], index[2]))
    out = fast_marching.Execute(speed_image)

    return out


def colliding_fronts(segmentation, point1, point2):
    """
    Function to calculate the colliding fronts method from a seed point.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    point : np.array
        Seed point.

    Returns
    -------
    path : vtkPolyData
    """
    # Index of seed point
    index1 = segmentation.TransformPhysicalPointToIndex(point1.tolist())
    index2 = segmentation.TransformPhysicalPointToIndex(point2.tolist())
    # Create fast marching filter
    fast_marching = sitk.CollidingFrontsImageFilter()
    fast_marching.AddSeedPoint1((index1[0], index1[1], index1[2]))
    fast_marching.AddSeedPoint2((index2[0], index2[1], index2[2]))
    path = fast_marching.Execute(segmentation)

    return path


def interpolate_gradient(gradient, current, seg_img):
    """
    Function to interpolate the gradient at a point.

    Parameters
    ----------
    gradient : np.array
        Gradient of the distance map.
    current : np.array
        Current point.
    seg_img : sitk image
        Segmentation of the vessel.

    Returns
    -------
    gradient_current : np.array
    """
    # Get index of current point
    current_index = seg_img.TransformPhysicalPointToIndex(current.tolist())
    # Get fractional part of current point
    frac = current - np.array(seg_img
                              .TransformIndexToPhysicalPoint(current_index))
    # Get gradient at current point
    gradient_current = gradient[:, current_index[0],
                                current_index[1],
                                current_index[2]]
    # Get gradient at neighboring points
    gradient_x1 = gradient[:, current_index[0]+1,
                           current_index[1],
                           current_index[2]]
    gradient_y1 = gradient[:, current_index[0],
                           current_index[1]+1,
                           current_index[2]]
    gradient_z1 = gradient[:, current_index[0],
                           current_index[1],
                           current_index[2]+1]
    gradient_x2 = gradient[:, current_index[0]+1,
                           current_index[1]+1,
                           current_index[2]]
    gradient_y2 = gradient[:, current_index[0],
                           current_index[1]+1,
                           current_index[2]+1]
    gradient_z2 = gradient[:, current_index[0]+1,
                           current_index[1],
                           current_index[2]+1]
    gradient_x3 = gradient[:, current_index[0]+1,
                           current_index[1]+1,
                           current_index[2]+1]
    # Interpolate gradient
    gradient_current = (1-frac[0]) * (1-frac[1]) * \
        (1-frac[2]) * gradient_current + \
        frac[0] * (1-frac[1]) * (1-frac[2]) * gradient_x1 + \
        (1-frac[0]) * frac[1] * (1-frac[2]) * gradient_y1 + \
        (1-frac[0]) * (1-frac[1]) * frac[2] * gradient_z1 + \
        frac[0] * frac[1] * (1-frac[2]) * gradient_x2 + \
        (1-frac[0]) * frac[1] * frac[2] * gradient_y2 + \
        frac[0] * (1-frac[1]) * frac[2] * gradient_z2 + \
        frac[0] * frac[1] * frac[2] * gradient_x3

    return gradient_current


def backtracking_gradient(gradient, distance_map_surf_np,
                          seg_img, seed, target, relax_factor=1,
                          verbose=False):
    """
    Function to backtrack from a target point to
    a seed point using the gradient.

    Parameters
    ----------
    gradient : np.array
        Gradient of the distance map.
    distance_map_surf_np : np.array
        Distance map of the vessel, from boundaries to center.
    seg_img : sitk image
        Segmentation of the vessel.
    seed : np.array
        Seed point
    target : np.array
        Target point
    relax_factor : float
        Factor to relax the tolerance

    Returns
    -------
    points : list of np.array
        Points from target to seed.
    success : bool
        Boolean indicating if the backtracking was successful.
    """
    # print(f"   Backtracking from {target} to {seed}")
    max_number_points = 100000
    use_gradient_grid = True
    step_size = 0.01

    success = True

    # Index of seed and target points
    target_index = seg_img.TransformPhysicalPointToIndex(target.tolist())

    # Initialize points
    current = target
    current_index = target_index
    points = [current]
    # Initialize tolerance
    tol = distance_map_surf_np[target_index[0],
                               target_index[1],
                               target_index[2]]
    # print(f"Tolerance: {tol}")

    # Backtrack until we reach the seed point
    while (np.linalg.norm(current - seed) > tol*relax_factor
           and len(points) < max_number_points):
        # print(f"""Current: {current},
        #       Seed: {seed},
        #       Dist between: {np.linalg.norm(current - seed)}""")
        # print(f"Seg value at current: {seg_img[current_index]}")
        # Get gradient at current point
        if use_gradient_grid:
            gradient_current = gradient[:,
                                        current_index[0],
                                        current_index[1],
                                        current_index[2]]
        else:
            gradient_current = interpolate_gradient(gradient, current, seg_img)
        
        # Calculate gradient magnitude before normalization
        gradient_magnitude = np.linalg.norm(gradient_current)
        # if verbose:
        #     print(f"      Step {len(points)}: gradient magnitude = {gradient_magnitude:.6f}")
        
        # Normalize gradient
        gradient_current = gradient_current / gradient_magnitude
        # Move in the direction of the gradient
        current = current - step_size * gradient_current
        # Get index of current point
        current_index = seg_img.TransformPhysicalPointToIndex(current.tolist())
        # Add current point to path
        points.append(current)
        # If index is on the border, break
        if (current_index[0] < 0
           or current_index[1] < 0
           or current_index[2] < 0
           or current_index[0] >= seg_img.GetSize()[0]
           or current_index[1] >= seg_img.GetSize()[1]
           or current_index[2] >= seg_img.GetSize()[2]):
            print("   Fail: Reached border, breaking")
            success = False
            break
        # Update tolerance
        tol = distance_map_surf_np[current_index[0],
                                   current_index[1],
                                   current_index[2]]

    if len(points) == max_number_points:
        print("   Fail: Reached max number of points")
        success = False
    elif success and verbose:
        print(f"   Success: {len(points)} points")

    # Add seed point to path
    points.append(seed)

    return points, success


def calc_centerline_fmm(segmentation, seed=None, targets=None,
                        min_res=300, out_dir=None, write_files=False,
                        move_target_if_fail=False,
                        relax_factor=1, return_target=False,
                        return_target_all=False,
                        verbose=False, return_failed=False,
                        return_success_list=False,
                        post_process_kwargs=None,
                        ):
    """
    Function to calculate the centerline of a segmentation
    using the fast marching method. The method goes as follows:

    1. Create a distance map from the segmentation.
    2. Define seed points as indices.
    3. Calculate distance map using the fast marching method.
    4. Calculate path from seed to target using the distance map
       by backtracking the gradient from the target to the seed.
    5. Repeat for all targets.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    seed : np.array/indices
        Seed point.
    targets : list of np.array/indices
        List of target points.
    post_process_kwargs : dict, optional
        Extra keyword arguments for :func:`post_process_centerline`, e.g.
        ``{'merge_method': 'tree', 'tree_merge_radius_factor': 1.2}``. Use
        ``tree_merge_decimate_max_points_per_branch=None`` to disable
        pre-merge arc-length decimation. Default remains vtkClean-based merge
        (``merge_method='clean'``).

    Returns
    -------
    centerline : vtkPolyData
        Centerline of the vessel.
    """
    output = None
    original_segmentation = segmentation
    was_resampled = False
    if verbose:
        print(f"Resolution of segmentation: {segmentation.GetSize()}")

    # Resample if segmentation resolution is too low
    if segmentation.GetSize()[2] < min_res:
        print(f"    Resampling segmentation from size {segmentation.GetSize()} to size {min_res}")
        # Divide spacing so that the size is at least 50
        divide = segmentation.GetSize()[2] / min_res
        segmentation = sitk.Resample(segmentation,
                                     [int(x/divide) for x in segmentation
                                      .GetSize()],
                                     sitk.Transform(),
                                     sitk.sitkNearestNeighbor,
                                     segmentation.GetOrigin(),
                                     [x*divide for x in segmentation
                                      .GetSpacing()],
                                     segmentation.GetDirection(), 0,
                                     segmentation.GetPixelID())
        was_resampled = True
        print(f"    New size: {segmentation.GetSize()}")

    # Keep user-provided index coordinates anatomically consistent with
    # the resampled grid by mapping old-index -> physical -> new-index.
    if was_resampled:
        if seed is not None and not isinstance(seed, np.ndarray):
            seed_original = [int(i) for i in seed]
            seed_phys = original_segmentation.TransformIndexToPhysicalPoint(
                seed_original)
            seed = list(segmentation.TransformPhysicalPointToIndex(seed_phys))

        if (targets is not None
           and len(targets) > 0
           and not isinstance(targets[0], np.ndarray)):
            targets_resampled = []
            for target in targets:
                target_original = [int(i) for i in target]
                target_phys = original_segmentation.TransformIndexToPhysicalPoint(
                    target_original)
                target_resampled = list(
                    segmentation.TransformPhysicalPointToIndex(target_phys))
                targets_resampled.append(target_resampled)
            targets = targets_resampled
    # Create distance map
    distance_map_surf = distance_map_from_seg(segmentation)
    # Switch sign of distance map
    distance_map_surf = distance_map_surf * -1
    distance_map_surf_np = sitk.GetArrayFromImage(distance_map_surf).transpose(2, 1, 0)
    # Get maximum value of distance map
    max_surf = distance_map_surf_np.max()
    if verbose:
        print(f"Max of distance map: {max_surf}")

    # Write distance map to file
    if out_dir and write_files:
        sitk.WriteImage(distance_map_surf,
                        os.path.join(out_dir, 'distance_map_surf.mha'))

    # If seed and targets are not defined, use create using cluster map
    if seed is None and targets is None:
        print("Need to create seed and targets")
        seed, targets, output = cluster_map(segmentation,
                                            return_wave_distance_map=True,
                                            out_dir=out_dir,
                                            write_files=write_files,
                                            verbose=verbose)
    elif seed is None:
        print(f"Need to create seed, targets given: {targets}")
        index = np.where(distance_map_surf_np == max_surf)
        # have same format as targets
        if isinstance(targets[0], np.ndarray):
            # then physical point
            index = [int(index[0][0]), int(index[1][0]), int(index[2][0])]
            seed = segmentation.TransformIndexToPhysicalPoint(index)
            seed = np.array(seed)
        else:
            # else list of indices
            seed = [int(index[0][0]), int(index[1][0]), int(index[2][0])]
    elif targets is None:
        print(f"Need to create targets, seed given: {seed}")
        _, targets, output = cluster_map(segmentation,
                                         return_wave_distance_map=True,
                                         out_dir=out_dir,
                                         write_files=write_files,
                                         verbose=verbose)
        # have same format as seed
        if isinstance(seed, list):
            # then list of indices
            targets = [list(
                segmentation.TransformPhysicalPointToIndex(target.tolist()))
                for target in targets]
    if verbose:
        print(f"Seed: {seed}")
        print(f"Targets: {targets}")

    # if seed/targets is np.array, convert to index
    if isinstance(seed, np.ndarray):
        seed = list(segmentation.TransformPhysicalPointToIndex(seed.tolist()))
        # if any are outside or on boundary, move inside
        seed = check_border(seed, segmentation.GetSize())
        seed_np = np.array(segmentation.TransformIndexToPhysicalPoint(seed))
        targets_np = [target for target in targets]
        targets = [list(
            segmentation.TransformPhysicalPointToIndex(target.tolist()))
            for target in targets]
        for i, target in enumerate(targets):
            targets[i] = check_border(target, segmentation.GetSize())
    else:
        seed_np = segmentation.TransformIndexToPhysicalPoint(seed)
        targets_np = [segmentation.TransformIndexToPhysicalPoint(target)
                      for target in targets]

    if not output:
        if verbose:
            print("Still need to calculate fast marching method") 
        # Mask distance map with segmentation
        distance_map = sitk.Mask(distance_map_surf, segmentation)
        # print(f"Max: {sitk.GetArrayFromImage(distance_map).max()}")
        # print(f"Min: {sitk.GetArrayFromImage(distance_map).min()}")
        # Scale distance map to 1-255
        distance_map = sitk.RescaleIntensity(distance_map, 0.01, 1)
        # print(f"Max: {sitk.GetArrayFromImage(distance_map).max()}")
        # print(f"Min: {sitk.GetArrayFromImage(distance_map).min()}")

        # Calculate distance map using fast marching method
        if verbose:
            print("Starting fast marching method")
        output = fast_marching_method(distance_map, seed, stopping_value=1000)
        if verbose:
            print("Finished fast marching method")
    if verbose:
        print(f"Max of output: {sitk.GetArrayFromImage(output).max()}")
        print(f"Min: {sitk.GetArrayFromImage(output).min()}")
    # sitk.WriteImage(output,
    #                 '/Users/numisveins/Downloads/debug_centerline/output.mha')
    output_mask = sitk.Mask(output, segmentation)

    if verbose:
        print(f"Max of output mask: {sitk.GetArrayFromImage(output_mask).max()}")
        print(f"Min: {sitk.GetArrayFromImage(output_mask).min()}")
    if out_dir and write_files:
        sitk.WriteImage(output,
                        os.path.join(out_dir, 'out_fmm.mha'))
        sitk.WriteImage(output_mask,
                        os.path.join(out_dir, 'masked_out_fmm.mha'))
        sitk.WriteImage(segmentation,
                        os.path.join(out_dir, 'segmentation_from_fmm.mha'))
    # Get gradient of distance map
    gradient = gradient_matrix(
        sitk.GetArrayFromImage(output).transpose(2, 1, 0))
    if verbose:
        # Calculate and print gradient magnitude statistics
        gradient_magnitude = np.sqrt(np.sum(gradient**2, axis=0))
        print("Gradient calculated")
        print(f"    Gradient magnitude - Min: {gradient_magnitude.min():.6f}, Max: {gradient_magnitude.max():.6f}, Mean: {gradient_magnitude.mean():.6f}")

    points_list, success_list = [], []
    for t_num, target_np in enumerate(targets_np):
        if verbose:
            print(f"        Starting target {t_num+1}/{len(targets_np)}")
        # Calculate path from seed to target
        points, success = backtracking_gradient(gradient,
                                                distance_map_surf_np,
                                                segmentation,
                                                seed_np, target_np,
                                                relax_factor=relax_factor,
                                                verbose=verbose)
        if not success and move_target_if_fail:
            print("\n   Trying to move target inside segmentation")
            pot_point = calc_points_target(max_surf, distance_map_surf,
                                           target_np)

            if pot_point is not None:
                print(f"    New target: {pot_point}")
                points, success = backtracking_gradient(gradient,
                                                        distance_map_surf_np,
                                                        segmentation,
                                                        seed_np, pot_point,
                                                        relax_factor=relax_factor,
                                                        verbose=verbose)
        # Add points to list
        points_list.append(points)
        success_list.append(success)

    # Create vtk polydata for points
    centerline = create_centerline_polydata(points_list,
                                            success_list,
                                            distance_map_surf,
                                            return_failed=return_failed)
    _pp = {'verbose': verbose}
    _pp.update(post_process_kwargs or {})
    centerline = post_process_centerline(centerline, **_pp)

    print(f"    Centerline calculated, success ratio: {success_list.count(True)} / {len(success_list)}")

    # If success is all False, return False
    if not success_list.count(True):
        success_overall = False
    else:
        success_overall = True

    if return_target:
        # only return successful targets
        targets_np = [targets_np[i] for i in range(len(targets_np))
                      if success_list[i]]
        return centerline, success_overall, targets_np
    elif return_target_all:
        # return all targets, even if not successful
        return centerline, success_overall, targets_np
    elif return_success_list:
        # return success list
        return centerline, success_overall, success_list
    # else return only centerline and success overall
    else:
        return centerline, success_overall


def calc_points_target(max_surf, distance_map_surf, target_np,
                       N=10000):
    """
    Function to move target to new locations by a distance of max_surf / 2
    """
    # Calculate N random points around target
    points = []
    for i in range(N):
        point = target_np + np.random.normal(0, 1, 3) * max_surf
        points.append(point)

    # Find the point with highest distance value
    max_dist = 0
    max_point = None
    for point in points:
        index = distance_map_surf.TransformPhysicalPointToIndex(point.tolist())
        # continue if index is outside bounds
        if (index[0] < 0 or index[1] < 0 or index[2] < 0
              or index[0] >= distance_map_surf.GetSize()[0]
              or index[1] >= distance_map_surf.GetSize()[1]
              or index[2] >= distance_map_surf.GetSize()[2]):
            continue
        dist = distance_map_surf.GetPixel(index)
        if dist > max_dist:
            max_dist = dist
            max_point = point

    return max_point


def create_centerline_polydata(points_list, success_list, distance_map_surf,
                               return_failed=False):
    """
    Function to create a vtk polydata from a list of points.
    Each list of points is a path from seed to target and is stored as a line.
    Each list of points starts with target and ends with seed.
    All lists have the same seed point.


    Uses distance map to add radius to points.
    The distance value at each point is used as the radius.
    Radius is stored as array in the vtk polydata under name
    'MaximumInscribedSphereRadius'
    Global node id is stored as array in the vtk polydata under name
    'GlobalNodeID'
    Global node id is the index of the point in the list of points.

    Parameters
    ----------
    points_list : list of list of np.array
        List of points.
    distance_map_surf : sitk image
        Distance map of the vessel.

    Returns
    -------
    centerline : vtkPolyData
        Centerline of the vessel with radius.
    """
    # Create vtk polydata
    centerline = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    radii = vtk.vtkDoubleArray()
    radii.SetName("MaximumInscribedSphereRadius")
    global_node_id = vtk.vtkIntArray()
    global_node_id.SetName("GlobalNodeID")
    cent_id = vtk.vtkIntArray()
    cent_id.SetName("CenterlineId")
    min_radius = float(max(distance_map_surf.GetSpacing()))

    # Iterate over all points
    for ind, points_path in enumerate(points_list):
        # Remove any nan values
        points_path = [point for point in points_path
                       if not np.isnan(point).any()]
        # flip the list so that the seed is the first point
        points_path = points_path[::-1]
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(points_path))
        for i, point in enumerate(points_path):
            # Get index of point
            index = distance_map_surf.TransformPhysicalPointToIndex(
                point.tolist())
            # make sure index is within bounds
            index = [max(0, min(index[0], distance_map_surf.GetSize()[0]-1)),
                     max(0, min(index[1], distance_map_surf.GetSize()[1]-1)),
                     max(0, min(index[2], distance_map_surf.GetSize()[2]-1))]
            # Get distance value at point
            radius = distance_map_surf.GetPixel(index)
            radius = max(float(radius), min_radius)
            # Add point to points
            id = points.InsertNextPoint(point)
            line.GetPointIds().SetId(i, id)
            # Add radius to radii
            radii.InsertNextValue(radius)
            # Add global node id
            global_node_id.InsertNextValue(i)
            # Add centerline id
            cent_id.InsertNextValue(ind)

        # Add line to lines if was successful
        if success_list[ind]:
            lines.InsertNextCell(line)
        elif return_failed:
            # If return_failed is True, still add the line
            lines.InsertNextCell(line)
            print(f"Failed to reach target {ind+1}, still adding line with {len(points_path)} points.")

    # Add points, lines and radii to polydata
    centerline.SetPoints(points)
    centerline.SetLines(lines)
    centerline.GetPointData().AddArray(radii)
    centerline.GetPointData().AddArray(global_node_id)
    centerline.GetPointData().AddArray(cent_id)

    return centerline


def _polyline_arc_length(coords):
    """Sum of Euclidean segment lengths for Nx3 coordinates."""
    if len(coords) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def _dist_point_segment(p, a, b):
    """
    Closest point on segment a--b to p.
    Returns distance, closest point q, parameter t in [0,1] along a->b.
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-30:
        q = a.copy()
        return float(np.linalg.norm(p - q)), q, 0.0
    t = float(np.dot(p - a, ab) / denom)
    t_cl = min(1.0, max(0.0, t))
    q = a + t_cl * ab
    return float(np.linalg.norm(p - q)), q, t_cl


def _extract_polylines_from_polydata(centerline):
    """
    Return list of dicts: coords (N,3), radii (N,) or None, centerline_id (int).
    One entry per line/polyline cell with at least two points.
    """
    n_cells = centerline.GetNumberOfCells()
    radii_arr = centerline.GetPointData().GetArray(
        "MaximumInscribedSphereRadius")
    cid_arr = centerline.GetPointData().GetArray("CenterlineId")
    polylines = []
    for ci in range(n_cells):
        cell = centerline.GetCell(ci)
        ctype = cell.GetCellType()
        if ctype not in (vtk.VTK_LINE, vtk.VTK_POLY_LINE):
            continue
        np_cell = cell.GetNumberOfPoints()
        if np_cell < 2:
            continue
        ids = [cell.GetPointId(j) for j in range(np_cell)]
        coords = np.array([centerline.GetPoint(pid) for pid in ids])
        if radii_arr is not None:
            radii = np.array(
                [radii_arr.GetValue(pid) for pid in ids], dtype=float)
        else:
            radii = None
        if cid_arr is not None:
            cl_id = int(cid_arr.GetValue(ids[0]))
        else:
            cl_id = ci
        polylines.append({
            'coords': coords,
            'radii': radii,
            'centerline_id': cl_id,
        })
    return polylines


def _resample_polyline_by_arc_length(coords, max_points, radii=None):
    """
    Resample a polyline to at most ``max_points`` vertices spaced uniformly
    by arc length (endpoints preserved when n >= 2).

    Parameters
    ----------
    coords : (n, 3) array
    max_points : int
        Must be >= 2.
    radii : (n,) array or None
        Interpolated linearly along segments when provided.

    Returns
    -------
    coords_out, radii_out
        radii_out is None if radii was None.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2 or max_points < 2 or n <= max_points:
        out_r = None if radii is None else np.asarray(radii, dtype=float).copy()
        return coords.copy(), out_r

    seg_len = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cumul = np.zeros(n, dtype=float)
    cumul[1:] = np.cumsum(seg_len)
    total = float(cumul[-1])
    if total < 1e-30:
        c2 = np.vstack((coords[0], coords[-1]))
        if radii is None:
            return c2, None
        r = np.asarray(radii, dtype=float)
        return c2, np.array([r[0], r[-1]], dtype=float)

    targets = np.linspace(0.0, total, max_points)
    out_coords = np.zeros((max_points, 3), dtype=float)
    if radii is not None:
        rad = np.asarray(radii, dtype=float)
        out_rad = np.zeros(max_points, dtype=float)
    else:
        rad = None
        out_rad = None

    for j, d in enumerate(targets):
        idx = int(np.searchsorted(cumul, d, side='right') - 1)
        idx = max(0, min(idx, n - 2))
        denom = float(seg_len[idx]) + 1e-30
        t = (d - cumul[idx]) / denom
        t = min(1.0, max(0.0, t))
        out_coords[j] = (1.0 - t) * coords[idx] + t * coords[idx + 1]
        if rad is not None:
            out_rad[j] = (1.0 - t) * rad[idx] + t * rad[idx + 1]

    return out_coords, out_rad


def _decimate_centerline_polylines_before_tree_merge(polydata,
                                                     max_points_per_branch):
    """
    Per-cell arc-length decimation for VTK_LINE / VTK_POLY_LINE cells only.
    Other cell types are omitted from the output. Vertices are not merged
    across cells (safe before tree merge).

    Parameters
    ----------
    polydata : vtkPolyData
    max_points_per_branch : int or None
        If None or < 2, returns ``polydata`` unchanged.

    Returns
    -------
    vtkPolyData
    """
    if max_points_per_branch is None:
        return polydata
    max_points_per_branch = int(max_points_per_branch)
    if max_points_per_branch < 2:
        return polydata

    n_cells = polydata.GetNumberOfCells()
    if n_cells == 0:
        return polydata

    radii_arr = polydata.GetPointData().GetArray(
        "MaximumInscribedSphereRadius")
    cid_arr = polydata.GetPointData().GetArray("CenterlineId")
    gid_arr = polydata.GetPointData().GetArray("GlobalNodeID")

    out = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    out_rad = None
    if radii_arr is not None:
        out_rad = vtk.vtkDoubleArray()
        out_rad.SetName("MaximumInscribedSphereRadius")
    out_cid = vtk.vtkIntArray()
    out_cid.SetName("CenterlineId")
    out_gid = None
    if gid_arr is not None:
        out_gid = vtk.vtkIntArray()
        out_gid.SetName("GlobalNodeID")

    out_branch_idx = 0
    for ci in range(n_cells):
        cell = polydata.GetCell(ci)
        ctype = cell.GetCellType()
        if ctype not in (vtk.VTK_LINE, vtk.VTK_POLY_LINE):
            continue
        np_cell = cell.GetNumberOfPoints()
        if np_cell < 2:
            continue
        ids = [cell.GetPointId(j) for j in range(np_cell)]
        coords = np.array([polydata.GetPoint(pid) for pid in ids],
                          dtype=float)
        if radii_arr is not None:
            radii = np.array(
                [radii_arr.GetValue(pid) for pid in ids], dtype=float)
        else:
            radii = None

        c_dec, r_dec = _resample_polyline_by_arc_length(
            coords, max_points_per_branch, radii=radii)
        m = c_dec.shape[0]
        cl_id = (int(cid_arr.GetValue(ids[0])) if cid_arr is not None
                 else out_branch_idx)

        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(m)
        for k in range(m):
            pid = pts.InsertNextPoint(
                float(c_dec[k, 0]), float(c_dec[k, 1]), float(c_dec[k, 2]))
            pl.GetPointIds().SetId(k, pid)
            if out_rad is not None and r_dec is not None:
                out_rad.InsertNextValue(float(r_dec[k]))
            out_cid.InsertNextValue(cl_id)
            if out_gid is not None:
                out_gid.InsertNextValue(k)
        lines.InsertNextCell(pl)
        out_branch_idx += 1

    if lines.GetNumberOfCells() == 0:
        return polydata

    out.SetPoints(pts)
    out.SetLines(lines)
    if out_rad is not None:
        out.GetPointData().AddArray(out_rad)
    out.GetPointData().AddArray(out_cid)
    if out_gid is not None:
        out.GetPointData().AddArray(out_gid)
    return out


def _tree_merge_state_to_polydata(state_pts, state_radii, state_cid,
                                  state_gid, out_cells,
                                  state_bif_label=None):
    """Build vtkPolyData from merged tree representation."""
    out = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for p in state_pts:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    lines = vtk.vtkCellArray()
    for cell_ids in out_cells:
        if len(cell_ids) < 2:
            continue
        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(len(cell_ids))
        for j, vid in enumerate(cell_ids):
            pl.GetPointIds().SetId(j, int(vid))
        lines.InsertNextCell(pl)
    out.SetPoints(pts)
    out.SetLines(lines)
    rad_vtk = vtk.vtkDoubleArray()
    rad_vtk.SetName("MaximumInscribedSphereRadius")
    for r in state_radii:
        rad_vtk.InsertNextValue(float(r))
    out.GetPointData().AddArray(rad_vtk)
    cid_vtk = vtk.vtkIntArray()
    cid_vtk.SetName("CenterlineId")
    for c in state_cid:
        cid_vtk.InsertNextValue(int(c))
    out.GetPointData().AddArray(cid_vtk)
    gid_vtk = vtk.vtkIntArray()
    gid_vtk.SetName("GlobalNodeID")
    for g in state_gid:
        gid_vtk.InsertNextValue(int(g))
    out.GetPointData().AddArray(gid_vtk)
    if state_bif_label is not None:
        bif_vtk = vtk.vtkIntArray()
        bif_vtk.SetName("BifurcationLabel")
        for b in state_bif_label:
            bif_vtk.InsertNextValue(int(b))
        out.GetPointData().AddArray(bif_vtk)
    return out


def merge_centerline_branches_tree(
        centerline,
        radius_factor=0.2,
        min_tolerance=1e-5,
        max_widen_attempts=3,
        widen_factor=1.5,
        branch_scan_stride_divisor=1000,
        connector_interp_points=3,
        bifurcation_label_radius=0.0,
        vertex_eps=1e-5,
        verbose=False,
        decimate_max_points_per_branch=5000):
    """
    Merge overlapping branch polylines into a tree: start from the longest
    polyline, then attach each other branch by walking from its distal end
    inward until within a radius-based tolerance of the merged geometry,
    add a connector segment, and drop the overlapping proximal segment.

    If no junction is found after repeated tolerance widening, the full
    branch is appended unchanged.

    Parameters
    ----------
    centerline : vtkPolyData
        Input with one vtkPolyLine (or vtkLine) per branch; optional
        MaximumInscribedSphereRadius and CenterlineId point arrays.
    radius_factor : float
        Multiplier on local inscribed radius for acceptance tolerance.
    min_tolerance : float
        Floor on tolerance (physical units).
    max_widen_attempts : int
        Number of times to multiply tolerance by widen_factor and retry.
    widen_factor : float
        Factor applied when widening tolerance after a failed pass.
    branch_scan_stride_divisor : int
        Distal-to-proximal scan stride is ``max(1, n // branch_scan_stride_divisor)``
        where ``n`` is the number of points on that branch (e.g. divisor 1000
        gives stride 5 when ``n`` is 5000).
    connector_interp_points : int
        Number of interior points to linearly interpolate between the attach
        point on the merged tree and the first kept branch point, creating a
        denser connector segment at merge junctions.
    bifurcation_label_radius : float
        Radius used to label points near successful merge junctions.
        If <= 0, no bifurcation labeling is added.
    vertex_eps : float
        If the closest point on merged geometry is within this distance of
        an endpoint of the segment, snap to that vertex instead of inserting.
    verbose : bool
    decimate_max_points_per_branch : int or None
        Before merging, resample each branch polyline to at most this many
        points along cumulative arc length (per cell, no cross-cell welding).
        ``None`` or values ``< 2`` disable decimation.

    Returns
    -------
    vtkPolyData
        Merged centerline (no vtkCleanPolyData applied here).
    """
    merge_t0 = time.perf_counter()

    def _print_merge_time():
        if verbose:
            dt = time.perf_counter() - merge_t0
            print(f"Total tree merge time: {dt:.3f} s")

    centerline = _decimate_centerline_polylines_before_tree_merge(
        centerline, decimate_max_points_per_branch)

    polylines = _extract_polylines_from_polydata(centerline)
    if not polylines:
        _print_merge_time()
        return vtk.vtkPolyData()
    if len(polylines) == 1:
        _print_merge_time()
        return centerline
    branch_scan_stride_divisor = int(branch_scan_stride_divisor)
    if branch_scan_stride_divisor < 1:
        raise ValueError("branch_scan_stride_divisor must be >= 1")
    connector_interp_points = int(connector_interp_points)
    if connector_interp_points < 0:
        raise ValueError("connector_interp_points must be >= 0")
    bifurcation_label_radius = float(bifurcation_label_radius)
    if bifurcation_label_radius < 0:
        raise ValueError("bifurcation_label_radius must be >= 0")

    polylines.sort(key=lambda pl: _polyline_arc_length(pl['coords']),
                   reverse=True)

    # Longest branch seeds the merged structure
    first = polylines[0]
    coords0 = first['coords']
    rad0 = first['radii']
    cid0 = first['centerline_id']

    state_pts = [coords0[i].copy() for i in range(len(coords0))]
    if rad0 is not None:
        state_radii = [float(rad0[i]) for i in range(len(coords0))]
    else:
        med = _polyline_arc_length(coords0) / max(len(coords0) - 1, 1)
        state_radii = [float(med)] * len(coords0)
    state_cid = [int(cid0)] * len(coords0)
    state_gid = [int(i) for i in range(len(coords0))]
    state_bif_label = [0] * len(coords0)
    merge_point_ids = []
    segs = [(i, i + 1) for i in range(len(coords0) - 1)]
    # O(1) duplicate-edge checks; append_edges_for_cell used to scan all segs.
    seg_exist = {(min(u, v), max(u, v)) for u, v in segs}
    out_cells = [list(range(len(coords0)))]
    locator = None
    locator_seg_ids = []
    locator_dirty = True

    def rebuild_segment_locator():
        nonlocal locator, locator_seg_ids, locator_dirty
        seg_poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for p in state_pts:
            pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        lines = vtk.vtkCellArray()
        locator_seg_ids = []
        for seg_idx, (ia, ib) in enumerate(segs):
            if ia == ib:
                continue
            ln = vtk.vtkLine()
            ln.GetPointIds().SetId(0, int(ia))
            ln.GetPointIds().SetId(1, int(ib))
            lines.InsertNextCell(ln)
            locator_seg_ids.append(seg_idx)
        seg_poly.SetPoints(pts)
        seg_poly.SetLines(lines)
        loc = vtk.vtkStaticCellLocator()
        loc.SetDataSet(seg_poly)
        loc.BuildLocator()
        locator = loc
        locator_dirty = False

    def closest_to_segments(p):
        nonlocal locator_dirty
        if not segs:
            return np.inf, None, -1, 0.0, -1, -1
        if locator_dirty or locator is None:
            rebuild_segment_locator()
        cp = [0.0, 0.0, 0.0]
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)
        locator.FindClosestPoint(
            (float(p[0]), float(p[1]), float(p[2])),
            cp,
            cell_id,
            sub_id,
            dist2)
        loc_cell_id = int(cell_id)
        if loc_cell_id < 0 or loc_cell_id >= len(locator_seg_ids):
            return np.inf, None, -1, 0.0, -1, -1
        best_si = locator_seg_ids[loc_cell_id]
        best_a, best_b = segs[best_si]
        q = np.asarray(cp, dtype=float)
        ab = state_pts[best_b] - state_pts[best_a]
        denom = float(np.dot(ab, ab))
        if denom < 1e-30:
            best_t = 0.0
        else:
            best_t = float(np.dot(q - state_pts[best_a], ab) / denom)
            best_t = min(1.0, max(0.0, best_t))
        best_d = float(np.sqrt(float(dist2)))
        return best_d, q, best_si, best_t, best_a, best_b

    def split_segment_and_update_cells(seg_idx, q, t_cl, ra, rb, new_cid):
        """Insert q on segs[seg_idx] (a,b); return new vertex index."""
        nonlocal locator_dirty
        a, b = segs[seg_idx]
        seg_exist.discard((min(a, b), max(a, b)))
        new_id = len(state_pts)
        state_pts.append(q.copy())
        state_radii.append(float((1.0 - t_cl) * ra + t_cl * rb))
        state_cid.append(int(new_cid))
        state_gid.append(int(new_id))
        state_bif_label.append(0)
        segs[seg_idx:seg_idx + 1] = [(a, new_id), (new_id, b)]
        seg_exist.add((min(a, new_id), max(a, new_id)))
        seg_exist.add((min(new_id, b), max(new_id, b)))
        locator_dirty = True
        for cell in out_cells:
            for k in range(len(cell) - 1):
                u, v = cell[k], cell[k + 1]
                if {u, v} == {a, b}:
                    cell.insert(k + 1, new_id)
                    break
        return new_id

    def resolve_junction(q, seg_idx, t_cl, a, b, branch_cid):
        pa = state_pts[a]
        pb = state_pts[b]
        ra = state_radii[a]
        rb = state_radii[b]
        if np.linalg.norm(q - pa) <= vertex_eps:
            return a
        if np.linalg.norm(q - pb) <= vertex_eps:
            return b
        return split_segment_and_update_cells(
            seg_idx, q, t_cl, ra, rb, branch_cid)

    merge_pt_eps = max(min_tolerance * 0.01, 1e-9)
    inv_merge_pt_eps = 1.0 / merge_pt_eps
    point_bins = {}

    def point_bin_key(p):
        return (int(math.floor(float(p[0]) * inv_merge_pt_eps)),
                int(math.floor(float(p[1]) * inv_merge_pt_eps)),
                int(math.floor(float(p[2]) * inv_merge_pt_eps)))

    def add_point_to_bins(pid):
        key = point_bin_key(state_pts[pid])
        if key not in point_bins:
            point_bins[key] = [pid]
        else:
            point_bins[key].append(pid)

    for pid in range(len(state_pts)):
        add_point_to_bins(pid)

    def add_point_if_new(p, r, branch_cid, merge_pt_eps_unused):
        p = np.asarray(p, dtype=float).ravel()
        bx, by, bz = point_bin_key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (bx + dx, by + dy, bz + dz)
                    if key not in point_bins:
                        continue
                    for j in point_bins[key]:
                        sj = state_pts[j]
                        if np.linalg.norm(sj - p) <= merge_pt_eps:
                            return j
        jid = len(state_pts)
        state_pts.append(p.copy())
        state_radii.append(float(r))
        state_cid.append(int(branch_cid))
        state_gid.append(int(jid))
        state_bif_label.append(0)
        add_point_to_bins(jid)
        return jid

    def add_point_force_new(p, r, branch_cid):
        """Insert a new point even if it overlaps existing points."""
        p = np.asarray(p, dtype=float).ravel()
        jid = len(state_pts)
        state_pts.append(p.copy())
        state_radii.append(float(r))
        state_cid.append(int(branch_cid))
        state_gid.append(int(jid))
        state_bif_label.append(0)
        add_point_to_bins(jid)
        return jid

    def append_edges_for_cell(cell_ids):
        nonlocal locator_dirty
        added = False
        for u, v in zip(cell_ids[:-1], cell_ids[1:]):
            if u == v:
                continue
            ek = (min(u, v), max(u, v))
            if ek in seg_exist:
                continue
            seg_exist.add(ek)
            segs.append((u, v))
            added = True
        if added:
            locator_dirty = True

    for pl in polylines[1:]:
        if verbose:
            print(f"Processing branch: {pl['centerline_id']}; Number of points: {len(pl['coords'])}")
            # print(f"Number of radii: {len(pl['radii'])}")
            # print(f"Number of segments: {len(segs)}")
            # print(f"Number of out_cells: {len(out_cells)}")
            # print(f"Number of state_pts: {len(state_pts)}")
            # print(f"Number of state_radii: {len(state_radii)}")
            # print(f"Number of state_cid: {len(state_cid)}")
        bcoords = pl['coords']
        brad = pl['radii']
        n = len(bcoords)
        if n < 2:
            continue
        if brad is None:
            brlen = _polyline_arc_length(bcoords)
            default_r = brlen / max(n - 1, 1)
        else:
            default_r = None
        branch_stride = max(1, n // branch_scan_stride_divisor)
        scan_indices = list(range(n - 1, -1, -branch_stride))
        if scan_indices and scan_indices[-1] != 0:
            scan_indices.append(0)

        junction_i = None
        attach_id = None
        scale = 1.0
        for _attempt in range(max_widen_attempts):
            for i in scan_indices:
                p = bcoords[i]
                if brad is not None:
                    loc_r = float(brad[i])
                else:
                    loc_r = float(default_r)
                tol = max(min_tolerance,
                          scale * radius_factor * max(loc_r, 1e-12))
                d, q, si, t_cl, a, b = closest_to_segments(p)
                if d <= tol:
                    junction_i = i
                    attach_id = resolve_junction(
                        q, si, t_cl, a, b, pl['centerline_id'])
                    merge_point_ids.append(attach_id)
                    break
            if junction_i is not None:
                break
            else:
                if verbose:
                    print(f"Could not join branch centerline_id={pl['centerline_id']}; appending full polyline after {max_widen_attempts} widen attempts")
            scale *= widen_factor

        if junction_i is None:
            if verbose:
                print("    tree merge: could not join branch "
                      f"centerline_id={pl['centerline_id']}; appending full "
                      f"polyline after {max_widen_attempts} widen attempts")
            new_ids = []
            for i in range(n):
                r_i = float(brad[i]) if brad is not None else float(
                    default_r)
                pid = add_point_if_new(
                    bcoords[i], r_i, pl['centerline_id'], merge_pt_eps)
                new_ids.append(pid)
            # Deduplicate consecutive equal ids
            trimmed = [new_ids[0]]
            for pid in new_ids[1:]:
                if pid != trimmed[-1]:
                    trimmed.append(pid)
            if len(trimmed) >= 2:
                out_cells.append(trimmed)
                append_edges_for_cell(trimmed)
            continue

        tail_ids = []
        for k in range(junction_i, n):
            r_k = (float(brad[k]) if brad is not None
 else float(default_r))
            tid = add_point_if_new(
                bcoords[k], r_k, pl['centerline_id'], merge_pt_eps)
            tail_ids.append(tid)

        # If the first kept branch point snaps onto the attach vertex,
        # keep an explicit connector point at the merged junction.
        if tail_ids and tail_ids[0] == attach_id:
            j_r = (float(brad[junction_i]) if brad is not None
                   else float(default_r))
            if np.linalg.norm(bcoords[junction_i] - state_pts[attach_id]) > (
                    1e-12):
                tail_ids[0] = add_point_force_new(
                    bcoords[junction_i], j_r, pl['centerline_id'])

        new_cell = [attach_id]
        if tail_ids:
            start_p = state_pts[attach_id]
            end_p = state_pts[tail_ids[0]]
            connector_vec = end_p - start_p
            if float(np.dot(connector_vec, connector_vec)) > 1e-24:
                start_r = float(state_radii[attach_id])
                end_r = float(state_radii[tail_ids[0]])
                for j in range(1, connector_interp_points + 1):
                    alpha = j / float(connector_interp_points + 1)
                    p_interp = (1.0 - alpha) * start_p + alpha * end_p
                    r_interp = (1.0 - alpha) * start_r + alpha * end_r
                    interp_id = add_point_force_new(
                        p_interp, r_interp, pl['centerline_id'])
                    if new_cell[-1] != interp_id:
                        new_cell.append(interp_id)
        for tid in tail_ids:
            if tid == attach_id:
                continue
            if new_cell and tid == new_cell[-1]:
                continue
            new_cell.append(tid)
        if len(new_cell) >= 2:
            out_cells.append(new_cell)
            append_edges_for_cell(new_cell)
        elif verbose:
            print("    tree merge: degenerate cell after merge for "
                  f"centerline_id={pl['centerline_id']}, skipped")

    if bifurcation_label_radius > 0 and merge_point_ids:
        merge_unique = sorted(set(merge_point_ids))
        merge_pts = np.asarray(
            [state_pts[mid] for mid in merge_unique], dtype=float)
        tree = cKDTree(merge_pts)
        pts_arr = np.asarray(state_pts, dtype=float)
        near = tree.query_ball_point(pts_arr, r=bifurcation_label_radius)
        for j, hits in enumerate(near):
            if hits:
                state_bif_label[j] = 1

    out = _tree_merge_state_to_polydata(
        state_pts, state_radii, state_cid, state_gid, out_cells,
        state_bif_label=state_bif_label)
    _print_merge_time()
    return out


def post_process_centerline(centerline, verbose=False,
                            merge_method='clean',
                            tree_merge_radius_factor=1.0,
                            tree_merge_min_tolerance=1e-6,
                            tree_merge_max_widen_attempts=3,
                            tree_merge_widen_factor=1.5,
                            tree_merge_branch_scan_stride_divisor=5000,
                            tree_merge_connector_interp_points=5,
                            tree_merge_bifurcation_label_radius=1.0,
                            tree_merge_vertex_eps=1e-5,
                            tree_merge_decimate_max_points_per_branch=5000):
    """
    Function to post process the centerline using vtk functionalities.

    merge_method ``'clean'`` (default):
        1. Remove duplicate points (vtkCleanPolyData).
        2. Smooth centerline.

    merge_method ``'tree'``:
        1. Arc-length decimate each branch (see ``tree_merge_decimate_max_points_per_branch``), then merge branches with merge_centerline_branches_tree (radius-based tolerance, longest branch first, connector segments; no vtk clean).
        2. Smooth centerline.

    Parameters
    ----------
    centerline : vtkPolyData
        Centerline of the vessel.
    merge_method : {'clean', 'tree'}
        Post-merge strategy before smoothing.
    tree_merge_radius_factor : float
        Passed to merge_centerline_branches_tree when merge_method='tree'.
    tree_merge_min_tolerance : float
    tree_merge_max_widen_attempts : int
    tree_merge_widen_factor : float
    tree_merge_branch_scan_stride_divisor : int
        Passed to merge_centerline_branches_tree as ``branch_scan_stride_divisor``.
    tree_merge_connector_interp_points : int
    tree_merge_bifurcation_label_radius : float
    tree_merge_vertex_eps : float
    tree_merge_decimate_max_points_per_branch : int or None
        Passed to :func:`merge_centerline_branches_tree` as
        ``decimate_max_points_per_branch`` (arc-length decimation per branch
        before merging). Ignored when ``merge_method`` is ``'clean'``.

    Returns
    -------
    centerline : vtkPolyData
    """
    if merge_method not in ('clean', 'tree'):
        raise ValueError("merge_method must be 'clean' or 'tree'")

    if verbose:
        print(f"Merge method: {merge_method}")
        print(f"""Number of points before post processing:
              {centerline.GetNumberOfPoints()}""")
    if merge_method == 'clean':
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(centerline)
        cleaner.SetTolerance(0.001)
        cleaner.Update()
        centerline = cleaner.GetOutput()
    else:
        centerline = merge_centerline_branches_tree(
            centerline,
            radius_factor=tree_merge_radius_factor,
            min_tolerance=tree_merge_min_tolerance,
            max_widen_attempts=tree_merge_max_widen_attempts,
            widen_factor=tree_merge_widen_factor,
            branch_scan_stride_divisor=tree_merge_branch_scan_stride_divisor,
            connector_interp_points=tree_merge_connector_interp_points,
            bifurcation_label_radius=tree_merge_bifurcation_label_radius,
            vertex_eps=tree_merge_vertex_eps,
            verbose=verbose,
            decimate_max_points_per_branch=(
                tree_merge_decimate_max_points_per_branch),
        )
        # Now clean
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(centerline)
        cleaner.SetTolerance(0.001)
        cleaner.Update()
        centerline = cleaner.GetOutput()

    # Smooth centerline
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(centerline)
    smoother.SetNumberOfIterations(1500)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    # smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(0.1)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    centerline = smoother.GetOutput()

    if verbose:
        print(f"""Number of points after post processing:
              {centerline.GetNumberOfPoints()}""")

    return centerline


def check_border(seed, seg_size):
    """
    Function to check if a point is on the border of the segmentation.
    If it is, move it inside the segmentation.

    Parameters
    ----------
    seed : np.array
        Seed point.
    seg_size : np.array
        Segmentation size.

    Returns
    -------
    seed : np.array
    """
    for i in range(3):
        if seed[i] <= 0:
            seed[i] = 2
        if seed[i] >= seg_size[i]:
            seed[i] = seg_size[i] - 3

    return seed


# @jit(nopython=True)
def get_neighbors(index, cluster_map_shape):
    """
    Function to get the neighbors of a voxel.

    Parameters
    ----------
    index : np.array
        Index of the voxel.
    cluster_map_shape : tuple
        Shape of the cluster map.

    Returns
    -------
    neighbors : list of np.array
    """
    neighbors = []
    for i in range(3):
        for j in [-1, 1]:
            neighbor = index.copy()
            neighbor[i] += j
            if neighbor[i] >= 0 and neighbor[i] < cluster_map_shape[i]:
                neighbors.append(neighbor)
    # print(f"Neighbors: {neighbors}")
    return neighbors


def get_neighbors_diag(index, cluster_map_shape):
    """
    Function to get the neighbors of a voxel including diagonals.

    Parameters
    ----------
    index : np.array
        Index of the voxel.
    cluster_map_shape : tuple
        Shape of the cluster map.

    Returns
    -------
    neighbors : list of np.array
    """
    neighbors = []
    for i in range(3):
        for j in [-1, 0, 1]:
            neighbor = index.copy()
            neighbor[i] += j
            if neighbor[i] >= 0 and neighbor[i] < cluster_map_shape[i]:
                neighbors.append(neighbor)

    return neighbors


# @jit(nopython=True)
def check_connections(value, connections, values_connected,
                      neighbors, cluster_map):
    for neighbor in neighbors:
        # If neighbor is not in the current cluster
        if (cluster_map[neighbor[0], neighbor[1], neighbor[2]] != value
           and cluster_map[neighbor[0], neighbor[1], neighbor[2]] != 0
           and cluster_map[neighbor[0], neighbor[1], neighbor[2]]
           not in values_connected):
            # Increment number of connections
            connections += 1
            # Add value of connected cluster
            values_connected.append(cluster_map[neighbor[0],
                                                neighbor[1],
                                                neighbor[2]])
            # If more than one connection, break
            if connections > 1:
                break
    return connections, values_connected


# @jit(nopython=True, parallel=True)
def find_end_clusters(cluster_map):
    """
    Function to find the end clusters of a cluster map.
    The end clusters are the clusters that only connect to one other cluster.

    Clusters are connected if they share a face (6-connectivity), matching
    ``get_neighbors``.

    Implemented as one pass over inter-voxel faces between different non-zero
    labels instead of ``np.argwhere`` per label (which is O(n_clusters * volume)).

    Parameters
    ----------
    cluster_map : np.ndarray
        Cluster map. Each cluster has a unique integer value.

    Returns
    -------
    end_clusters : list of int
    """
    chunks = []
    for axis in range(3):
        sl1 = [slice(None)] * 3
        sl2 = [slice(None)] * 3
        sl1[axis] = slice(1, None)
        sl2[axis] = slice(0, -1)
        c1 = cluster_map[tuple(sl1)]
        c2 = cluster_map[tuple(sl2)]
        m = (c1 != 0) & (c2 != 0) & (c1 != c2)
        if not np.any(m):
            continue
        a = c1[m]
        b = c2[m]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        chunks.append(np.column_stack((lo, hi)))

    if not chunks:
        return []

    edges = np.unique(np.vstack(chunks), axis=0)
    lo = edges[:, 0]
    hi = edges[:, 1]
    u = np.concatenate([lo, hi])
    v = np.concatenate([hi, lo])
    order = np.argsort(u, kind="mergesort")
    u_s = u[order]
    v_s = v[order]
    starts = np.flatnonzero(np.r_[True, u_s[1:] != u_s[:-1]])
    ends = np.r_[starts[1:], u_s.size]
    labels_at_starts = u_s[starts]

    neighbor_sets = {}
    for s, e, lab in zip(starts, ends, labels_at_starts):
        neighbor_sets[int(lab)] = set(v_s[s:e])

    return [lab for lab, neigh in neighbor_sets.items() if len(neigh) == 1]


def cluster_map(segmentation, return_wave_distance_map=False,
                out_dir=None, write_files=False, verbose=False):
    """
    Function to cluster a distance map of a segmentation into integer values.

    The segmentation is converted to a inverted distance map
    (high value inside).
    The maximum value in the distance map is found.
    The index of the maximum value is used as the seed point for
    the fast marching method.
    The wave distance map is calculated using the fast marching method
    and the seed point.
    The wave distance map is converted to integers.
    Each connected region of the same integer value is considered a cluster.
    Each cluster is assigned a unique integer value.

    Parameters
    ----------
    segmentation : sitk image
        Segmentation of the vessel.
    Returns
    -------
    seed : np.array
        Seed point.
    clusters : list of np.array
        List of cluster points.
    """
    max_time_steps = 1000
    # Resample segmentation to smaller
    divide = 1
    if divide != 1:
        segmentation = sitk.Resample(segmentation,
                                     [int(x/divide)
                                      for x in segmentation.GetSize()],
                                     sitk.Transform(),
                                     sitk.sitkNearestNeighbor,
                                     segmentation.GetOrigin(),
                                     [x*divide
                                      for x in segmentation.GetSpacing()],
                                     segmentation.GetDirection(), 0,
                                     segmentation.GetPixelID())
    # Create distance map
    distance_map = distance_map_from_seg(segmentation)
    # Invert distance map
    distance_map = distance_map * -1

    distance_map_masked = sitk.Mask(distance_map, segmentation)

    # Shift distance map by 0.5
    distance_map = distance_map + 1

    distance_map_masked = sitk.Mask(distance_map, segmentation)

    # Find maximum value
    dist_map_np = sitk.GetArrayFromImage(distance_map).transpose(2, 1, 0)
    max_value = np.max(dist_map_np)
    # Find index of maximum value
    index = np.where(dist_map_np == max_value)
    # If multiple max values, take first; use .item() so we pass scalars not 0-d/1-d arrays
    z0, y0, x0 = index[0][0], index[1][0], index[2][0]
    # Create fast marching filter
    fast_marching = sitk.FastMarchingImageFilter()
    fast_marching.AddTrialPoint((z0.item(), y0.item(), x0.item()))
    fast_marching.SetStoppingValue(max_time_steps)
    # Create speed image by masking distance map with segmentation
    speed_image = sitk.Mask(distance_map, segmentation)
    # And rescale to 0.0001-max_value
    speed_image = sitk.RescaleIntensity(speed_image, 0.00001, float(max_value))
    # And raise all values to the power of 0.5
    speed_image = sitk.Pow(speed_image, 0.5)
    print("Done preprocessing speed image")

    # Calculate wave distance map
    wave_distance_map_output = fast_marching.Execute(speed_image)
    print("Done fast marching")
    # Only keep values inside the segmentation
    wave_distance_map = sitk.Mask(wave_distance_map_output, segmentation)
    # Convert wave distance map to integers
    wave_distance_map = sitk.Cast(wave_distance_map, sitk.sitkInt32)
    if out_dir and write_files:
        sitk.WriteImage(segmentation,
                        os.path.join(out_dir, 'segmentation_from_cluster.mha'))
        sitk.WriteImage(wave_distance_map,
                        os.path.join(out_dir, 'wave_distance_map.mha'))
        sitk.WriteImage(speed_image,
                        os.path.join(out_dir, 'speed_image.mha'))
        sitk.WriteImage(wave_distance_map_output,
                        os.path.join(out_dir,
                                     'wave_distance_map_output.mha'))
    print("Converted wave distance map to integers")

    # Get unique values in wave distance map (cached NumPy view reused below)
    wave_distance_map_np = sitk.GetArrayFromImage(wave_distance_map).transpose(2, 1, 0)
    unique_values = np.unique(wave_distance_map_np)
    if verbose:
        print(f"Unique values: {unique_values}")
    # Remove values above threshold 100 thousand
    unique_values = unique_values[unique_values < 100000]
    # Build cluster labels in NumPy and convert once at the end.
    cluster_map_np = np.zeros_like(wave_distance_map_np, dtype=np.int32)

    time_start = time.time()
    # Group every N values
    max_unique = int(np.max(unique_values)) if unique_values.size > 0 else 0
    if max_unique > 100:
        N = 10
    elif max_unique > 50:
        N = 5
    elif max_unique > 20:
        N = 3
    elif max_unique > 10:
        N = 2
    else:
        N = 1

    # Quantize wave distance into bucket ids once, then find connected
    # components for equal-valued neighbors in one graph pass (26-connectivity).
    valid_wave = (wave_distance_map_np > 0) & (wave_distance_map_np < 100000)
    bucket_ids = np.zeros_like(wave_distance_map_np, dtype=np.int32)
    if np.any(valid_wave):
        bucket_ids[valid_wave] = ((wave_distance_map_np[valid_wave] - 1) // N) + 1

    active_mask = bucket_ids > 0
    cluster_count = 1
    if np.any(active_mask):
        shape = bucket_ids.shape
        flat_indices = np.arange(bucket_ids.size, dtype=np.int64).reshape(shape)
        active_flats = flat_indices[active_mask]
        active_bucket_ids = bucket_ids[active_mask].astype(np.int32, copy=False)

        full_to_active = np.full(bucket_ids.size, -1, dtype=np.int64)
        full_to_active[active_flats] = np.arange(active_flats.size, dtype=np.int64)

        # Half-neighborhood offsets for 26-connectivity.
        neighbor_offsets = (
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        )

        edge_us = []
        edge_vs = []
        for dx, dy, dz in neighbor_offsets:
            x1 = slice(max(0, -dx), shape[0] - max(0, dx))
            x2 = slice(max(0, dx), shape[0] - max(0, -dx))
            y1 = slice(max(0, -dy), shape[1] - max(0, dy))
            y2 = slice(max(0, dy), shape[1] - max(0, -dy))
            z1 = slice(max(0, -dz), shape[2] - max(0, dz))
            z2 = slice(max(0, dz), shape[2] - max(0, -dz))

            b1 = bucket_ids[x1, y1, z1]
            b2 = bucket_ids[x2, y2, z2]
            matches = (b1 != 0) & (b1 == b2)
            if not np.any(matches):
                continue

            u_full = flat_indices[x1, y1, z1][matches]
            v_full = flat_indices[x2, y2, z2][matches]
            edge_us.append(full_to_active[u_full])
            edge_vs.append(full_to_active[v_full])

        n_active = active_flats.size
        if edge_us:
            u = np.concatenate(edge_us)
            v = np.concatenate(edge_vs)
            rows = np.concatenate([u, v])
            cols = np.concatenate([v, u])
            data = np.ones(rows.size, dtype=np.uint8)
            graph = csr_matrix((data, (rows, cols)), shape=(n_active, n_active))
        else:
            graph = csr_matrix((n_active, n_active), dtype=np.uint8)

        _, component_labels = connected_components(graph,
                                                   directed=False,
                                                   return_labels=True)
        comp_counts = np.bincount(component_labels)

        order = np.argsort(component_labels, kind="mergesort")
        comp_sorted = component_labels[order]
        flat_sorted = active_flats[order]
        bucket_sorted = active_bucket_ids[order]
        starts = np.r_[0, np.flatnonzero(comp_sorted[1:] != comp_sorted[:-1]) + 1]
        comp_ids = comp_sorted[starts]
        comp_first_flat = flat_sorted[starts]
        comp_bucket = bucket_sorted[starts]
        comp_sizes = comp_counts[comp_ids]

        if verbose:
            buckets_present, num_connected = np.unique(comp_bucket, return_counts=True)
            for bucket_id, n_conn in zip(buckets_present, num_connected):
                value = int((bucket_id - 1) * N + 1)
                print(f"Values: {value} to {value+N-1};   Num connected: {int(n_conn)}")

        keep = comp_sizes >= 5
        keep_comp_ids = comp_ids[keep]
        keep_first_flat = comp_first_flat[keep]
        keep_bucket = comp_bucket[keep]

        # Preserve original order semantics: increasing bucket then scan order.
        keep_order = np.lexsort((keep_first_flat, keep_bucket))
        keep_comp_ids = keep_comp_ids[keep_order]

        comp_to_cluster = np.zeros(comp_counts.size, dtype=np.int32)
        for comp_id in keep_comp_ids:
            comp_to_cluster[comp_id] = cluster_count
            cluster_count += 1

        cluster_labels_active = comp_to_cluster[component_labels]
        cluster_map_flat = np.zeros(bucket_ids.size, dtype=np.int32)
        cluster_map_flat[active_flats] = cluster_labels_active
        cluster_map_np = cluster_map_flat.reshape(shape)

    cluster_map_img = sitk.GetImageFromArray(np.transpose(cluster_map_np, (2, 1, 0)))
    cluster_map_img.CopyInformation(wave_distance_map)

    print(f"Cluster count: {cluster_count}")
    print(f"Time to create cluster map: {time.time() - time_start:0.2f}")

    # Write image
    if out_dir and write_files:
        sitk.WriteImage(cluster_map_img, os.path.join(out_dir,
                                                      'cluster_map_img.mha'))

    time_start = time.time()
    end_clusters = find_end_clusters(cluster_map_np)
    print(f"Time to find end clusters: {time.time() - time_start:0.2f}")
    print(f"End clusters: {end_clusters}")
    print(f"Number of end clusters: {len(end_clusters)}")

    distance_map_np = sitk.GetArrayFromImage(distance_map_masked).transpose(
        2, 1, 0)
    time_start = time.time()
    end_points = get_end_points(cluster_map_img,
                                end_clusters,
                                distance_map_masked,
                                cluster_map_np=cluster_map_np,
                                distance_map_np=distance_map_np)
    print(f"Time to get end points: {time.time() - time_start:0.2f}")

    # Convert end points to physical points
    end_points_phys = [segmentation.TransformIndexToPhysicalPoint(
        end_point.tolist()) for end_point in end_points]

    # Create vtk polydata for points
    if out_dir and write_files:
        polydata_point = points2polydata(end_points_phys)
        write_geo(os.path.join(out_dir, 'end_points.vtp'), polydata_point)

    seed_np = np.array(
        segmentation.TransformIndexToPhysicalPoint(
            (int(z0), int(y0), int(x0))))
    end_points_phys_np = [np.array(point) for point in end_points_phys]

    # write seed and end points as npy
    # np.save('/Users/numisveins/Downloads/debug_centerline/seed.npy',
    #         seed_np)
    # np.save('/Users/numisveins/Downloads/debug_centerline/end_points.npy',
    #         end_points_phys_np)

    if return_wave_distance_map:

        return seed_np, end_points_phys_np, wave_distance_map_output

    return seed_np, end_points_phys_np


def extract_disconnected_body_around_seed(segmentation, seed, verbose=False):
    """
    Extract the disconnected body (connected component) that contains the given seed point.
    
    This function isolates a single connected component from a multi-component segmentation
    by finding the component that contains the specified seed point. This is useful for
    calculating centerlines on individual components rather than the entire segmentation.
    
    Parameters
    ----------
    segmentation : sitk.Image
        Binary segmentation image that may contain multiple disconnected bodies
    seed : np.array
        Seed point in physical coordinates that should be contained within the
        extracted component
    verbose : bool, optional
        If True, prints detailed information about the extraction process. Defaults to False.
        
    Returns
    -------
    component_segmentation : sitk.Image
        Binary segmentation containing only the connected component that contains
        the seed point. Has the same properties (size, spacing, origin, direction)
        as the input segmentation.
        
    Raises
    ------
    ValueError
        If the seed point is not within any connected component of the segmentation
        
    Example
    -------
    >>> segmentation = sitk.ReadImage("multi_vessel.nii.gz")
    >>> seed = np.array([10.5, 20.3, 15.7])  # Physical coordinates
    >>> single_component = extract_disconnected_body_around_seed(segmentation, seed, verbose=True)
    >>> print(f"Original volume: {np.sum(sitk.GetArrayFromImage(segmentation))}")
    >>> print(f"Component volume: {np.sum(sitk.GetArrayFromImage(single_component))}")
    """
    if verbose:
        print(f"Extracting disconnected body around seed: {seed}")
    
    # Convert seed to index coordinates
    try:
        seed_index = segmentation.TransformPhysicalPointToIndex(seed.tolist())
    except Exception as e:
        raise ValueError(f"Failed to convert seed point to image coordinates: {e}")
    
    # Validate seed is within image bounds
    seg_size = segmentation.GetSize()
    if (seed_index[0] < 0 or seed_index[1] < 0 or seed_index[2] < 0 or
        seed_index[0] >= seg_size[0] or seed_index[1] >= seg_size[1] or seed_index[2] >= seg_size[2]):
        # Try to clamp to bounds
        seed_index = [max(0, min(seed_index[0], seg_size[0]-1)),
                     max(0, min(seed_index[1], seg_size[1]-1)),
                     max(0, min(seed_index[2], seg_size[2]-1))]
        if verbose:
            print(f"  Warning: Seed was outside bounds, clamped to: {seed_index}")
    
    # Check if seed is within segmentation mask
    seg_value = segmentation.GetPixel(seed_index)
    if seg_value == 0:
        if verbose:
            print(f"  Warning: Seed at index {seed_index} is outside segmentation mask")
        # Find nearest non-zero voxel
        seg_array = sitk.GetArrayFromImage(segmentation).transpose(2, 1, 0)
        nonzero_indices = np.argwhere(seg_array > 0)
        if len(nonzero_indices) == 0:
            raise ValueError("No non-zero voxels found in segmentation")
        
        distances = np.linalg.norm(nonzero_indices - np.array(seed_index), axis=1)
        nearest_idx = nonzero_indices[np.argmin(distances)]
        seed_index = nearest_idx.tolist()
        if verbose:
            print(f"  Moved seed to nearest segmentation voxel: {seed_index}")
    
    if verbose:
        print(f"  Seed index: {seed_index}")
        print(f"  Original segmentation volume: {np.sum(sitk.GetArrayFromImage(segmentation))}")
    
    # Find connected components
    connected_filter = sitk.ConnectedComponentImageFilter()
    connected_filter.SetFullyConnected(True)  # Include diagonal connections
    connected_components = connected_filter.Execute(segmentation)
    
    # Get the label of the component containing the seed
    seed_label = connected_components.GetPixel(seed_index)
    
    if seed_label == 0:
        raise ValueError(f"Seed point at {seed} (index {seed_index}) is not within any connected component")
    
    if verbose:
        # Get component statistics for reporting
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(connected_components)
        num_components = label_stats.GetNumberOfLabels()
        seed_component_size = label_stats.GetNumberOfPixels(seed_label)
        
        print(f"  Found {num_components} total components")
        print(f"  Seed is in component {seed_label} with {seed_component_size} voxels")
    
    # Create binary mask for the component containing the seed
    component_mask = sitk.BinaryThreshold(connected_components,
                                        lowerThreshold=seed_label,
                                        upperThreshold=seed_label,
                                        insideValue=1,
                                        outsideValue=0)
    
    # Apply mask to original segmentation to extract just this component
    component_segmentation = sitk.Mask(segmentation, component_mask)
    
    # Ensure the result is binary
    component_segmentation = sitk.BinaryThreshold(component_segmentation,
                                                lowerThreshold=1,
                                                upperThreshold=255,
                                                insideValue=1,
                                                outsideValue=0)
    
    if verbose:
        component_volume = np.sum(sitk.GetArrayFromImage(component_segmentation))
        print(f"  Extracted component volume: {component_volume}")
        reduction_ratio = component_volume / np.sum(sitk.GetArrayFromImage(segmentation))
        print(f"  Volume reduction: {reduction_ratio:.3f} ({100*(1-reduction_ratio):.1f}% reduction)")
    
    return component_segmentation


def create_seeds_from_disconnected_bodies(segmentation, nr_seeds=None, verbose=False):
    """
    Create seeds based on disconnected bodies in the segmentation.
    For each disconnected body, pick the point with the highest surface distance value.
    
    Parameters
    ----------
    segmentation : sitk image
        Input segmentation with potentially multiple disconnected bodies
    nr_seeds : int, optional
        Number of seeds to create. If None, creates one seed per disconnected body.
        If the requested number exceeds the number of disconnected bodies, it will be
        automatically reduced to match the number of available bodies with a warning.
        Seeds are created from the largest disconnected bodies when fewer than all are requested.
    verbose : bool
        Print detailed information
        
    Returns
    -------
    seeds : list of np.array
        List of seed points in physical coordinates
    component_info : dict
        Information about the connected components
        
    Example
    -------
    >>> segmentation = sitk.ReadImage("multi_vessel.nii.gz")
    >>> seeds, info = create_seeds_from_disconnected_bodies(segmentation, nr_seeds=3, verbose=True)
    >>> print(f"Created {len(seeds)} seeds from {info['num_components']} components")
    """
    if verbose:
        print(f"Analyzing disconnected bodies in segmentation...")
        seg_array = sitk.GetArrayFromImage(segmentation)
        print(f"  Total non-zero voxels: {np.sum(seg_array > 0)}")
    
    # Find connected components
    connected_filter = sitk.ConnectedComponentImageFilter()
    connected_filter.SetFullyConnected(True)  # Include diagonal connections
    connected_components = connected_filter.Execute(segmentation)
    
    # Get component statistics
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(connected_components)
    
    num_components = label_stats.GetNumberOfLabels()
    if verbose:
        print(f"  Found {num_components} disconnected bodies")
    
    # Get component sizes and sort by size (largest first)
    component_info = []
    for label in label_stats.GetLabels():
        size = label_stats.GetNumberOfPixels(label)
        centroid = label_stats.GetCentroid(label)
        component_info.append({
            'label': label,
            'size': size,
            'centroid': centroid
        })
    
    # Sort by size (largest first)
    component_info.sort(key=lambda x: x['size'], reverse=True)
    
    if verbose:
        print(f"  Component sizes: {[comp['size'] for comp in component_info]}")
    
    # Determine how many seeds to create
    if nr_seeds is None:
        nr_seeds = num_components
        if verbose:
            print(f"  No seed count specified, creating one seed per component ({nr_seeds} seeds)")
    else:
        original_nr_seeds = nr_seeds
        nr_seeds = min(nr_seeds, num_components)
        
        if verbose:
            if original_nr_seeds > num_components:
                print(f"  Requested {original_nr_seeds} seeds, but only {num_components} disconnected bodies found")
                print(f"  Adjusting to create {nr_seeds} seeds (one per disconnected body)")
            else:
                print(f"  Creating {nr_seeds} seeds from largest components")
        
        # Also provide a warning even when not verbose
        elif original_nr_seeds > num_components:
            print(f"Warning: Requested {original_nr_seeds} seeds, but only found {num_components} disconnected bodies. Creating {nr_seeds} seeds.")
    
    # Calculate surface distance map for the entire segmentation
    distance_map = distance_map_from_seg(segmentation)
    # Invert so positive values are inside (higher = more central)
    distance_map = distance_map * -1
    distance_map_array = sitk.GetArrayFromImage(distance_map).transpose(2, 1, 0)
    connected_array = sitk.GetArrayFromImage(connected_components).transpose(2, 1, 0)
    
    seeds = []
    selected_components = component_info[:nr_seeds]
    
    for i, comp in enumerate(selected_components):
        label = comp['label']
        if verbose:
            print(f"    Processing component {label} (size: {comp['size']})")
        
        # Get all indices belonging to this component
        component_indices = np.argwhere(connected_array == label)
        
        if len(component_indices) == 0:
            if verbose:
                print(f"      Warning: No voxels found for component {label}")
            continue
        
        # Find the point with maximum distance value within this component
        max_distance = -np.inf
        best_index = None
        
        for idx in component_indices:
            distance_val = distance_map_array[idx[0], idx[1], idx[2]]
            if distance_val > max_distance:
                max_distance = distance_val
                best_index = idx
        
        if best_index is not None:
            # Convert index to physical coordinates
            physical_point = segmentation.TransformIndexToPhysicalPoint(best_index.tolist())
            seeds.append(np.array(physical_point))
            
            if verbose:
                print(f"      Seed {i+1}: index {best_index} -> physical {physical_point}")
                print(f"      Distance value: {max_distance:.6f}")
                print(f"      Component size: {comp['size']} voxels")
                
                # Check seed quality
                if max_distance < 0.5:
                    print(f"      Warning: Low distance value may indicate problematic seed placement")
                
                # Check component shape characteristics
                if comp['size'] < 100:
                    print(f"      Warning: Small component size may lead to centerline calculation issues")
        else:
            if verbose:
                print(f"      Warning: Could not find valid seed for component {label}")
                print(f"      Component had {len(component_indices)} indices but no valid distance values")
    
    if verbose:
        print(f"  Successfully created {len(seeds)} seeds")
    
    # Write seeds as polydata for visualization/debugging
    if len(seeds) > 0:
        seeds_polydata = points2polydata(seeds)
        return seeds, {
            'num_components': num_components,
            'component_info': component_info,
            'selected_components': selected_components,
            'seeds_polydata': seeds_polydata
        }
    else:
        return seeds, {
            'num_components': num_components,
            'component_info': component_info,
            'selected_components': selected_components,
            'seeds_polydata': None
        }


def get_end_points(cluster_map_img, end_clusters, distance_map_masked,
                   cluster_map_np=None, distance_map_np=None):
    """
    Function to get the end points of a cluster map.

    For each end cluster label, returns the voxel of that label with maximum
    distance value. Tie and non-positive behavior matches legacy:
    keep the first voxel in ``np.argwhere`` order among equal maxima, and
    return ``None`` when all distances in the label are <= 0.

    Parameters
    ----------
    cluster_map_img : sitk image
        Cluster map.
    end_clusters : list of int
        End clusters.
    distance_map_masked : sitk image
        Distance map masked with the segmentation.
    cluster_map_np : np.ndarray, optional
        Same voxel layout as ``sitk.GetArrayFromImage(cluster_map_img).T(2,1,0)``.
        If provided, avoids copying the cluster map from SimpleITK.
    distance_map_np : np.ndarray, optional
        Same layout for the masked distance map.

    Returns
    -------
    end_points : list of np.array
        One entry per ``end_cluster``; ``None`` if the label has no voxels.
    """
    if not end_clusters:
        return []

    if cluster_map_np is None:
        cluster_map_np = sitk.GetArrayFromImage(
            cluster_map_img).transpose(2, 1, 0)
    if distance_map_np is None:
        distance_map_np = sitk.GetArrayFromImage(
            distance_map_masked).transpose(2, 1, 0)

    end_set = np.asarray(end_clusters, dtype=np.int64)
    mask = np.isin(cluster_map_np, end_set)
    if not np.any(mask):
        return [None] * len(end_clusters)

    labs = cluster_map_np[mask].astype(np.int64, copy=False)
    dists = np.asarray(distance_map_np[mask], dtype=np.float64)
    z, y, x = np.where(mask)
    coords = np.column_stack((z, y, x))

    # Stable sort: primary label, descending distance, then ascending (z,y,x)
    # so the first row per label matches legacy (max distance, then first
    # argwhere index among ties).
    order = np.lexsort((x, y, z, -dists, labs))
    sorted_labs = labs[order]
    sorted_coords = coords[order]
    sorted_dists = dists[order]
    uniq, first = np.unique(sorted_labs, return_index=True)
    label_to_point = {
        int(lab): sorted_coords[idx] if sorted_dists[idx] > 0 else None
        for lab, idx in zip(uniq, first)
    }

    return [label_to_point.get(int(c), None) for c in end_clusters]


def calc_multi_component_centerlines(segmentation, nr_seeds=None,
                                   min_res=300, out_dir=None, write_files=False,
                                   move_target_if_fail=False, relax_factor=1,
                                   verbose=False, return_failed=False,
                                   post_process_kwargs=None):
    """
    Calculate centerlines for multi-component segmentations by creating seeds
    from disconnected bodies and computing centerlines for each component.
    
    This function:
    1. Analyzes the segmentation to find disconnected bodies
    2. Creates optimal seed points for each disconnected body
    3. Calculates centerlines using fast marching method for each seed
    4. Combines all centerlines into a unified polydata structure
    
    Parameters
    ----------
    segmentation : sitk.Image
        Binary segmentation image containing one or more disconnected bodies
    nr_seeds : int, optional
        Number of seeds to generate. If None, creates one seed per disconnected body.
        If the requested number exceeds the number of disconnected bodies, it will be
        automatically reduced to match the number of available bodies with a warning.
    min_res : int, optional
        Minimum resolution for centerline calculation. If segmentation resolution
        is lower, it will be resampled. Defaults to 300.
    out_dir : str, optional
        Output directory for intermediate files. Defaults to None.
    write_files : bool, optional
        Whether to write intermediate files to disk. Defaults to False.
    move_target_if_fail : bool, optional
        Whether to attempt moving target points if centerline calculation fails.
        Defaults to False.
    relax_factor : float, optional
        Factor to relax the tolerance for backtracking convergence. Defaults to 1.
    verbose : bool, optional
        If True, prints detailed information about the process. Defaults to False.
    return_failed : bool, optional
        Whether to include failed centerline attempts in the output. Defaults to False.
    post_process_kwargs : dict, optional
        Extra kwargs forwarded to :func:`calc_centerline_fmm` for centerline
        post-processing (e.g. ``{'merge_method': 'tree'}``).
        
    Returns
    -------
    unified_centerline : vtk.vtkPolyData
        Combined centerlines from all disconnected bodies as a single polydata
    success_info : dict
        Dictionary containing success information:
        - 'overall_success': bool, whether any centerlines were successfully calculated
        - 'component_successes': list of bool, success status for each component
        - 'num_components': int, total number of disconnected bodies found
        - 'num_successful': int, number of successful centerline calculations
        - 'seeds_used': list of np.array, seed points that were used
        
    Example
    -------
    >>> segmentation = sitk.ReadImage("multi_vessel.nii.gz")
    >>> centerline, info = calc_multi_component_centerlines(segmentation, verbose=True)
    >>> print(f"Calculated centerlines for {info['num_successful']}/{info['num_components']} components")
    >>> # Save unified centerline
    >>> from seqseg.modules.vtk_functions import write_geo
    >>> write_geo("unified_centerline.vtp", centerline)
    """
    if verbose:
        print("Starting multi-component centerline calculation...")
        print(f"Original segmentation size: {segmentation.GetSize()}")

    # Crop to the non-zero mask bounding box to reduce unnecessary
    # computation for sparse/full-volume segmentations.
    # Build a foreground mask in a pixel-type-agnostic way.
    # Using BinaryThreshold with sys.maxsize can overflow for certain input
    # pixel types and produce an invalid [lower, upper] range in ITK.
    seg_nonzero = sitk.Cast(sitk.NotEqual(segmentation, 0), sitk.sitkUInt8)
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(seg_nonzero)
    if not label_stats.HasLabel(1):
        if verbose:
            print("No foreground voxels found in segmentation")
        empty_polydata = vtk.vtkPolyData()
        return empty_polydata, {
            'overall_success': False,
            'component_successes': [],
            'num_components': 0,
            'num_successful': 0,
            'seeds_used': []
        }

    bbox = label_stats.GetBoundingBox(1)  # (x, y, z, size_x, size_y, size_z)
    roi_buffer_voxels = 10
    image_size = segmentation.GetSize()
    roi_index = []
    roi_size = []
    for dim in range(3):
        start = int(bbox[dim]) - roi_buffer_voxels
        end = int(bbox[dim] + bbox[dim + 3]) + roi_buffer_voxels
        start = max(0, start)
        end = min(int(image_size[dim]), end)
        roi_index.append(start)
        roi_size.append(end - start)
    segmentation = sitk.RegionOfInterest(segmentation, roi_size, roi_index)
    if verbose:
        print(
            "Cropped segmentation to ROI index "
            f"{tuple(roi_index)} size {tuple(roi_size)} "
            f"(buffer={roi_buffer_voxels} voxels)"
        )
        
    # Step 1: Create seeds from disconnected bodies
    if verbose:
        print("Step 1: Analyzing disconnected bodies and creating seeds...")
    
    seeds, component_info = create_seeds_from_disconnected_bodies(
        segmentation, nr_seeds=nr_seeds, verbose=verbose)
    
    # Write seeds as polydata file if output directory is specified
    if out_dir and write_files and component_info.get('seeds_polydata') is not None:
        seeds_filename = os.path.join(out_dir, 'multi_component_seeds.vtp')
        write_geo(seeds_filename, component_info['seeds_polydata'])
        if verbose:
            print(f"  Seeds written to: {seeds_filename}")
    
    if len(seeds) == 0:
        if verbose:
            print("No seeds could be created from the segmentation")
        # Return empty polydata
        empty_polydata = vtk.vtkPolyData()
        return empty_polydata, {
            'overall_success': False,
            'component_successes': [],
            'num_components': 0,
            'num_successful': 0,
            'seeds_used': []
        }
    
    if verbose:
        print(f"Step 2: Calculating centerlines for {len(seeds)} seeds...")
    
    # Step 2: Calculate centerlines for each seed
    centerlines = []
    success_list = []
    component_ratios = []
    component_target_counts = []  # Store actual target counts for each component
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"  Processing seed {i+1}/{len(seeds)}: {seed}")
            
        # Add seed validation and diagnostics
        try:
            # Validate seed is within segmentation bounds
            seed_index = segmentation.TransformPhysicalPointToIndex(seed.tolist())
            seg_size = segmentation.GetSize()
            
            if (seed_index[0] < 0 or seed_index[1] < 0 or seed_index[2] < 0 or
                seed_index[0] >= seg_size[0] or seed_index[1] >= seg_size[1] or seed_index[2] >= seg_size[2]):
                if verbose:
                    print(f"    Warning: Seed {i+1} is outside segmentation bounds")
                    print(f"      Seed index: {seed_index}, Segmentation size: {seg_size}")
                # Try to move seed inside bounds
                seed_index = [max(0, min(seed_index[0], seg_size[0]-1)),
                             max(0, min(seed_index[1], seg_size[1]-1)),
                             max(0, min(seed_index[2], seg_size[2]-1))]
                seed = np.array(segmentation.TransformIndexToPhysicalPoint(seed_index))
                if verbose:
                    print(f"      Adjusted seed to: {seed}, index: {seed_index}")
            
            # Check if seed is in segmentation mask
            seg_value = segmentation.GetPixel(seed_index)
            if seg_value == 0:
                if verbose:
                    print(f"    Warning: Seed {i+1} is outside segmentation mask (value={seg_value})")
                # Find nearest non-zero voxel
                seg_array = sitk.GetArrayFromImage(segmentation).transpose(2, 1, 0)
                nonzero_indices = np.argwhere(seg_array > 0)
                if len(nonzero_indices) > 0:
                    distances = np.linalg.norm(nonzero_indices - np.array(seed_index), axis=1)
                    nearest_idx = nonzero_indices[np.argmin(distances)]
                    seed = np.array(segmentation.TransformIndexToPhysicalPoint(nearest_idx.tolist()))
                    if verbose:
                        print(f"      Moved seed to nearest segmentation voxel: {seed}")
                else:
                    if verbose:
                        print(f"    Error: No non-zero voxels found in segmentation for seed {i+1}")
                    empty_centerline = vtk.vtkPolyData()
                    centerlines.append(empty_centerline)
                    success_list.append(False)
                    continue
            
            if verbose:
                print(f"    Seed {i+1} validation passed - proceeding with centerline calculation")
            
            # Extract disconnected body around seed
            segmentation_body = extract_disconnected_body_around_seed(
                segmentation, seed, verbose=verbose)
            
            # Calculate centerline using fast marching method
            centerline, success, target_success_list = calc_centerline_fmm(
                segmentation_body, 
                seed=seed,
                targets=None,  # Let calc_centerline_fmm find targets automatically
                min_res=min_res,
                out_dir=out_dir,
                write_files=write_files,
                move_target_if_fail=move_target_if_fail,
                relax_factor=relax_factor,
                verbose=verbose,
                return_failed=return_failed,
                return_success_list=True,
                post_process_kwargs=post_process_kwargs,
            )
            
            centerlines.append(centerline)
            success_list.append(success)
            
            # Store component success ratio and counts for detailed reporting
            if target_success_list and len(target_success_list) > 0:
                successful_targets = target_success_list.count(True)
                total_targets = len(target_success_list)
                component_success_ratio = successful_targets / total_targets
                component_target_counts.append((successful_targets, total_targets))
            else:
                component_success_ratio = 0.0
                component_target_counts.append((0, 1))  # Default to 0/1 for failed cases
            
            component_ratios.append(component_success_ratio)
            
            if verbose:
                if success:
                    num_points = centerline.GetNumberOfPoints()
                    num_cells = centerline.GetNumberOfCells()
                    print(f"    Success: {num_points} points, {num_cells} cells")
                    
                    # Additional diagnostics for successful centerlines
                    if centerline.GetPointData().GetArray("MaximumInscribedSphereRadius"):
                        radius_array = centerline.GetPointData().GetArray("MaximumInscribedSphereRadius")
                        radii = [radius_array.GetValue(i) for i in range(radius_array.GetNumberOfTuples())]
                        print(f"    Radius range: {min(radii):.3f} - {max(radii):.3f}")
                else:
                    print(f"    Failed to calculate centerline for seed {i+1}")
                    print(f"      This could be due to:")
                    print(f"        - Insufficient targets found from this seed")
                    print(f"        - Gradient backtracking failed")
                    print(f"        - Component too small or isolated")
                    
        except Exception as e:
            if verbose:
                print(f"    Error calculating centerline for seed {i+1}: {str(e)}")
                print(f"      Seed location: {seed}")
                import traceback
                print(f"      Full traceback: {traceback.format_exc()}")
            # Create empty polydata for failed case
            empty_centerline = vtk.vtkPolyData()
            centerlines.append(empty_centerline)
            success_list.append(False)
            component_ratios.append(0.0)  # Add 0.0 ratio for failed case
            component_target_counts.append((0, 1))  # Add 0/1 for failed case
    
    # Step 3: Combine all centerlines into unified polydata
    if verbose:
        successful_count = sum(success_list)
        print(f"Step 3: Combining {successful_count} successful centerlines...")
    
    # Filter out unsuccessful centerlines
    successful_centerlines = []
    successful_centerlines_count = 0
    
    for i, (centerline, success) in enumerate(zip(centerlines, success_list)):
        if success and centerline.GetNumberOfPoints() > 0:
            successful_centerlines.append(centerline)
            successful_centerlines_count += 1
    
    # Use appendPolyData from vtk_functions to combine all successful centerlines
    if successful_centerlines:
        unified_centerline = appendPolyData(successful_centerlines)
    else:
        # Create empty polydata if no successful centerlines
        unified_centerline = vtk.vtkPolyData()
    
    # Create success info
    success_info = {
        'overall_success': successful_centerlines_count > 0,
        'component_successes': success_list,
        'component_ratios': component_ratios,
        'component_target_counts': component_target_counts,
        'num_components': component_info['num_components'],
        'num_successful': successful_centerlines_count,
        'seeds_used': seeds
    }
    
    if verbose:
        print(f"Multi-component centerline calculation complete:")
        print(f"  Total components: {success_info['num_components']}")
        print(f"  Successful centerlines: {success_info['num_successful']}")
        print(f"  Final unified centerline: {unified_centerline.GetNumberOfPoints()} points, {unified_centerline.GetNumberOfCells()} cells")
    
    return unified_centerline, success_info


def test_centerline_fmm(directory, out_dir):
    """
    Function that tests the centerline calculation
    using the fast marching method.
    It loops through the segmentations .nii.gz files in the directory
    and calculates the centerline.
    And writes the centerlines as .vtp files.
    """
    # Get all files in directory
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]

    # Loop through all files
    for file in files[10:200]:
        print(f"\n\nCalculating centerline for: {file}\n\n")
        # Load segmentation
        segmentation = sitk.ReadImage(os.path.join(directory, file))
        # pfn = os.path.join(out_dir,
        #                    'segmentation_'+file.replace('.nii.gz', '.mha'))
        # sitk.WriteImage(segmentation, pfn)
        # Get surface mesh
        surface = evaluate_surface(segmentation)
        pfn = os.path.join(out_dir, 'surface_'+file.split('.')[0]+'.vtp')
        write_geo(pfn, surface)
        # Calculate caps
        caps = calc_caps(surface)
        print(f"  # Caps: {len(caps)}")
        # Calculate centerline
        centerline = calc_centerline_fmm(
            segmentation, caps[0],
            [cap for i, cap in enumerate(caps) if i != 0]
            )
        # Write centerline
        name = file.split('.')[0]
        pfn = os.path.join(out_dir, 'centerline_fm_'+name+'.vtp')
        write_geo(pfn, centerline)


if __name__ == '__main__':

    ###
    # NOTE: A bug exists where if the Direction of the segmentation
    # is not identity, the centerline calculation will fail.
    ###
    # Path to segmentation
    # path_segs = '/Users/nsveinsson/Documents/datasets/vmr/vmr_coronaries/ct/truths/'
    path_segs = '/Users/nsveinsson/Documents/datasets/airRC_dataset/truths/'

    # Output directory
    # out_dir = path_segs + '/centerlines_fmm_only_successful/'
    out_dir = path_segs.replace('truths','centerlines_fmm_test_airways')
    os.makedirs(out_dir, exist_ok=True)

    # Image extension
    img_ext = '.mha'

    # If make binary
    make_binary = False

    # Else choose label value to segment
    label_value = 1

    # If keep largest component
    keep_largest_component = True

    # If fill holes
    fill_holes = False

    # Verbose
    return_failed = False
    # Keep main output minimal: only write the final centerline file.
    write_files = True
    verbose = True

    # Start index
    start = 0
    stop = -1

    # If contains string, only process those files
    contains_str = ''

    # Path to spacing file
    if_spacing_file = False
    spacing_file = '/Users/nsveinsson/Documents/datasets/CAS_cerebral_dataset/CAS2023_trainingdataset/mm/meta.csv'

    # Path to end points
    if_end_points = False
    end_points_dir = '/Users/numisveins/Documents/datasets/CAS_dataset/CAS2023_trainingdataset/end_points/'

    # List of segmentations
    segs = [f for f in os.listdir(path_segs) if f.endswith(img_ext)]
    if contains_str:
        segs = [f for f in segs if contains_str in f]
    segs = sorted(segs)

    if stop == -1:
        stop = len(segs)

    if if_spacing_file:
        import pandas as pd
        spacing_df = pd.read_csv(spacing_file)
        # only keep 'spacing', they are sorted
        spacing_values = spacing_df['spacing'].values
        # read as tuples
        spacing_values = [tuple(map(float, x[1:-1].split(','))) for x in spacing_values]

    # Loop through all segmentations
    for seg in segs[start:stop]:

        print(f"\n\nCalculating centerline for: {seg}\n\n")
        path_seg = os.path.join(path_segs, seg)
        name = path_seg.split('/')[-1].split('.')[0]

        # skip if already done
        if os.path.exists(os.path.join(out_dir, 'done.txt')):
            with open(os.path.join(out_dir, 'done.txt'), 'r') as f:
                done_lines = f.read().splitlines()
                f.close()
            # Extract case names from lines (format: "name: success_info")
            done_names = [line.split(':')[0].strip() for line in done_lines if ':' in line]
            if name in done_names:
                print(f"Already done with: {seg}")
                continue
    
        # Load segmentation
        segmentation = sitk.ReadImage(path_seg)

        # Print img info
        print(f"Segmentation info:")
        print(f"  Size: {segmentation.GetSize()}")
        print(f"  Spacing: {segmentation.GetSpacing()}")
        print(f"  Origin: {segmentation.GetOrigin()}")
        print(f"  Direction: {segmentation.GetDirection()}")

        # Set Direction to identity
        segmentation.SetDirection((1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0))

        # Cast to uint8
        segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)

        # Make binary
        if make_binary:
            max_value = int(sitk.GetArrayFromImage(segmentation).max())
            print(f"Max value in segmentation: {max_value}")
            segmentation = sitk.BinaryThreshold(segmentation,
                                                lowerThreshold=1,
                                                upperThreshold=max_value,
                                                insideValue=1,
                                                outsideValue=0)
        else:
            # Threshold to label value
            segmentation = sitk.BinaryThreshold(segmentation,
                                                lowerThreshold=label_value,
                                                upperThreshold=label_value,
                                                insideValue=1,
                                                outsideValue=0)
        # Fill holes
        if fill_holes:
            holes = sitk.BinaryFillholeImageFilter()
            segmentation = holes.Execute(segmentation)

        # Keep largest component
        if keep_largest_component:
            connected = sitk.ConnectedComponentImageFilter()
            connected.SetFullyConnected(True)
            seg_cc = connected.Execute(segmentation)
            label_shape = sitk.LabelShapeStatisticsImageFilter()
            label_shape.Execute(seg_cc)
            largest_label = 0
            largest_size = 0
            for label in label_shape.GetLabels():
                size = label_shape.GetNumberOfPixels(label)
                if size > largest_size:
                    largest_size = size
                    largest_label = label
            print(f"Largest component label: {largest_label}, size: {largest_size}")
            segmentation = sitk.BinaryThreshold(seg_cc,
                                                lowerThreshold=largest_label,
                                                upperThreshold=largest_label,
                                                insideValue=1,
                                                outsideValue=0)
        
        if if_spacing_file:
            # set the spacing
            segmentation.SetSpacing(spacing_values[segs.index(seg)])
        time_start = time.time()
        # Get end points
        if if_end_points:
            end_points = np.load(os.path.join(end_points_dir, name+'.npy'))
            targets = []
            for i in range(len(end_points)):
                targets.append(end_points[i])
        else:
            targets = None
        # Calculate centerline
        # centerline, success_overall, targets = calc_centerline_fmm(
        #     segmentation,
        #     out_dir=out_dir,
        #     write_files=write_files,
        #     seed=None,
        #     targets=targets,
        #     move_target_if_fail=False,
        #     return_failed=True,
        #     return_target_all=True,
        #     return_target=False,
        #     verbose=verbose,
        #     # min_res=500,
        #     )
        
        # Calculate multi-component centerlines
        centerline_multi, success_info = calc_multi_component_centerlines(
            segmentation,
            nr_seeds=None,
            min_res=700,
            out_dir=out_dir,
            write_files=write_files,
            move_target_if_fail=False,
            relax_factor=1,
            verbose=verbose,
            return_failed=return_failed,
            post_process_kwargs={'merge_method': 'tree'}
        )

        print(f"Time in seconds: {time.time() - time_start:0.3f}")
        # pfn = os.path.join(out_dir, name+'.vtp')
        # write_geo(pfn, centerline)
        # print(f"Centerline written to: {pfn}")
        # print(f"Success: {success_overall}")

        pfn = os.path.join(out_dir, name+'_multi.vtp')
        write_geo(pfn, centerline_multi)
        print(f"Multi-component centerline written to: {pfn}")
        print(f"Success info: {success_info}")

        # Calculate success ratio
        success_ratio = success_info['num_successful'] / success_info['num_components'] if success_info['num_components'] > 0 else 0.0
        
        # Create component success info with ratios as fractions
        component_details = []
        for i, component_success in enumerate(success_info['component_successes']):
            # Get component target counts if available
            if 'component_target_counts' in success_info and i < len(success_info['component_target_counts']):
                successful, total = success_info['component_target_counts'][i]
                component_details.append(f"C{i+1}:{successful}/{total}")
            else:
                # Fallback to binary status
                status = "✓" if component_success else "✗"
                component_details.append(f"C{i+1}:{status}")
        component_detail_str = " ".join(component_details)
        
        # write to done.txt with success ratio
        with open(os.path.join(out_dir, 'done.txt'), 'a') as f:
            f.write(f"{name}: {success_info['num_successful']}/{success_info['num_components']} ({success_ratio:.3f}) [{component_detail_str}]\n")
            f.close()
        print(f"Done with: {name} - Success ratio: {success_ratio:.3f} - Components: {component_detail_str}")
