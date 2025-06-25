# TODO: Implement centerline calculation using pathfinding and exploration
import sys
import os
import time
import vtk
import SimpleITK as sitk
import numpy as np
# from numba import jit
sys.path.insert(0, './')
from seqseg.modules.sitk_functions import create_new, distance_map_from_seg
from seqseg.modules.vtk_functions import (calc_caps, evaluate_surface, 
                                   points2polydata, write_geo)

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
    print(f"   Backtracking from {target} to {seed}")
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
        # print(f"Gradient norm: {np.linalg.norm(gradient_current)}")
        # print(f"Gradient: {gradient_current}")
        # Normalize gradient
        gradient_current = gradient_current / np.linalg.norm(gradient_current)
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
                        verbose=False, return_failed=False):
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

    Returns
    -------
    centerline : vtkPolyData
        Centerline of the vessel.
    """
    output = None
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
        print(f"    New size: {segmentation.GetSize()}")
    # Create distance map
    distance_map_surf = distance_map_from_seg(segmentation)
    # Switch sign of distance map
    distance_map_surf = distance_map_surf * -1
    distance_map_surf_np = sitk.GetArrayFromImage(
        distance_map_surf).transpose(2, 1, 0)
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
                                            write_files=write_files)
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
                                         write_files=write_files)
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
        seed_np = seed
        seed = list(segmentation.TransformPhysicalPointToIndex(seed.tolist()))
        # if any are outside or on boundary, move inside
        seed = check_border(seed, segmentation.GetSize())
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
        sitk.WriteImage(output_mask,
                        os.path.join(out_dir, 'masked_out_fmm.mha'))
    # Get gradient of distance map
    gradient = gradient_matrix(
        sitk.GetArrayFromImage(output).transpose(2, 1, 0))
    if verbose:
        print("Gradient calculated")

    points_list, success_list = [], []
    for t_num, target_np in enumerate(targets_np):
        if verbose:
            print(f"        Starting target {t_num+1}/{len(targets_np)}")
        # Calculate path from seed to target
        points, success = backtracking_gradient(gradient,
                                                distance_map_surf_np,
                                                segmentation,
                                                seed_np, target_np,
                                                relax_factor=relax_factor)
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
                                                        relax_factor=relax_factor)
        # Add points to list
        points_list.append(points)
        success_list.append(success)

    # Create vtk polydata for points
    centerline = create_centerline_polydata(points_list,
                                            success_list,
                                            distance_map_surf,
                                            return_failed=return_failed)
    centerline = post_process_centerline(centerline)

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
    # else return only centerline and success
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


def post_process_centerline(centerline, verbose=False):
    """
    Function to post process the centerline using vtk functionalities.

    1. Remove duplicate points.
    2. Smooth centerline.

    Parameters
    ----------
    centerline : vtkPolyData
        Centerline of the vessel.

    Returns
    -------
    centerline : vtkPolyData
    """
    if verbose:
        print(f"""Number of points before post processing:
              {centerline.GetNumberOfPoints()}""")
    # Remove duplicate points
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(centerline)
    cleaner.SetTolerance(0.001)
    cleaner.Update()
    centerline = cleaner.GetOutput()

    # Smooth centerline
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(centerline)
    smoother.SetNumberOfIterations(15)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(0.001)
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

    Clusters are connected if they share a face.

    We find the end clusters by iterating over all clusters and checking
    neighboring voxels and which cluster they belong to.
    If neighboring voxels only belong to one other cluster
    then the current cluster is an end cluster.

    Parameters
    ----------
    cluster_map_img : image np array
        Cluster map. Each cluster has a unique integer value.

    Returns
    -------
    end_clusters : list of int
    """
    # Get unique values
    unique_values = np.unique(cluster_map)
    # Initialize end clusters
    end_clusters = []
    # Iterate over all clusters
    for value in unique_values:
        if value == 0:
            continue
        # Get indices of current cluster
        indices = np.argwhere(cluster_map == value)
        # Initialize number of connections
        connections = 0
        values_connected = []
        # Iterate over all indices
        for index in indices:
            # Get neighboring indices
            neighbors = get_neighbors(index, cluster_map.shape)
            # Iterate over all neighbors
            connections, values_connected = check_connections(value,
                                                              connections,
                                                              values_connected,
                                                              neighbors,
                                                              cluster_map)
            # If more than one connection, break
            if connections > 1:
                break
        # If only one connection, add to end clusters
        if connections == 1:
            # print(f"Cluster {value} is an end cluster")
            end_clusters.append(value)

    return end_clusters


def cluster_map(segmentation, return_wave_distance_map=False,
                out_dir=None, write_files=False):
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
    # If multiple max values, take first
    if len(index[0]) > 1:
        index = [index[0][0], index[1][0], index[2][0]]
    # Create fast marching filter
    fast_marching = sitk.FastMarchingImageFilter()
    fast_marching.AddTrialPoint((int(index[0]), int(index[1]), int(index[2])))
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
        sitk.WriteImage(wave_distance_map,
                        os.path.join(out_dir, 'wave_distance_map.mha'))

    # Get unique values in wave distance map
    unique_values = np.unique(
        sitk.GetArrayFromImage(wave_distance_map).transpose(2, 1, 0))
    print(f"Unique values: {unique_values}")
    # Remove values above threshold 100 thousand
    unique_values = unique_values[unique_values < 100000]
    # Create new image with same size as wave distance map
    cluster_map_img = sitk.Image(wave_distance_map.GetSize(), sitk.sitkInt32)
    cluster_map_img.CopyInformation(wave_distance_map)

    time_start = time.time()
    # For each unique value, create a cluster
    # Group every N values
    if np.max(unique_values) > 100:
        N = 10
    elif np.max(unique_values) > 50:
        N = 5
    elif np.max(unique_values) > 20:
        N = 3
    elif np.max(unique_values) > 10:
        N = 2
    else:
        N = 1
    cluster_count = 1
    for i, value in enumerate(range(1, np.max(unique_values)+1, N)):
        if value == 0:
            continue
        # Create mask for value
        mask = sitk.BinaryThreshold(wave_distance_map,
                                    lowerThreshold=value,
                                    upperThreshold=(value+N-1))
        print(f"Values: {value} to {value+N-1}")
        # Connected components
        connected = sitk.ConnectedComponentImageFilter()
        # Fully connected to include diagonal connections
        connected.SetFullyConnected(True)
        connected = connected.Execute(mask)
        # Get number of connected components
        num_connected = np.max(
            sitk.GetArrayFromImage(connected).transpose(2, 1, 0))
        print(f"   Num connected: {num_connected}")
        # Assign unique value to each connected component
        for j in range(1, num_connected+1):
            mask = sitk.BinaryThreshold(connected,
                                        lowerThreshold=j,
                                        upperThreshold=j)
            # if cluster is too small, ignore
            # print(f"Cluster has size:{np.sum(sitk.GetArrayFromImage(mask))}")
            if np.sum(sitk.GetArrayFromImage(mask).transpose(2, 1, 0)) < 5:
                continue
            cluster_map_img = sitk.Mask(cluster_map_img, sitk.Not(mask))
            mask = sitk.Cast(mask, sitk.sitkInt32) * (cluster_count)
            cluster_count += 1
            cluster_map_img = cluster_map_img + mask

    print(f"Cluster count: {cluster_count}")
    print(f"Time to create cluster map: {time.time() - time_start:0.2f}")

    # Write image
    if out_dir and write_files:
        sitk.WriteImage(cluster_map_img, os.path.join(out_dir,
                                                      'cluster_map_img.mha'))

    time_start = time.time()
    end_clusters = find_end_clusters(sitk.GetArrayFromImage(
        cluster_map_img).transpose(2, 1, 0))
    print(f"Time to find end clusters: {time.time() - time_start:0.2f}")
    print(f"End clusters: {end_clusters}")
    print(f"Number of end clusters: {len(end_clusters)}")

    time_start = time.time()
    end_points = get_end_points(cluster_map_img,
                                end_clusters,
                                distance_map_masked)
    print(f"Time to get end points: {time.time() - time_start:0.2f}")

    # Convert end points to physical points
    end_points_phys = [segmentation.TransformIndexToPhysicalPoint(
        end_point.tolist()) for end_point in end_points]

    # Create vtk polydata for points
    if out_dir and write_files:
        polydata_point = points2polydata(end_points_phys)
        write_geo(os.path.join(out_dir, 'end_points.vtp'), polydata_point)

    seed_np = np.array(
        segmentation.TransformIndexToPhysicalPoint((int(index[0]),
                                                    int(index[1]),
                                                    int(index[2]))))
    end_points_phys_np = [np.array(point) for point in end_points_phys]

    # write seed and end points as npy
    # np.save('/Users/numisveins/Downloads/debug_centerline/seed.npy',
    #         seed_np)
    # np.save('/Users/numisveins/Downloads/debug_centerline/end_points.npy',
    #         end_points_phys_np)

    if return_wave_distance_map:

        return seed_np, end_points_phys_np, wave_distance_map_output

    return seed_np, end_points_phys_np


def get_end_points(cluster_map_img, end_clusters, distance_map_masked):
    """
    Function to get the end points of a cluster map.

    Parameters
    ----------
    cluster_map_img : sitk image
        Cluster map.
    end_clusters : list of int
        End clusters.
    distance_map_masked : sitk image
        Distance map masked with the segmentation.

    Returns
    -------
    end_points : list of np.array
    """
    # Get array from cluster map
    cluster_map = sitk.GetArrayFromImage(
        cluster_map_img).transpose(2, 1, 0)
    # Get array from distance map
    distance_map = sitk.GetArrayFromImage(
        distance_map_masked).transpose(2, 1, 0)
    # Initialize end points
    end_points = []
    # Iterate over all end clusters
    for end_cluster in end_clusters:
        # Get indices of current cluster
        indices = np.argwhere(cluster_map == end_cluster)
        # Initialize maximum distance
        max_distance = 0
        # Initialize end point
        end_point = None
        # Iterate over all indices
        for index in indices:
            # Get distance at current index
            distance = distance_map[tuple(index)]
            # If distance is greater than maximum distance
            if distance > max_distance:
                # Update maximum distance
                max_distance = distance
                # Update end point
                end_point = index
        # Add end point to list
        end_points.append(end_point)
        # print(f"End point: {end_point}, Max distance: {max_distance}")

    return end_points


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

    # Out directory
    out_dir = '/global/scratch/users/numi/CAS_dataset/CAS2023_trainingdataset/centerlines_fmm/'

    # Path to segmentation
    path_segs = '/Users/numisveins/Documents/vascular_data_3d/truths/'
    path_segs = '/global/scratch/users/numi/CAS_dataset/CAS2023_trainingdataset/truths/'

    # Start index
    start = 0

    # Path to spacing file
    if_spacing_file = True
    spacing_file = '/global/scratch/users/numi/CAS_dataset/CAS2023_trainingdataset/meta.csv'

    # Path to end points
    if_end_points = False
    end_points_dir = '/Users/numisveins/Documents/datasets/CAS_dataset/CAS2023_trainingdataset/end_points/'

    # List of segmentations
    segs = [f for f in os.listdir(path_segs) if f.endswith('.mha')]
    segs = sorted(segs)

    if if_spacing_file:
        import pandas as pd
        spacing_df = pd.read_csv(spacing_file)
        # only keep 'spacing', they are sorted
        spacing_values = spacing_df['spacing'].values
        # read as tuples
        spacing_values = [tuple(map(float, x[1:-1].split(','))) for x in spacing_values]

    # Loop through all segmentations
    for seg in segs[start:]:

        print(f"\n\nCalculating centerline for: {seg}\n\n")
        path_seg = os.path.join(path_segs, seg)
        name = path_seg.split('/')[-1].split('.')[0]

        # skip if already done
        if os.path.exists(os.path.join(out_dir, 'done.txt')):
            with open(os.path.join(out_dir, 'done.txt'), 'r') as f:
                done = f.read().splitlines()
                f.close()
            if name in done:
                print(f"Already done with: {seg}")
                continue
    
        # Load segmentation
        segmentation = sitk.ReadImage(path_seg)

        if if_spacing_file:
            # set the spacing
            segmentation.SetSpacing(spacing_values[segs.index(seg)])
        # Write segmentation
        sitk.WriteImage(segmentation, os.path.join(out_dir.replace('images_w', 'truths'),
                        name+'.mha'))

        time_start = time.time()
        # Get end points
        if if_end_points:
            end_points = np.load(os.path.join(end_points_dir, name+'.npy'))
            # print as polydata
            polydata_point = points2polydata(end_points)
            write_geo(os.path.join(out_dir, 'end_points_'+name+'.vtp'),
                      polydata_point)
            targets = []
            for i in range(len(end_points)):
                targets.append(end_points[i])
        else:
            targets = None
        # Calculate centerline
        centerline, success_overall = calc_centerline_fmm(
            segmentation,
            out_dir=out_dir,
            write_files=False,
            seed=None,
            targets=targets,
            move_target_if_fail=False,
            )

        print(f"Time in seconds: {time.time() - time_start:0.3f}")
        pfn = os.path.join(out_dir, name+'.vtp')
        write_geo(pfn, centerline)
        print(f"Centerline written to: {pfn}")
        print(f"Success: {success_overall}")

        # write to done.txt
        with open(os.path.join(out_dir, 'done.txt'), 'a') as f:
            f.write(name+'\n')
            f.close()
        print(f"Done with: {name}")
