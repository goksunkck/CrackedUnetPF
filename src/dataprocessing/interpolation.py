import numpy as np
from scipy.interpolate import LinearNDInterpolator
from multiprocessing import Pool, cpu_count
from functools import partial


def _interpolate_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing."""
    key, input_obj, interp_size, pixels, offset = args
    print(f'\r- Processing {key}...', end='')
    res = interpolate(input_obj, interp_size, offset=offset, pixels=pixels)
    return key, res


def interpolate_on_array(input_by_nodemap, interp_size, offset=0, pixels=256):
    """
    Extracts Coors, Disps and EpsVonMises from dictionary of input data objects.
    Interpolates these quantities onto an array of a certain pixel size by calling
    the function InterpOnArray for each entry in dictionary

    :param input_by_nodemap: (dict of input_data_object objects)
    :param interp_size: (int or float) indicates the edge length of field of view.
    This field of view is located at 0 < x < size and -size/2 < y < size/2
    :param offset: (int or float) indicates the x-value offset (can be negative)
    :param pixels: (int) indicates the edge length of field of view in pixels

    :return: interp_coors_by_nodemap, interp_disps_by_nodemap, interp_eps_VM_by_nodemap
             (dicts) of Coors, Disps and EpsVonMises interpolated on arrays
    """
    print('\rData will be interpolated on an array using multiprocessing...')

    interp_coors_by_nodemap = {}
    interp_disps_by_nodemap = {}
    interp_eps_vm_by_nodemap = {}

    # Prepare arguments for parallel processing
    tasks = [
        (key, input_by_nodemap[key], interp_size, pixels, offset)
        for key in input_by_nodemap
    ]

    # Use all available CPUs
    num_processes = cpu_count()
    print(f'Starting pool with {num_processes} processes for {len(tasks)} tasks.')

    with Pool(processes=num_processes) as pool:
        results = pool.map(_interpolate_wrapper, tasks)

    print('\nInterpolation complete. Aggregating results...')

    for key, (interp_coors, interp_disps, interp_eps_vm) in results:
        interp_coors_by_nodemap[key] = interp_coors
        interp_disps_by_nodemap[key] = interp_disps
        interp_eps_vm_by_nodemap[key] = interp_eps_vm

    print('Results aggregated.\n')

    return interp_coors_by_nodemap, interp_disps_by_nodemap, interp_eps_vm_by_nodemap


def interpolate(input_data_object, size, offset=0, pixels=256):
    """Extracts and interpolates Coors, Disps and EpsVonMises
    from an InputData object onto arrays of size pixels x pixels.

    :param input_data_object: (InputData object) This object gives access to all DIC data for a given nodemap.
        This can be accessed by e.g. InputData.dXVec for the X-Vector of DIC data
    :param size: (int) indicates the edge length of field of view.
        This field of view is located at 0 < x < size and -size/2 < y < size/2
    :param offset: (int or float) indicates an x-value offset (can be negative)
    :param pixels: (int) indicates the edge length of field of view in pixels

    :return: Arrays interp_coors, interp_disps, interp_eps_vm all with a shape (pixels, pixels)
        giving coordinates, displacements and Von-Mises Strains on each pixel in the array
    """
    if not isinstance(size, (int, float)):
        raise TypeError(
            'Argument "size" should be an integer, indicating the edge length of field of view.\n'
            'This field of view is located at 0 < x < size and -size/2 < y < size/2')
    if not isinstance(pixels, int):
        raise TypeError(
            'Argument "pixels" should be an integer, indicating the edge length of field of view '
            'in pixels.\n Default is 256')

    x_coordinate = input_data_object.coor_x
    y_coordinate = input_data_object.coor_y

    x_displacement = input_data_object.disp_x
    y_displacement = input_data_object.disp_y

    eps_vm = input_data_object.eps_vm

    if size >= 0:
        x_coor_interp = np.linspace(offset, size + offset, pixels)
        y_coor_interp = np.linspace(-size / 2.0, size / 2.0, pixels)
    else:
        x_coor_interp = np.linspace(size + offset, offset, pixels)
        y_coor_interp = np.linspace(size / 2.0, -size / 2.0, pixels)

    x_grid, y_grid = np.meshgrid(x_coor_interp, y_coor_interp)

    # Optimization: Use LinearNDInterpolator on all fields at once to compute triangulation only once
    points = np.column_stack((x_coordinate, y_coordinate))
    values = np.column_stack((x_displacement, y_displacement, eps_vm))
    
    # Create interpolator
    interp = LinearNDInterpolator(points, values)
    
    # Query points
    xi = np.stack((x_grid, y_grid), axis=-1)
    
    # Interpolate
    interpolated_values = interp(xi)
    
    x_disp_interp = interpolated_values[:, :, 0]
    y_disp_interp = interpolated_values[:, :, 1]
    interp_eps_vm = interpolated_values[:, :, 2]

    # If size is negative, then the left-hand side of the specimen is flipped such that the crack
    # always starts from the left and the x-displacement is multiplied by -1
    if size < 0:
        x_disp_interp = np.fliplr(x_disp_interp) * -1
        y_disp_interp = np.fliplr(y_disp_interp)
        interp_eps_vm = np.fliplr(interp_eps_vm)
        x_grid = np.fliplr(x_grid)
        y_grid = np.fliplr(y_grid)

    interp_coors = np.asarray([x_grid, y_grid])
    interp_disps = np.asarray([x_disp_interp, y_disp_interp])

    return interp_coors, interp_disps, interp_eps_vm
