# Code to extract centerlines from surface models

from vmtk import vmtkscripts
import numpy as np

# Function that takes in surface file and reads it


def read_surface(filename):

    surfaceReader = vmtkscripts.vmtkSurfaceReader()
    surfaceReader.InputFileName = filename
    surfaceReader.Execute()

    return surfaceReader.Surface

# Functon that calculates centerlines


def calc_network(Surface):

    centerline_calc = vmtkscripts.vmtkNetworkExtraction()
    centerline_calc.Surface = Surface
    centerline_calc.Execute()

    return centerline_calc.Network

# Function that takes in centerlines, writes a file


def write_network(Network, filename):

    print("Writing centerline to: " + filename)
    cent_write = vmtkscripts.vmtkNetworkWriter()
    cent_write.Network = Network
    cent_write.OutputFileName = filename
    cent_write.Execute()

    return


def network(filename_in, filename_out):

    Surface = read_surface(filename_in)
    Network = calc_network(Surface)
    write_network(Network, filename_out)

    return Network


def calc_centerline(Surface, method, var_source=None, var_target=None,
                    number=None, caps=None, point=None):
    """
    Calculate centerlines in surface model via vmtk
    Methods "pickpoint","openprofiles","carotidprofiles" don't require inputs
    Args:
        Surface: VTK PolyData of surface
        method: one of
            ["pickpoint","openprofiles","carotidprofiles","profileidlist","idlist","pointlist"]
        var_source: list of ids (int) or points (float) for vascular inlets
        var_target: list of ids (int) or points (float) for outlets
    Returns:
        poly: VTK PolyData of centerline
    """

    centerline_calc = vmtkscripts.vmtkCenterlines()
    centerline_calc.Surface = Surface

    if number <= 1 and len(caps) == 1:
        method = "pointlist"
        var_target = caps[0].tolist()
        var_source = point.tolist()
        centerline_calc.SourcePoints = var_source
        centerline_calc.TargetPoints = var_target
    else:
        # import pdb; pdb.set_trace()
        if method == "profileidlist":
            centerline_calc.AppendEndPoints = 0
            if var_source is None:
                var_source = [0]
            centerline_calc.SourceIds = var_source
            # centerline_calc.TargetIds = None #var_target

        # use current point and match to caps
        elif method == "pointlist":
            # create a list of coords
            target = []
            for i in range(len(caps)):
                # if i == var_source: continue # skip source cap
                target = target + caps[i].tolist()
            centerline_calc.SourcePoints = point.tolist()  # var_source
            centerline_calc.TargetPoints = target  # var_target

    print(f"Method is {method}")
    centerline_calc.SeedSelectorName = method
    centerline_calc.Execute()
    print("Centerline Calc Executed")
    # print(centerline_calc.Centerlines)

    return centerline_calc.Centerlines


def resample_centerline(Centerlines):
    """
    Function to resample centerline
    """
    centerline_calc = vmtkscripts.vmtkCenterlineResampling()
    centerline_calc.Centerlines = Centerlines
    centerline_calc.Execute()
    print('Centerline Resampler Executed')

    return centerline_calc.Centerlines


def smooth_centerline(Centerlines):
    """
    Function to resample centerline
    """
    centerline_calc = vmtkscripts.vmtkCenterlineSmoothing()
    centerline_calc.Centerlines = Centerlines
    centerline_calc.Execute()
    print('Centerline Smoother Executed')

    return centerline_calc.Centerlines


def calc_branches(Centerlines):

    if Centerlines.GetNumberOfPoints() != 0:
        print(f'Centerline has: {Centerlines.GetNumberOfPoints()}, points',
              flush=True)
        calc_branch = vmtkscripts.vmtkBranchExtractor()
        calc_branch.Centerlines = Centerlines
        calc_branch.Execute()
    else:
        print("0 points in centerline")
        # cause error
        # print(error)
    print('Branch Extractor Executed')

    return calc_branch.Centerlines


def write_centerline(Centerline, fd_out):

    cent_write = vmtkscripts.vmtkSurfaceWriter()
    cent_write.Surface = Centerline
    cent_write.OutputFileName = fd_out
    cent_write.Execute()


def cent2numpy(Centerline):
    cent_np = vmtkscripts.vmtkCenterlinesToNumpy()
    cent_np.Centerlines = Centerline
    cent_np.Execute()
    cent_np = cent_np.ArrayDict

    return cent_np


def centerline(fd_in, fd_out, method, var_source=None, var_target=None):

    Surface = read_surface(fd_in)
    Centerline = calc_centerline(Surface, method, var_source, var_target)

    cent_np = vmtkscripts.vmtkCenterlinesToNumpy()
    cent_np.Centerlines = Centerline
    cent_np.Execute()
    cent_np = cent_np.ArrayDict

    if cent_np['Points'].size != 1:
        write_centerline(Centerline, fd_out)
    else:
        print("*********** Error: Did not write centerline ***********")
        Centerline = None

    return Centerline, cent_np


def get_surface_caps(Surface, method):
    """
    Calculate caps of surface model via vmtk
    Methods "pickpoint","openprofiles","carotidprofiles" don't require inputs
    Args:
        Surface: VTK PolyData of surface
        method: one of
            ["simple","centerpoint","smooth","annular","concaveannular"]
    Returns:
        Surface: VTK PolyData of surface, with array of cap IDs
    """
    calc_caps = vmtkscripts.vmtkSurfaceCapper()
    calc_caps.Method = method
    calc_caps.Surface = Surface
    calc_caps.Interactive = 0
    calc_caps.CellEntityIdsArrayName = "CapID"
    calc_caps.Execute()
    surface_caps = calc_caps.Surface

    surf_np = vmtkscripts.vmtkSurfaceToNumpy()
    surf_np.Surface = surface_caps
    surf_np.Execute()
    surf_np = surf_np.ArrayDict

    print('Number of cells: ', len(surf_np['CellData']['CapID']))
    print('Number of points: ', len(surf_np['Points']))
    print(len(np.where(surf_np['CellData']['CapID'] == 1)[0]))
    print(surf_np['CellData']['CapID'].max())
    print(surf_np['CellData']['CapID'].min())

    return surface_caps, surf_np


def smooth_surface_vmtk(polydata, iterations, parameter, method=None):
    """
    Smooths a surface model via vmtk
    Args:
        polydata: VTK PolyData of surface
        method: one of
            ["taubin","laplace"]
        iterations: number of smoothing iterations
        parameter: either
            passaband for taubin
            relaxation factor for laplace
    Returns:
        Surface: VTK PolyData of surface, with array of cap IDs
    """
    n = iterations
    if method == 'taubin' or method is None:
        smoother = vmtkscripts.vmtkSurfaceSmoothing()
        smoother.Surface = polydata
        smoother.iterations = n
        smoother.passband = parameter
    elif method == 'laplace':
        smoother = vmtkscripts.vmtkSurfaceSmoothing()
        smoother.Surface = polydata
        smoother.Method = 'laplace'
        smoother.iterations = n
        smoother.relaxation = parameter
    else:
        print('VMTK smoothing method not supported')

    smoother.Execute()
    return smoother.Surface


if __name__ == '__main__':

    sample = 'smooth_mc_0146_1001_10.vtp'

    filename_in = '/Users/numisveinsson/Documents/Side_SV_projects/' + \
                  'SV_ML_Training/3d_ml_data/test4_assume_centerlines' + \
                  '/mc_surfaces/'+sample
    filename_out = '/Users/numisveinsson/Downloads/network_' + sample

    Network = network(filename_in, filename_out)

    fd_out = '/Users/numisveinsson/Downloads/centerline_' + sample

    Centerline = centerline(filename_in, fd_out)

    # Some demo code

    centerlineReader = vmtkscripts.vmtkSurfaceReader()
    centerlineReader.InputFileName = '/Users/numisveinsson/Downloads' + \
                                     '/0065_1001_53_centerlines.vtp'
    centerlineReader.Execute()
    clNumpyAdaptor = vmtkscripts.vmtkCenterlinesToNumpy()
    clNumpyAdaptor.Centerlines = centerlineReader.Surface
    clNumpyAdaptor.Execute()
    numpyCenterlines = clNumpyAdaptor.ArrayDict

    from vmtk import pypes
    myArguments = 'vmtkmarchingcubes -ifile myimage.vti' + \
                  ' -l 800 --pipe vmtksurfaceviewer'
    myPype = pypes.PypeRun(myArguments)
    mySurface = myPype.GetScriptObject('vmtkmarchingcubes', '0').Surface
