import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import glob
import numpy as np
import vtk
from vtk_utils.vtk_utils import *
import matplotlib.pyplot as plt
import argparse

def plot_dice_scores_over_epochs(dir_n):
    classes = ['WH', 'LV', 'RV', 'LA', 'RA', 'Myo', 'Ao', 'PA']
    csv_fns = glob.glob(os.path.join(dir_n, '**', 'dice.csv'))
    # compute total dice scores first
    dice_list, epoch_list = [], []
    for fn in csv_fns:
        epoch = int(os.path.basename(os.path.dirname(fn)).split('_')[-1])
        epoch_list.append(epoch)
        data = np.loadtxt(fn, delimiter=',', skiprows=1)
        mean_dice = np.mean(data, axis=0)
        dice_list.append(mean_dice)
    sortid = np.argsort(epoch_list)
    epoch_list = [epoch_list[i] for i in sortid]
    dice_list = [dice_list[i] for i in sortid]
    dice_list = np.array(dice_list)
    plt.figure()
    for i in range(dice_list.shape[-1]):
        plt.plot(epoch_list, dice_list[:, i], label=classes[i], linewidth=2)
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.ylim([0., 0.9])
    plt.show()

def setupLuts(name='cool_to_warm'):
    luts = []
    if name == 'blue_to_red':
        # HSV (Blue to REd)  Default
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        lut.SetNumberOfColors(256)
        lut.Build()
    elif name =='cool_to_warm':
        # Diverging (Cool to Warm) color scheme
        ctf = vtk.vtkColorTransferFunction()
        ctf.SetColorSpaceToDiverging()
        ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
        ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)
        cc = list()
        for i in range(256):
          cc.append(ctf.GetColor(float(i) / 255.0))
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        for i, item in enumerate(cc):
          lut.SetTableValue(i, item[0], item[1], item[2], 1.0)
        lut.Build()
    elif name =='shock':
        # Shock
        ctf = vtk.vtkColorTransferFunction()
        min = 93698.4
        max = 230532
        ctf.AddRGBPoint(self._normalize(min, max,  93698.4),  0.0,         0.0,      1.0)
        ctf.AddRGBPoint(self._normalize(min, max, 115592.0),  0.0,         0.905882, 1.0)
        ctf.AddRGBPoint(self._normalize(min, max, 138853.0),  0.0941176,   0.733333, 0.027451)
        ctf.AddRGBPoint(self._normalize(min, max, 159378.0),  1.0,         0.913725, 0.00784314)
        ctf.AddRGBPoint(self._normalize(min, max, 181272.0),  1.0,         0.180392, 0.239216)
        ctf.AddRGBPoint(self._normalize(min, max, 203165.0),  1.0,         0.701961, 0.960784)
        ctf.AddRGBPoint(self._normalize(min, max, 230532.0),  1.0,         1.0,      1.0)
        cc = list()
        for i in xrange(256):
          cc.append(ctf.GetColor(float(i) / 255.0))
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        for i, item in enumerate(cc):
          lut.SetTableValue(i, item[0], item[1], item[2], 1.0)
        lut.Build()
    elif name =='set3':
        colorSeries = vtk.vtkColorSeries()
        # Select a color scheme.
        #colorSeriesEnum = colorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_9
        # colorSeriesEnum = colorSeries.BREWER_DIVERGING_SPECTRAL_10
        # colorSeriesEnum = colorSeries.BREWER_DIVERGING_SPECTRAL_3
        # colorSeriesEnum = colorSeries.BREWER_DIVERGING_PURPLE_ORANGE_9
        # colorSeriesEnum = colorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_9
        # colorSeriesEnum = colorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_9
        colorSeriesEnum = colorSeries.BREWER_QUALITATIVE_SET3
        # colorSeriesEnum = colorSeries.CITRUS
        colorSeries.SetColorScheme(colorSeriesEnum)
        lut = vtk.vtkLookupTable()
        #lut.SetNumberOfColors(20)
        colorSeries.BuildLookupTable(lut, vtk.vtkColorSeries.CATEGORICAL)
        #colorSeries.BuildLookupTable(lut, vtk.vtkColorSeries.ORDINAL)
        lut.SetNanColor(1, 0, 0, 1)
    else:
        raise NotImplementedError('Requested color scheme not implemented.')
    return lut

def visualize_template_over_epochs(dir_n):
    pass

def visualize_template_vasc(mesh_fn, array_name='CapID', threshold_name='CapID', show_edge=False, id_to_visualize=-1, tilt_factor=0., digit_size=256, camera=None, gt=False):
    mesh = load_vtk_mesh(mesh_fn)
    mesh.GetPointData().RemoveArray('Normals_')
    com_filter = vtk.vtkCenterOfMass()
    com_filter.SetInputData(mesh)
    com_filter.Update()
    center = com_filter.GetCenter()

    # poly_la = thresholdPolyData(mesh, threshold_name, (2, 2),'point')
    # poly_lv = thresholdPolyData(mesh, threshold_name, (0, 0),'point')
    # poly_rv = thresholdPolyData(mesh, threshold_name, (1, 1),'point')
    pdb.set_trace()
    view_up = np.mean(vtk_to_numpy(mesh.GetPoints().GetData()), axis=0) - \
            np.mean(vtk_to_numpy(mesh.GetPoints().GetData()), axis=0)
    view_horizontal = np.mean(vtk_to_numpy(mesh.GetPoints().GetData()), axis=0) - \
            np.mean(vtk_to_numpy(mesh.GetPoints().GetData()), axis=0)
    #view_up /= np.linalg.norm(view_up)
    nrm = np.cross(view_up, view_horizontal)
    #nrm /= np.linalg.norm(nrm)
    
    # tilt the view a little bit
    view_up = view_up + tilt_factor * nrm
    #view_up /= np.linalg.norm(view_up)
    nrm = np.cross(view_up, view_horizontal)
    #nrm /= np.linalg.norm(nrm)

    # compute extend
    extend = np.matmul(vtk_to_numpy(mesh.GetPoints().GetData()) - np.expand_dims(np.array(center), 0), view_up.reshape(3,1))
    extend_size = np.max(extend) - np.min(extend)
    nrm *= np.linalg.norm(view_up)*extend_size*2.3

    mesh_mapper = vtk.vtkPolyDataMapper()
    if id_to_visualize > -1:
        poly_myo = thresholdPolyData(mesh, threshold_name, (id_to_visualize, id_to_visualize), 'point')
        mesh_mapper.SetInputData(poly_myo)
    else:
        mesh_mapper.SetInputData(mesh)
    mesh_mapper.SelectColorArray(array_name)
    mesh_mapper.SetScalarModeToUsePointFieldData()
    mesh_mapper.RemoveVertexAttributeMapping('Normals_')
    
    # need the following line for categorical color map
    lut = setupLuts('set3')
    rng = mesh.GetPointData().GetArray(array_name).GetRange()
    lut.SetTableRange(rng)
    labels = np.unique(vtk_to_numpy(mesh.GetPointData().GetArray(array_name)))
    lut.SetNumberOfTableValues(len(labels))
    values = vtk.vtkVariantArray()
    for i in range(len(labels)):
        v = vtk.vtkVariant(int(labels[i]))
        values.InsertNextValue(v)
    for i in range(values.GetNumberOfTuples()):
        lut.SetAnnotation(i, values.GetValue(i).ToString())

    mesh_mapper.SetLookupTable(lut)
    mesh_mapper.SetScalarRange(mesh.GetPointData().GetArray(array_name).GetRange())
    actor = vtk.vtkActor()
    actor.SetMapper(mesh_mapper)

    prop = vtk.vtkProperty()
    prop.SetEdgeVisibility(show_edge)
    prop.SetOpacity(1)
    prop.SetAmbient(0.2)
    #prop.SetRenderLinesAsTubes(True)
    actor.SetProperty(prop)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    colors = vtk.vtkNamedColors()
    ren.SetBackground(colors.GetColor3d("White"))
    if camera is None:
        camera = ren.MakeCamera()
        camera.SetClippingRange(0.1, 1000)
        camera.SetFocalPoint(*center)
        camera.SetViewUp(*view_up)
        camera.SetPosition(center[0]+nrm[0], center[1]+nrm[1], center[2]+nrm[2])
    ren.SetActiveCamera(camera)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize((digit_size, digit_size))
    # use this line to generate figure
    ren_win.Render()
    # use the following lines to start an iterative window
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    iren.Start() 
    return ren_win, camera

def visualize_template(mesh_fn, array_name='RegionId', threshold_name='RegionId', show_edge=False, id_to_visualize=-1, tilt_factor=0., digit_size=256, camera=None, gt=False):
    mesh = load_vtk_mesh(mesh_fn)
    mesh.GetPointData().RemoveArray('Normals_')
    com_filter = vtk.vtkCenterOfMass()
    com_filter.SetInputData(mesh)
    com_filter.Update()
    center = com_filter.GetCenter()

    poly_la = thresholdPolyData(mesh, threshold_name, (2, 2),'point')
    poly_lv = thresholdPolyData(mesh, threshold_name, (0, 0),'point')
    poly_rv = thresholdPolyData(mesh, threshold_name, (1, 1),'point')
    pdb.set_trace()
    view_up = np.mean(vtk_to_numpy(poly_la.GetPoints().GetData()), axis=0) - \
            np.mean(vtk_to_numpy(poly_lv.GetPoints().GetData()), axis=0)
    view_horizontal = np.mean(vtk_to_numpy(poly_rv.GetPoints().GetData()), axis=0) - \
            np.mean(vtk_to_numpy(poly_lv.GetPoints().GetData()), axis=0)
    view_up /= np.linalg.norm(view_up)
    nrm = np.cross(view_up, view_horizontal)
    nrm /= np.linalg.norm(nrm)
    
    # tilt the view a little bit
    view_up = view_up + tilt_factor * nrm
    view_up /= np.linalg.norm(view_up)
    nrm = np.cross(view_up, view_horizontal)
    nrm /= np.linalg.norm(nrm)

    # compute extend
    extend = np.matmul(vtk_to_numpy(mesh.GetPoints().GetData()) - np.expand_dims(np.array(center), 0), view_up.reshape(3,1))
    extend_size = np.max(extend) - np.min(extend)
    nrm *= np.linalg.norm(view_up)*extend_size*2.3

    mesh_mapper = vtk.vtkPolyDataMapper()
    if id_to_visualize > -1:
        poly_myo = thresholdPolyData(mesh, threshold_name, (id_to_visualize, id_to_visualize), 'point')
        mesh_mapper.SetInputData(poly_myo)
    else:
        mesh_mapper.SetInputData(mesh)
    mesh_mapper.SelectColorArray(array_name)
    mesh_mapper.SetScalarModeToUsePointFieldData()
    mesh_mapper.RemoveVertexAttributeMapping('Normals_')
    
    # need the following line for categorical color map
    lut = setupLuts('set3')
    rng = mesh.GetPointData().GetArray(array_name).GetRange()
    lut.SetTableRange(rng)
    labels = np.unique(vtk_to_numpy(mesh.GetPointData().GetArray(array_name)))
    lut.SetNumberOfTableValues(len(labels))
    values = vtk.vtkVariantArray()
    for i in range(len(labels)):
        v = vtk.vtkVariant(int(labels[i]))
        values.InsertNextValue(v)
    for i in range(values.GetNumberOfTuples()):
        lut.SetAnnotation(i, values.GetValue(i).ToString())

    mesh_mapper.SetLookupTable(lut)
    mesh_mapper.SetScalarRange(mesh.GetPointData().GetArray(array_name).GetRange())
    actor = vtk.vtkActor()
    actor.SetMapper(mesh_mapper)

    prop = vtk.vtkProperty()
    prop.SetEdgeVisibility(show_edge)
    prop.SetOpacity(1)
    prop.SetAmbient(0.2)
    #prop.SetRenderLinesAsTubes(True)
    actor.SetProperty(prop)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    colors = vtk.vtkNamedColors()
    ren.SetBackground(colors.GetColor3d("White"))
    if camera is None:
        camera = ren.MakeCamera()
        camera.SetClippingRange(0.1, 1000)
        camera.SetFocalPoint(*center)
        camera.SetViewUp(*view_up)
        camera.SetPosition(center[0]+nrm[0], center[1]+nrm[1], center[2]+nrm[2])
    ren.SetActiveCamera(camera)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize((digit_size, digit_size))
    # use this line to generate figure
    ren_win.Render()
    # use the following lines to start an iterative window
    #iren = vtk.vtkRenderWindowInteractor()
    #iren.SetRenderWindow(ren_win)
    #iren.Start() 
    return ren_win, camera

def write_image(ren_win, filename):
    print("Writing PNG to : ", filename)
    w2im = vtk.vtkWindowToImageFilter()
    w2im.SetInput(ren_win)
    w2im.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(w2im.GetOutputPort())
    writer.Write()

def generate_figure(mesh_fn, digit_size, id_to_visualize=-1, tilt_factor=0.5, camera=None):
    ren_win, camera = visualize_template(mesh_fn, array_name='RegionId', threshold_name='RegionId', show_edge=False, id_to_visualize=id_to_visualize, tilt_factor=tilt_factor, digit_size=digit_size, camera=camera)
    # convert to image
    w2im = vtk.vtkWindowToImageFilter()
    w2im.SetInput(ren_win)
    w2im.Update()
    im = w2im.GetOutput()
    img_scalar = im.GetPointData().GetScalars()
    dims = im.GetDimensions()
    n_comp = img_scalar.GetNumberOfComponents()
    temp = vtk_to_numpy(img_scalar)
    numpy_data = temp.reshape(dims[1],dims[0],n_comp)
    numpy_data = numpy_data.transpose(0,1,2)
    numpy_data = np.flipud(numpy_data)
    return numpy_data.astype(int), camera

def plot_results_for_a_sample(dir_n, gt_dir_n, sample_n, camera_t=None, camera_t_myo=None):
    size = 512
    row = 3
    col = 3
    figure = np.ones((size * row, size * col, 3)).astype(int)*255
    
    ground_truth_fn = os.path.join(gt_dir_n, sample_n + '.vtp')
    final_prediction_fn = os.path.join(dir_n, sample_n + '_pred.vtp')
    prediction_noDs_fn_list = sorted(glob.glob(os.path.join(dir_n, sample_n + '_pred_noCorr*.vtp')))
    type_prediction_fn = os.path.join(dir_n, sample_n + '_pred_type.vtp')

    # first row (wh): gt; final pred; type
    pred_data, camera1 = generate_figure(final_prediction_fn, size, tilt_factor=0.5)
    gt_data, _ = generate_figure(ground_truth_fn, size, camera=camera1, tilt_factor=0.5)
    type_data, camera_t = generate_figure(type_prediction_fn, size, camera=camera_t, tilt_factor=0.5)
    figure[:size, :size, :] = gt_data 
    figure[:size, size:2*size, :] = pred_data
    figure[:size, 2*size:3*size, :] = type_data
    plt.text(0+size//3, 0, 'Ground Truth')
    plt.text(size+size//3, 0, 'Prediction')
    plt.text(2*size+size//3, 0, 'Type')
    
    # second row (Myo): gt; final_pred, type
    pred_data, camera2 = generate_figure(final_prediction_fn, size, tilt_factor=-1., id_to_visualize=4)
    gt_data, _ = generate_figure(ground_truth_fn, size, camera=camera2, tilt_factor=-1., id_to_visualize=4)
    type_data, camera_t_myo = generate_figure(type_prediction_fn, size, camera=camera_t_myo, tilt_factor=-1., id_to_visualize=4)
    figure[size:2*size,:size, :] = gt_data
    figure[size:2*size, size:2*size, :] = pred_data
    figure[size:2*size, 2*size:3*size, :] = type_data
    plt.text(0+size//3, size+size//5, 'Ground Truth')
    plt.text(size+size//3, size+size//5, 'Prediction')
    plt.text(2*size+size//3, size+size//5, 'Type')
    
    # thrid row (wh): prednoDs
    for i, fn in enumerate(prediction_noDs_fn_list):
        pred_data, _ = generate_figure(fn, size, tilt_factor=0.5, camera=camera1)
        figure[2*size:3*size, i*size:(i+1)*size, :] = pred_data
        plt.text(i*size+size//3,2*size, 'Pred - dS B{}'.format(i))

    plt.imshow(figure)
    plt.axis('off')
    fig = plt.gcf()
    #fig.set_size_inches(18.5, 18.5)
    fig.savefig(os.path.join(dir_n, sample_n+'.png'), dpi=300)
    return camera_t, camera_t_myo

if __name__ == '__main__':
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_AlterTypeLaterNrmWt0.02_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_AlterTypeLaterNrmWt0.005_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.1_FreezeT20Every5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.1_FreezeT20Every5Decay_smpleFac5'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1block_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1blockTypeInd_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/1blockSDVInt_bgFix_DSB1WT10n5_noContrast_randSwap0.15_FreezeTEvery5Decay_smpleFac20'
    #plot_dice_scores_over_epochs(dir_n)

    #mesh_fn = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/2blocks_smpleFac20/test_10/ct_1145_image_pred.vtp'
    #visualize_template(mesh_fn, array_name='RegionId', show_edge=False, id_to_visualize=4, tilt_factor=-0.5)
    #generate_figure(mesh_fn, digit_size=512)

    #dir_n = '/Users/fanweikong/Documents/Modeling/CHD/output/wh_raw_tests_cleanedall/FlowAllDsAllSeparateCondTypePtS/MultiBlocks/3blocks_bgFix_DSB3B2B1WT2n8n5n5_smpleFac20/small_test_20'
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_n') 
    args = parser.parse_args()
    # gt_dir_n = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all/imageCHDcleaned_all/vtk'
    # gt_dir_n = '/Users/fanweikong/Documents/Modeling/CHD/datasets/imageCHDcleaned/masks_all_aligned/imageCHDcleaned_all/vtk'
    # sample_n_list = ['ct_1102_image', 'ct_1103_image', 'ct_1145_image', 'ct_1146_image']
    # sample_n_list = ['ct_1024_image', 'ct_1052_image', 'ct_1060_image', 'ct_1067_image', \
    #         'ct_1077_image', 'ct_1081_image', 'ct_1101_image', 'ct_1102_image', \
    #         'ct_1103_image', 'ct_1145_image', 'ct_1146_image', 'ct_1147_image']
    # fns = glob.glob(os.path.join(args.dir_n, '*_pred.vtp'))
    # sample_n_list = [os.path.basename(f[:-9]) for f in fns]
    # print(fns)
    # print(sample_n_list)
    # for sample_n in sample_n_list:
    #     try:
    #         plot_results_for_a_sample(args.dir_n, gt_dir_n, sample_n)
    #     except Exception as e:
    #         print(e)

    visualize_template_vasc(args.dir_n, array_name='GlobalBoundaryPoints')