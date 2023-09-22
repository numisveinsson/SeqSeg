def vmr_directories(directory, model, global_scale, dir_seg_exist, cropped, original):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    for a specific model in the
    Vascular Model Repository
    """
    if original:
        dir_image = directory +'images/OSMSC' + model[0:4]+'/OSMSC'+model[0:4]+'-cm.mha'
        dir_seg = directory + 'images/OSMSC'+model[0:4]+'/'+model+'/'+model+'-cm.mha'
    else:
        dir_image = directory +'images/'+model.replace('_aorta','')+'.vtk'
        dir_seg = directory +'truths/'+model.replace('_aorta','')+'.vtk'

    if cropped:
        print('Images not found, check again')
    
    if global_scale:
        dir_image = directory +'scaled_images/'+model.replace('_aorta','')+'.vtk'

    if not dir_seg_exist:
        dir_seg = None

    dir_cent = directory + 'centerlines/'+model+'.vtp'
    dir_surf = directory + 'surfaces/'+model+'.vtp'

    return dir_image, dir_seg, dir_cent, dir_surf
