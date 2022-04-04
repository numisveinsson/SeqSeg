def vmr_directories(directory, model):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    for a specific model in the
    Vascular Model Repository
    """
    dir_image = directory +'images/OSMSC' + model[0:4]+'/OSMSC'+model[0:4]+'-cm.mha'
    dir_seg = directory + 'images/OSMSC'+model[0:4]+'/'+model+'/'+model+'-cm.mha'

    dir_cent = directory + 'centerlines/'+model+'.vtp'
    dir_surf = directory + 'surfaces/'+model+'.vtp'

    return dir_image, dir_seg, dir_cent, dir_surf
