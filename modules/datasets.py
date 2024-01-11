def vmr_directories(directory, model, dir_seg_exist, cropped, original, global_scale=None):
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
        dir_image = directory +'images/'+model+'.mha'
        dir_seg = directory +'truths/'+model+'.mha'

    if cropped:
        print('Images not found, check again')
    
    # if global_scale:
    #     dir_image = directory +'scaled_images/'+model.replace('_aorta','')+'.vtk'

    if not dir_seg_exist:
        dir_seg = None

    dir_cent = directory + 'centerlines/'+model+'.vtp'
    dir_surf = directory + 'surfaces/'+model+'.vtp'

    return dir_image, dir_seg, dir_cent, dir_surf

def get_directories(directory_data, case, img_ext, dir_seg =True):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    for a specific model in the
    Vascular Model Repository
    """
    dir_image = directory_data + 'images/'+case+img_ext
    dir_seg = directory_data + 'truths/'+case+img_ext if dir_seg else None
    dir_cent = directory_data + 'centerlines/'+case+'.vtp'
    dir_surf = directory_data + 'surfaces/'+case+'.vtp'

    return dir_image, dir_seg, dir_cent, dir_surf

def get_testing_samples_json(dir_json):
    """
    Get testing samples from json file
    The json file must be organized as follow:
    [
        {
            "name": "case1",
            "seeds": [
                [[x,y,z], [x,y,z]], [[x,y,z], [x,y,z]]
            ],
        },
        {
            "name": "case2",
            "seeds": [
                [[x,y,z], [x,y,z]]
            ],
        }
    ]
    where case1 has two seeds and case2 has one seed
    Returns:
        testing_samples: list of testing sample dictionaries
    """
    import json
    with open(dir_json) as f:
        data = json.load(f)

    testing_samples = []
    for case in data:
        name = case['name']
        seeds = case['seeds']
        for i, seed in enumerate(seeds):
            testing_samples.append([name, i, seed[0], seed[1], 'ct'])

    return testing_samples