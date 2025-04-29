import os


# def vmr_directories(directory, model, dir_seg_exist=True, global_scale=None):
#     """
#     Function to return the directories of
#         Image Volume
#         Segmentation Volume
#         Centerline VTP
#         Surface Mesh VTP
#     for a specific model in the
#     Vascular Model Repository
#     """

#     dir_image = directory + 'images/'+model+'.mha'
#     dir_seg = directory + 'truths/'+model+'.mha'

#     # if global_scale:
#     #     dir_image = directory + \
#     #         'scaled_images/'+model.replace('_aorta', '')+'.vtk'

#     if not dir_seg_exist:
#         dir_seg = None

#     dir_cent = directory + 'centerlines/'+model+'.vtp'
#     dir_surf = directory + 'surfaces/'+model+'.vtp'

#     return dir_image, dir_seg, dir_cent, dir_surf


def get_directories(directory_data, case, img_ext, dir_seg=True):
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

    return data


def get_testing_samples(dataset, data_dir=None):
    """
    Get testing samples for a specific dataset
    Returns:
        testing_samples: list of testing sample dictionaries
    """
    if data_dir:
        print('Data directory provided')
        directory = data_dir
        print('Checking for json file')
        dir_json = directory + 'seeds.json'
        if os.path.isfile(dir_json):
            print('Json file found')
            testing_samples = get_testing_samples_json(dir_json)
        else:
            print('Json file not found, trying image and centerline folders')
            testing_samples = os.listdir(directory + 'images/')
            # no hidden files
            testing_samples = [s for s in testing_samples if s[0] != '.']
            print('Image extensions:', testing_samples[0].split('.')[-1])
            # remove extension
            testing_samples = [s.split('.')[0] for s in testing_samples]
            print('Number of testing samples:', len(testing_samples))

            # testing samples centerlines
            testing_samples_cent = os.listdir(directory + 'centerlines/')
            # no hidden files
            testing_samples_cent = [s for s in testing_samples_cent
                                    if s[0] != '.']
            # remove extension
            testing_samples_cent = [s.split('.')[0]
                                    for s in testing_samples_cent]
            print('Number of testing samples centerlines:',
                  len(testing_samples_cent))

            # only keep images that have centerlines
            testing_samples = [s for s in testing_samples
                               if s in testing_samples_cent]
            print('Number of testing samples with centerlines:',
                  len(testing_samples))
            testing_samples.sort()
            testing_samples = [[s, 0, -50, -60] for s in testing_samples]
    else:
        print('No data directory provided.')
        print('Using default data directory based on training dataset.')

        directory = '/global/scratch/users/numi/vascular_data_3d/'
        if dataset in ['Dataset005_SEQAORTANDFEMOMR',
                       'Dataset018_SEQAORTASONEMR',
                       'Dataset020_SEQAORTAS015MR',
                       'Dataset022_SEQAORTAS025MR',
                       'Dataset024_SEQAORTAS050MR',
                       'Dataset026_SEQAORTAS075MR'
                       ]:
            
            testing_samples = [

                ['0006_0001', 0, 3, 5, 'mr'],  # Aortofemoral MR
                ['0063_1001', 0, 10, 20, 'mr'],  # Aortic MR
                ['0070_0001', 0, 10, 20, 'mr'],  # Aortic MR
                ['0090_0001', 0, 0, 5, 'mr'],  # Aortic MR
                ['0131_0000', 0, 10, 20, 'mr'],  # Aortic MR
                ['KDR12_aorta', 0, 20, 30, 'mr'],  # Aortic MR
                ['KDR33_aorta', 3, -10, -20, 'mr'],  # Aortic MR

            ]

        elif dataset in ['Dataset006_SEQAORTANDFEMOCT',
                         'Dataset017_SEQAORTASONECT',
                         'Dataset021_SEQAORTAS015CT',
                         'Dataset023_SEQAORTAS025CT',
                         'Dataset025_SEQAORTAS050CT',
                         'Dataset027_SEQAORTAS075CT'
                         ]:
        
            directory = '/global/scratch/users/numi/vascular_data_3d/'
            # dir_json = directory + 'seeds.json'
            testing_samples = [  # get_testing_samples_json(dir_json)

                ['0139_1001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0141_1001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0146_1001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0174_0000', 0, 5, 10, 'ct'],  # Aorta CT
                ['0176_0000', 0, 50, 60, 'ct'],  # Aorta CT
                ['0188_0001_aorta', 5, -10, -20, 'ct'],  # Aorta CT
                ['O150323_2009_aorta', 0, 10, 20, 'ct'],  # Aorta CT
                ['O344211000_2006_aorta', 0, 0, 1, 'ct'],  # Aorta CT
            ]

        elif dataset == 'Dataset007_SEQPULMONARYMR':
            testing_samples = [

                ['0085_1001', 0, 0, 10, 'mr'],  # Pulmonary MR
                # ['0085_1001',0,200,220,'mr'], # Pulmonary MR
                # ['0085_1001',1,200,220,'mr'], # Pulmonary MR
                ['0081_0001', 0, 20, 30, 'mr'],  # Pulmonary MR
                # ['0081_0001',1,200,220,'mr'], # Pulmonary MR
                # ['0081_0001',1,2000,220,'mr'], # Pulmonary MR
            ]
        elif dataset == 'Dataset009_SEQAORTASMICCT':

            directory = '/global/scratch/users/numi/test_data/miccai_aortas/'
            dir_json = directory + 'seeds.json'
            testing_samples = get_testing_samples_json(dir_json)

        elif dataset == 'Dataset010_SEQCOROASOCACT':

            directory = '/global/scratch/users/numi/ASOCA_test/'
            # directory = '/global/scratch/users/numi/Karthik_test/'
            dir_json = directory + 'seeds.json'
            testing_samples = get_testing_samples_json(dir_json)

        elif dataset == 'Dataset016_SEQPULMPARSECT':
            directory = '/global/scratch/users/numi/PARSE_dataset/'
            dir_asoca_json = directory + 'seeds.json'
            testing_samples = get_testing_samples_json(dir_asoca_json)

        elif dataset == 'Dataset048_SEQAORTAVMRGALACT':

            directory = '/global/scratch/users/numi/vascular_data_3d/'
            # dir_json = directory + 'seeds.json'
            testing_samples = [  # get_testing_samples_json(dir_json)
                ['0139_1001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0141_1001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0144_1001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0146_1001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0150_0001', 0, 0, 1, 'ct'],  # Aortofemoral CT
                ['0151_0001', 0, 0, 1, 'ct'],  # Aortofemoral CT
            ]

        else:
            print('Dataset not found')
            testing_samples = None

    #    ['0108_0001_aorta',4,-10,-20,'ct'],
    #    ['0183_1002_aorta',3,-10,-20,'ct'],
    #    ['0184_0001_aorta',3,-10,-20,'ct'],
    #    ['0188_0001_aorta',5,-10,-20,'ct'],
    #    ['0189_0001_aorta',4,-50,-60,'ct'],

    #    ['O0171SC_aorta',0,10,20,'ct'],
    #    ['O6397SC_aorta',0,10,20,'ct'],
    #    ['O8693SC_aorta',0,10,20,'ct'],
    #    ['O344211000_2006_aorta',0,10,20,'ct'],
    #    ['O11908_aorta',0,10,20,'ct'],
    #    ['O20719_2006_aorta',0,10,20,'ct'],
    #   these are left:
    #    ['O51001_2009_aorta',0,10,20,'ct'],
    #    ['O128301_2008_aorta',0,10,20,'ct'],
    #    ['O145207_aorta',0,10,20,'ct'],
    #    ['O150323_2009_aorta',0,10,20,'ct'],
    #    ['O227241_2006_aorta',0,10,20,'ct'],
    #    ['O351095_aorta',0,10,20,'ct'],
    #    ['O690801_2007_aorta',0,10,20,'ct'],

    #    ['KDR08_aorta',0,10,20,'mr'],
    #    ['KDR10_aorta',0,10,20,'mr'],
    #    ['KDR12_aorta',0,10,20,'mr'],
    #    ['KDR13_aorta',0,10,20,'mr'],
    #    ['KDR32_aorta',0,10,20,'mr'],
    #    ['KDR33_aorta',3,-10,-20,'mr'],
    #    ['KDR34_aorta',0,10,20,'mr'],
    #    ['KDR48_aorta',0,10,20,'mr'],
    #    ['KDR57_aorta',4,-10,-20,'mr'],

    #    ['0002_0001',0,150,170,'ct']  ,
    #    ['0002_0001',1,150, 170,'ct'] ,
    #    ['0001_0001',0,30,50,'ct'],
    #    ['0001_0001',7,30,50,'ct'],
    #    ['0001_0001',8,110,130,'ct'],
    #    ['0005_1001',0,300,320,'ct']  ,
    #    ['0005_1001',1,200,220,'ct']  ,

    return testing_samples, directory
