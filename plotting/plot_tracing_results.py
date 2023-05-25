from plot_functions import plot_histograms, plot_bar_grouped
import numpy as np
import pickle

def organize(data, tests, column_names):
    """
    Function to organize tests as:
    {...
    testx: (ct_results, mr_result)
    ...
    }
    """
    data_p = {}
    for i,test in enumerate(data['test']):
        if test in tests:
            data_p[test] = (data[column_names[0]][i], data[column_names[1]][i])

    return data_p

def combine_tests(data,tests,new_name):

    data_p = {}
    for key in data.keys():
        data_p[key] = []

    for test in tests:
        for i in range(len(data['test'])):
            if data['test'][i] == test:
                for key in data.keys():
                    data_p[key].append(data[key][i])

    for key in data_p:
        if key == 'test': data_p[key] = [new_name]
        else:
            data_p[key] =  [x for x in data_p[key] if x == x]

    if len(data_p['ct dice']) > 1:
        data_p['ct dice'] = [data_p['ct dice'][0]]
        data_p['mr dice'] = [data_p['mr dice'][1]]
        data_p['ct cent'] = [data_p['ct cent'][0]]
        data_p['mr cent'] = [data_p['mr cent'][1]]

    return data_p

if __name__=='__main__':

    set_values  = True

    results_dir = '/Users/numisveinsson/Documents_numi/Automatic_Centerline_Data/outputs/'
    results_dir2 = '/Users/numisveinsson/Documents_numi/Automatic_Centerline_Data/outputs_global/'

    fn = '_global_our_vs_unet'

    tests = ['Our', '3D U-Net']
    tests_our = ['test55','test53']
    tests_unet = ['test114','test115']

    if set_values:
        data = {'test': ['Our'],
                'ct dice': [0.90], 'mr dice': [0.867],
                'ct cent': [0.886], 'mr cent': [0.943]}
        data1 = {'test': ['3D U-Net'],
                'ct dice': [0.642], 'mr dice': [0.484],
                'ct cent': [0.232], 'mr cent': [0.267]}
    else:
        with open(results_dir+'results.pickle', 'rb') as handle:
            data = pickle.load(handle)
        with open(results_dir2+'results_2.pickle', 'rb') as handle:
            data1 = pickle.load(handle)

        # Function to combine two tests(one for ct, other for mr)
        data = combine_tests(data, tests_our,tests[0])
        data1 = combine_tests(data1, tests_unet,tests[1])

        #data1, tests = combine_tests(data, tests_our,'3D U-Net')

    for key in data.keys():
        for i in range(len(data1['test'])):
            data[key].append(data1[key][i])


    data_dice  = organize(data, tests ,['ct dice','mr dice'])
    data_cent  = organize(data, tests ,['ct cent','mr cent'])

    names_old = ['test54','test55','test53',
                 'test56','test57','test58',
                 'test100']
    names_new = [ 'Our: Both - Global Scaling', 'Our: CT - Global Scaling',
                  'Our: MR - Global Scaling',
                  'Our: Both - Local Scaling', 'Our: CT - Local Scaling',
                  'Our: MR - Local Scaling',
                  '3D U-Net']

    for data in [data_dice, data_cent]:
        for i,key in enumerate(names_old):
            if key in data.keys():
                data[names_new[i]] = data.pop(key)

    # Packages
    dice_pkg = [data_dice, 'Dice', 'Global Dice For Different Methods', 'dice'+fn]
    cent_pkg = [data_cent, 'Centerline Overlap', 'Tracing Score For Different Methods', 'cent'+fn]

    for pkg in [dice_pkg, cent_pkg]:

        title = pkg[2]
        labels = ("CT", "MR")
        x_label = 'Modality'
        y_label = pkg[1]
        file_save = './'+pkg[3]+'.png'
        colormap = None #'viridis'
        colors = ['#F8CECC','#DAE8FC']#None
        edge_colors = ['#B85450','#6C8EBF']#None
        plot_bar_grouped(pkg[0], labels, x_label, y_label, title, file_save, y_lim = (0,1), colormap = colormap, colors = colors, edge_colors = edge_colors)



    # import pdb; pdb.set_trace()
    #
    # # Obtain dice scores from test
    # dict_from_csv, np_from_csv = read_csv_file(file)
    #
    # sizes = dict_from_csv['SIZE']
    # tangentx, tangenty, tangentz = dict_from_csv['TANGENTX'].values, dict_from_csv['TANGENTY'].values, dict_from_csv['TANGENTZ'].values
    # radii = dict_from_csv['RADIUS']
    # bifurc = dict_from_csv['BIFURCATION']
    # num_bif = bifurc.values.sum()
    #
    # tangents = np.array([tangentx, tangenty, tangentz]).T
    #
    # tang_org = organize_vectors(tangents)
    #
    # #plot_scatter([sizes.values, tangenty.values, radii.values],'sizes', 'tangent_x', 'radii', 'Comparison', 'diff')
    #
    # import pdb; pdb.set_trace()
    #
    # plot_3d_vector([tangentx, tangenty, tangentz], x_name, y_name, title)
    #
    # plot_histograms(sizes.values, 'Subvolume Sidelength [cm]', 'Frequency', 'Histogram of Subvolume Sizes')
    #
    # plot_histograms(tangentx.values, 'Tangent - X', 'Frequency', 'Histogram of X component of Vessel Tangent')
    #
    # plot_histograms(tangenty.values, 'Tangent - Y', 'Frequency', 'Histogram of Y component of Vessel Tangent')
    #
    # plot_histograms(tangentz.values, 'Tangent - Z', 'Frequency', 'Histogram of Z component of Vessel Tangent')
    #
    # plot_histograms(radii.values, 'Vessel Radius [cm]', 'Frequency', 'Histogram of Vessel Radii in Samples')
    #
    # import pdb; pdb.set_trace()
    # # Open the tested images and save size data
    # dir = out_dir+modality+'_train/'
    # size, min_res = get_img_sizes(dir)
    #
    # fig, axis = plt.subplots(2,1)
    # scatter = axis[0].scatter(size, dice_scores, linewidth=0.1)
    # axis[0].grid()
    # axis[0].set_ylabel('Dice score')
    # axis[0].set_xlabel('Volume size [cm]')
    # axis[0].set_title('Test 1 - Dice against Volume Size')
    #
    # scatter = axis[1].scatter(min_res, dice_scores, linewidth=0.1)
    # axis[1].grid()
    # axis[1].set_ylabel('Dice score')
    # axis[1].set_xlabel('Min Dimension Resolution')
    # axis[1].set_title('Test 1 - Dice against Min Resolution')
    #
    # fig, axis = plt.subplots(2,1)
    # scatter = axis[0].hist(size, 20)
    # axis[0].grid()
    # axis[0].set_ylabel('Count')
    # axis[0].set_xlabel('Volume size [cm]')
    # axis[0].set_title('Test 1 - Volume Size')
    #
    # scatter = axis[1].hist(min_res, 20)
    # axis[1].grid()
    # axis[1].set_ylabel('Count')
    # axis[1].set_xlabel('Min Dimension Resolution')
    # axis[1].set_title('Test 1 - Min Resolution')
    # plt.show()
    #
    # fig, axis = plt.subplots(2,1)
    # import pdb; pdb.set_trace()
