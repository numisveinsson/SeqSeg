import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

def plot_bar_grouped(data, labels, x_label, y_label, title, file_save,
                        x_lim=None, y_lim=None,
                        colormap=None, colors=None, edge_colors=None):
    num_sub = len(data.keys())

    x = np.arange(len(labels))  # the label locations
    width = 1/(len(data.keys())+1)  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(constrained_layout=True)

    if colormap:
        category_colors = plt.colormaps[colormap](
            np.linspace(0.15, 0.85, num_sub))
    elif colors:
        category_colors = colors

    for attribute, measurement in data.items():

        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute,
                       color = category_colors[multiplier], edgecolor = edge_colors[multiplier],
                       linewidth = 2)
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.set_xticks(x + width*(num_sub-1)/2, labels)
    ax.legend(loc='lower left')

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    plt.savefig(file_save)
    plt.show()
    plt.close()

def plot_pie(data, labels, x_label, y_label, title, file_save, scale = None,
            emphasis = None):

    if not scale:
        scale = 0.0
    if emphasis:
        explode = np.zeros(len(data))
        explode[emphasis] = 0.1
    else:
        explode = np.ones(len(data))

    explode = scale*explode  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(data, explode=explode, labels=labels, autopct='%2.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(file_save, y =1.05)

    plt.savefig('./'+title +'.png')
    plt.show()
    plt.close()

def plot_bar_stacked():

    data1, data2, data3, data4, data5 = [],[],[],[],[]
    for test in [seriall, p44, corii]:
        i=0
        for data in [data1, data2, data3, data4, data5]:
            data.append(test[i])
            i += 1
    tests = ['Serial', 'Parallel V4', 'Cori']
    men_means = [20, 35, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    men_std = [2, 3, 4, 1, 2]
    women_std = [3, 5, 2, 3, 3]
    width = 0.35       # the width of the bars: can also be len(x) sequence

    # create data
    x = tests
    y1 = np.array(data1)
    y2 = np.array(data2)
    y3 = np.array(data3)
    y4 = np.array(data4)
    y5 = np.array(data5)

    # plot bars in stack manner
    plt.bar(x, y1, color='khaki')
    plt.bar(x, y2, bottom=y1, color='mediumslateblue')
    plt.bar(x, y3, bottom=y1+y2, color='darkorange')
    plt.bar(x, y4, bottom=y1+y2+y3, color='teal')
    plt.bar(x, y5, bottom=y1+y2+y3+y4, color='lightcoral')
    plt.xlabel("Test")
    plt.ylabel("Time [s]")
    plt.legend(labels)
    plt.title("Comparing Time Per Step")
    plt.show()
    plt.close()

def get_img_sizes(dir):
    import SimpleITK as sitk
    size = []
    min_res = []
    for file in [f for f in os.listdir(dir) if f.endswith('.nii.gz')]:
        ref_im =  sitk.ReadImage(dir+file)
        size.append( ref_im.GetSize()[0] * ref_im.GetSpacing()[0] )
        min_res.append(np.min(ref_im.GetSize()))

    return size, min_res

def read_csv_file(file):
    import pandas as pd
    dict = pd.read_csv(file)
    pd_array = dict.values

    return dict, pd_array

#def plot_histograms():


def plot_scatter(data, x_name, y_name, sizes, title, legend):

    fig, axis = plt.subplots(1,1)
    scatter = axis.scatter(data[0], data[1], linewidth=0.1, s = data[2])
    axis.grid()
    axis.set_ylabel(y_name)
    legend1 = axis.legend(legend ,loc = "upper right")
    axis.add_artist(legend1)
    kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="{x:.2f} cm",
          func=lambda s: s/100)
    legend = axis.legend(*scatter.legend_elements(**kw),
                    loc="upper left", title=x_name)
    plt.show()

def plot_histograms(data, x_name, y_name, title):

    # the histogram of the data
    n, bins, patches = plt.hist(data, 50, density=False, facecolor='g',
                                alpha=0.75)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    plt.grid(True)
    plt.show()

def plot_3d_vector(data, x_name, y_name, title):

    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    shape_ = data[0].shape
    ax.quiver(np.zeros(shape_), np.zeros(shape_),
            np.zeros(shape_),data[0], data[1], data[2],
            length=0.1)
    plt.show()

def organize_vectors(vectors):
    "Function to combine similar vectors, all of length 1 to begin with"

    keep_running = True
    while keep_running:
        ind_to_delete, mult = [], 1
        ind = arr=np.random.random(1)*(vectors.shape[0]-1)
        ind = ind[0].astype('int')
        vector = vectors[ind]
        if np.linalg.norm(vector)< 2:
            print("ind is: ", ind)
            for j in range(vectors.shape[0]):
                if j == ind:
                    continue
                if np.dot(vector, vectors[j]) > 0.99 and np.linalg.norm(vectors[j]<2):
                    #print('dot is ', np.dot(vector, vectors[j]))
                    mult += 1
                    ind_to_delete.append(j)
            vectors[ind] *= mult
            vectors = np.delete(vectors, ind_to_delete, axis = 0)
            print("now: ", vectors.shape[0])
        if vectors.shape[0] < 1000:
            break
    norms = 0
    for p in vectors:
        norm = np.linalg.norm(p)
        print()
    import pdb; pdb.set_trace()

    return vectors
