import pickle
import matplotlib.pyplot as plt
import numpy

# make figure and assign axis objects
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
# fig.subplots_adjust(wspace=0)
#
# # pie chart parameters
# overall_ratios = [.82, .18 ]
# labels = ['Data Movement', 'Calculation']
# explode = [0.1, 0]
# # rotate so that first wedge is split by the x-axis
# angle = -180 * overall_ratios[0]
# wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
#                      labels=labels, explode=explode)
#
# # bar chart parameters
# age_ratios = [.466, 0.534]
# age_labels = ['Extracting Subvolume', 'Writing to Global Assembly']
# bottom = 1
# width = .2
#
# # Adding from the top matches the legend.
# for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
#     bottom -= height
#     bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
#                  alpha=0.1 + 0.25 * j)
#     ax2.bar_label(bc, labels=[f'{height:.0%}'], label_type='center')
#
# ax2.set_title('Age of approvers')
# ax2.legend()
# ax2.axis('off')
# ax2.set_xlim(- 2.5 * width, 2.5 * width)
#
# # use ConnectionPatch to draw lines between the two plots
# theta1, theta2 = wedges[0].theta1, wedges[0].theta2
# center, r = wedges[0].center, wedges[0].r
# bar_height = sum(age_ratios)
#
# # draw top connecting line
# x = r * np.cos(np.pi / 180 * theta2) + center[0]
# y = r * np.sin(np.pi / 180 * theta2) + center[1]
# con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
#                       xyB=(x, y), coordsB=ax1.transData)
# con.set_color([0, 0, 0])
# con.set_linewidth(4)
# ax2.add_artist(con)
#
# # draw bottom connecting line
# x = r * np.cos(np.pi / 180 * theta1) + center[0]
# y = r * np.sin(np.pi / 180 * theta1) + center[1]
# con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
#                       xyB=(x, y), coordsB=ax1.transData)
# con.set_color([0, 0, 0])
# ax2.add_artist(con)
# con.set_linewidth(4)
#
# plt.show()

#import pdb; pdb.set_trace()
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'Adding to Global Assembly', 'Extracting Subvolume', 'ML Inference', 'Other Calculation'
# sizes = [38.2, 43.8, 17.2, 0.8]
# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.5f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()

for i in [48,49]:#range(1,9):

    test = 'test'+str(i) #'cent_test_' + str(i) + '_scaled'
    history_dir = './weights/'+test+'_history'
    output_dir = './weights/'
    #history_dir = '/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D/output_training/'+test+'_history'
    file = open(history_dir, 'rb')
    history = pickle.load(file)
    file.close()

    names = [*history.keys()]
    names.remove('lr')
    print('Metrics are: ', names)
    n = int(len(names)/2)
    names_done = []
    try:
        for i in range(len(names)):
            if names[i] not in names_done:
                print(i)
                name = names[i]
                if 'val' in name:
                    name2  = name
                    name = name.replace('val_','')
                else:
                    name2 = 'val_'+name
                plt.plot(history[name])
                plt.plot(history[name2])
                plt.title(test + ': '+name)
                plt.ylabel(name)
                plt.xlabel('epoch')
                plt.legend([name, name2], loc='upper right')
                plt.savefig(output_dir + test + '_'+ name +'.png')
                plt.show()
                #plt.wait(2)
                plt.close()
                names_done.append(name)
                names_done.append(name2)
    except Exception as e:
        print(e)

import pdb; pdb.set_trace()
# summarize history for accuracy
plt.plot(history['dice_loss'])
plt.plot(history['val_dice_loss'])
plt.title(test + ': model dice loss')
plt.ylabel('dice loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title(test + ': model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
