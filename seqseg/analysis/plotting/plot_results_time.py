import matplotlib.pyplot as plt
import numpy as np

# Pie charts for story

names = ['image data extraction ',
         'model inference       ',
         '',#'surface calculation   ',
         'centerline calculation',
         ' ',#'global assembly write ',
         'other calculation']

# 108 steps
serial  = [1.13180929, 0.70873534, 0.01873574, 0.577397, 0.62047277, 0.00001]
p1 = [2.11024304440147, 0.9609730871100175, 0.03416417774401213, 0.13363992540459885, 1.035707900398656, 0.0005200411144055819]
p2 = [2.4639598444888464, 0.9510593037856253, 0.03722771845365826, 0.1371730126832661, 0.10924773467214484, 0.00047482942280016447]
p3 = [2.8426294703232613, 1.0319310363970304, 0.04939505928441098, 0.18085532439382454, 0.05724083749871505, 0.0006236026161595395]
p4 = [0.18235421180725098, 2.2392032146453857, 0.05390942096710205, 0.17729604244232178, 0.11358654499053955, 0.00034987926483154297]

total_time = [6.009738099575043, 5.068479335308075, 3.062176299095154, 3.0165377855300903, 2.4963444034258524]
import pdb; pdb.set_trace()
plt.close()
x = ['Serial', 'Parallel V1', 'Parallel V2', 'Parallel V3', 'Parallel V4']
plt.bar(x, total_time, color='teal')
plt.xlabel("Version")
plt.ylabel("Time [min]")
#plt.legend(labels)
plt.title("Average Time To Do 108 Steps")
plt.show()

import pdb; pdb.set_trace()
# 312 steps
seriall  = [0.9862848722134916, 0.5214065378085493*4, 0.012854493845004243, 0.36319978397113445, 0.5147083117939032, 0.0]
p44 = [0.2692110714239952, 1.3652023191635425, 0.05413913650390429, 0.9402910241713891, 0.3074118449137761, 0.0006091762811709673]
corii = [0.29173608311152055, 0.2005511510170112*4, 0.018638058032019664, 0.3677826501555362, 0.16346378245596158]
total_timee = [13.547532749176025,6.092012015978495,2.198979167143504]

# serial_0002 = [0.80115550242293, 0.5453196074635076, 0.008681030700001093, 0.17216027125763816, 1.554440588996814, 0.0]
# total_time_0002 = [21.222498031457267, 9.758848853905995, , , ]
#

test = p2

title  = 'p2_ave_step'
Title_graph = 'Parallel V2 - Average Time Step'
sizes = test[:-1]

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = names[:-1]#'Adding to Global Assembly', 'Extracting Subvolume', 'ML Inference', 'Other Calculation'
#[38.2, 43.8, 17.2, 0.8]
scale = 0.0
explode = scale*np.array([1,1,1,1,1])  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%2.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(Title_graph, y =1.05)

plt.savefig('./'+title +'.png')
plt.show()
plt.close()

import pdb; pdb.set_trace()
# Stacked bar chart
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

import pdb; pdb.set_trace()
plt.close()

plt.bar(x, total_timee, color='teal')
plt.xlabel("Test")
plt.ylabel("Time [min]")
#plt.legend(labels)
plt.title("Total Time for 312 steps")
plt.show()


import pdb; pdb.set_trace()
# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.subplots_adjust(wspace=0)

# pie chart parameters
overall_ratios = [.82, .18 ]
labels = ['Data Movement', 'Calculation']
explode = [0.1, 0]
# rotate so that first wedge is split by the x-axis
angle = -180 * overall_ratios[0]
wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                     labels=labels, explode=explode)

# bar chart parameters
age_ratios = [.466, 0.534]
age_labels = ['Extracting Subvolume', 'Writing to Global Assembly']
bottom = 1
width = .2

# Adding from the top matches the legend.
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                 alpha=0.1 + 0.25 * j)
    #ax2.bar_label(bc, labels=[f'{height:.0%}'], label_type='center')

ax2.set_title('Age of approvers')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2
center, r = wedges[0].center, wedges[0].r
bar_height = sum(age_ratios)

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

plt.show()

import pdb; pdb.set_trace()



import pdb; pdb.set_trace()
#############
mean_mse = []
mean_mae = []
N = 12
mean_mae_scaled = []
mean_mse_scaled = []

names = []
for i in range(1,N):
    names.append(str(i))

print('length of names: ', len(names))
print('length of data: ', len(mean_mae))

#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
plt.bar(names,mean_mse_scaled)
plt.title('Average Squared Error - Test Set')
plt.ylabel('Error')
plt.xlabel('Training Nr')
plt.show()

#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
plt.bar(names,mean_mae_scaled)
plt.title('Average Absolute Error - Test Set')
plt.ylabel('Error')
plt.xlabel('Training Nr')
plt.show()
