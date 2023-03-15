
def plot_pie(data, labels, x_label, y_label, title, file_save, scale = None, emphasis = None):

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
