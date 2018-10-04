import numpy as np
import matplotlib.pyplot as plt


N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.37       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [0.289, 0, 0.059, 0.411, 0.087]
rects1 = ax.bar(ind, yvals, width, color='r')
#rects1 = ax.bar(ind, width, color='r')
zvals = [0.823,0.982,0.785, 0.597, 0.467]
rects2 = ax.bar(ind+width, zvals, width, color='b')
#kvals = [11,12,13]
#rects3 = ax.barind+width*2, kvals, width, color='b')

ax.set_ylabel('F1 Score')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('test-set1', 'test-set2', 'test-set3', 'test-set4', 'test-set5') )
#ax.legend( (rects1[0], rects2[0], rects3[0]), ('y', 'z', 'k') )
ax.legend( (rects1[0], rects2[0]), ('Stanford-NER', 'New Method') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)

plt.show()
