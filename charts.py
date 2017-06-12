#import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt



def bar_plot(path, x, y, str_x="", str_y="", title="", percent="false"):
	
	#constants
	alpha = 0.8
	precision = 3
	
	y_pos = np.arange(len(x))
	bar = plt.bar(y_pos, y, alpha=alpha)
	plt.xticks(y_pos, x)
	plt.ylabel(str_y)
	plt.xlabel(str_x)
	plt.title(title)
	
	#print value in every rectangle of bar
	(y_bottom, y_top) = plt.ylim()
	for rect in bar:
		height = rect.get_height()
		if(percent == "true"):
			plt.text(rect.get_x() + rect.get_width()/2., y_bottom, '{:.{prec}f}%'.format(height*100, prec=precision), ha='center', va='bottom')
		else:
			plt.text(rect.get_x() + rect.get_width()/2., y_bottom, '{:.{prec}f}'.format(height, prec=precision), ha='center', va='bottom')
	
	#plt.show()
	plt.savefig(path)
	plt.clf()

def bar_plot2(x, y, xlabels, str_x="", str_y="", title="", group1="", group2=""):

	n_groups = len(x)
	fig, ax = plt.subplots()
	index = np.arange(n_groups)

	bar_width = 0.35
	opacity = 0.8
	 
	rects1 = plt.bar(index, x, bar_width,
					 alpha=opacity,
					 color='b',
					 label=group1)
	 
	rects2 = plt.bar(index + bar_width, y, bar_width,
					 alpha=opacity,
					 color='g',
					 label=group2)
	 
	plt.xlabel(str_x)
	plt.ylabel(str_y)
	plt.title(title)
	plt.xticks(index + bar_width, xlabels)
	plt.legend()
	 
	#plt.tight_layout()
	plt.show()

#example: 
#x1 = [1,2,3] 
#x2 = [4,5,6]
#bar_plot2(x1, x2, ['A', 'B', 'C', 'D'], "xlabel", "ylabel", "tytul", "grupa1", "grupa2")