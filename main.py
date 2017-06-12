from mlp import MLP
import trans
import csv
import numpy as np

def readDataset(filename):
	file = open(filename)
	reader = csv.reader(file)
	header = next(reader) # the first line is the header

	# row[0] is ID - not needed
	# row[1] is output
	# row[2:] are inputs

	def readInputs(row):
		return [float(param) for param in row[2:]]

	def readOutputs(row):
	    if row[1] == 'M':
	        targets = [0.0]
	    else:
	        targets = [1.0]
	    return targets

	# read first row and initialize lists and minmax
	firstRow = next(reader)
	targets = [readOutputs(firstRow)]
	inputs = [readInputs(firstRow)]

	# output is binary, arleady known
	targetssize = len(targets[0])
	inputssize = len(inputs[0])

	for row in reader:
		inp = readInputs(row)
		tar = readOutputs(row)
		assert len(inp) == inputssize
		assert len(tar) == targetssize

		inputs.append(inp)
		targets.append(tar)

	return [inputs, targets]



class NetTester(object):
	def __init__(self, net, goal, epochs, learningRate):
		self.net = net
		self.learningRate = learningRate
		self.goal = goal
		self.epochs = epochs

	def __call__(self, train_inputs, train_targets, ver_inputs, ver_targets):
		assert len(ver_inputs) == len(ver_targets)
		assert len(train_inputs) == len(train_targets)

		self.net.init()
		
		#monitf param to see epochs
		def monitf(epoch, error):
			if (epoch % 100) == 0:
				print("Epoch no#{0}, error = {1}".format(epoch, error))
				
		# start train
		train_errorMSE = self.net.train(train_inputs, train_targets, self.goal, self.epochs, self.learningRate) 
		
		# verify trained net
		ver_outputs = [self.net.simulate(inp) for inp in ver_inputs]
		accuracies = [self.calcAccuracy(tar, out) 
			for [tar,out] in zip(ver_targets, ver_outputs)]

		# calculate mean
		ver_mean = sum(accuracies) / len(accuracies)
		return ver_mean, train_errorMSE

	@staticmethod
	def calcAccuracy(target, output):
		if target >= [0.5] and output >= [0.5]:
			return 1.0
		elif target < [0.5] and output < [0.5]:
			return 1.0
		else:
			return 0.0

def kFoldValidation(inputs, targets, k, testf):
	inputs_clusters = np.array_split(inputs, k)
	targets_clusters = np.array_split(targets, k)

	ver_errors = []
	train_errors = []
	for i in range(k):
		train_inputs = inputs_clusters[i]
		train_targets = targets_clusters[i]
		ver_inputs = np.vstack(inputs_clusters[:i] + inputs_clusters[(i+1):])
		ver_targets = np.vstack(targets_clusters[:i] + targets_clusters[(i+1):])

		ver_error, train_error = testf(train_inputs, train_targets, ver_inputs, ver_targets)
		
		ver_errors.append(ver_error)
		train_errors.append(train_error)
		
		print("K-Fold step#", i, 'net accuracy = {:.3f}%'.format(ver_error*100), "train error: {:.2f}".format(train_error))
		#print("K-Fold step#", i, "train error: {:.3f}".format(train_error))

	return ver_errors, train_errors
	
def multipleKFoldValidation(n, inputs, targets, k, testf):
	ver_errors = []
	train_errors = []
	for i in range(n):
		ver_error, train_error = kFoldValidation(inputs, targets, k, testf)
		
		ver_errors.append( np.mean(ver_error) )
		train_errors.append( np.mean(train_error) )
	
	return ver_errors, train_errors

#NETWORK INIT
[inputs, targets] = readDataset("data.csv")
nin = len(inputs[0]) 
nout = len(targets[0])
nhiddens = [30]

inputs = np.asarray(inputs)
targets = np.asarray(targets)
inputs_min = inputs.min(axis=0)
inputs_max = inputs.max(axis=0)

nlayers = len(nhiddens) + 1
mlp = MLP(nin, nout, nhiddens, inputs_min, inputs_max)
# mlp.transfs = [trans.LogSig()] * nlayers

learningRate = 0.08
goal = 0.005
epochs = 100
k = 5

#sigmoid tests
netTester = NetTester(mlp, goal, epochs, learningRate)
tests_num = 10
ver_means, train_errors = multipleKFoldValidation(tests_num, inputs, targets, k, netTester)

#tanh tests
mlp.transfs = [trans.TanSig()] * nlayers
netTester = NetTester(mlp, goal, epochs, learningRate)
ver_means2, train_errors2 = multipleKFoldValidation(tests_num, inputs, targets, k, netTester)

#plot data: validation error
v1 = np.mean(ver_means)
v2 = np.mean(ver_means2)
v = [v1, v2]
xlabels = ["sigmoid", "tanSig"]

#plot data: training error
t1 = np.mean(train_errors)
t2 = np.mean(train_errors2)
t = [t1, t2]

import charts
#wykres dla kazdego błędu w kazdym stepie
#charts.bar_plot(np.arange(k), mean, "kfold step", "validation error", "plot")

#wykres srednich błędów walidacji dla roznych funkcji aktywacji
charts.bar_plot("acc_r{:f}_g{:f}_e{:.0f}_l{:.0f}.png".format(learningRate, goal, epochs, nhiddens[0]), xlabels, v, "activation functions", "validation error", "mean validation error plot", "true")

#wykres srednich błędów trenowania dla roznych funkcji aktywacji
charts.bar_plot("err_r{:f}_g{:f}_e{:.0f}_l{:.0f}.png".format(learningRate, goal, epochs, nhiddens[0]), xlabels, t, "activation functions", "training error", "mean training error plot")

#wykres z podzialem na grupy
#charts.bar_plot2(x, y, xlabels, "kfold step", "validation error", "plot", "sigmoid f", "funkcja2")
