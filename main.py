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
	def __init__(self, net):
		self.net = net

	def __call__(self, train_inputs, train_targets, ver_inputs, ver_targets):
		assert len(ver_inputs) == len(ver_targets)
		assert len(train_inputs) == len(train_targets)

		self.net.init()

		# train again
		learningRate = 0.01
		goal = 0.01
		epochs = 5001
		def monitf(epoch, error):
			if (epoch % 100) == 0:
				print("Epoch no#", epoch, "error = ", error)

		self.net.train(train_inputs, train_targets, goal, epochs, learningRate, monitf)

		# verify trained net
		ver_outputs = [self.net.simulate(inp) for inp in ver_inputs]
		accuracies = [self.calcAccuracy(tar, out) 
			for [tar,out] in zip(ver_targets, ver_outputs)]

		# calculate mean
		mean = sum(accuracies) / len(accuracies)
		return mean

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

	errors = []
	for i in range(k):
		train_inputs = inputs_clusters[i]
		train_targets = targets_clusters[i]
		ver_inputs = np.vstack(inputs_clusters[:i] + inputs_clusters[(i+1):])
		ver_targets = np.vstack(targets_clusters[:i] + targets_clusters[(i+1):])

		error = testf(train_inputs, train_targets, ver_inputs, ver_targets)
		errors.append(error)
		print("K-Fold step#", i, "net accuracy = ", error*100, "%")

	return errors


[inputs, targets] = readDataset("data.csv")
nin = len(inputs[0]) 
nout = len(targets[0])
nhiddens = [5]
inputs_min = min(inputs)
inputs_max = max(inputs)
nlayers = len(nhiddens) + 1
transfs = [trans.LogSig()] * nlayers
mlp = MLP(nin, nout, nhiddens, inputs_min, inputs_max, transfs)

netTester = NetTester(mlp)
k = 8
mean = kFoldValidation(inputs, targets, k, netTester)
print("Mean error:", mean)