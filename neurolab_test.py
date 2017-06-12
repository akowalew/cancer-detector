from mlp import MLP
import trans
import csv
import numpy as np
import neurolab as nl


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
	
def kFoldValidation(net, inputs, targets, k):
	inputs_clusters = np.array_split(inputs, k)
	targets_clusters = np.array_split(targets, k)

	ver_errors = []
	train_errors = []
	for i in range(k):
		train_inputs = inputs_clusters[i]
		train_targets = targets_clusters[i]
		ver_inputs = np.vstack(inputs_clusters[:i] + inputs_clusters[(i+1):])
		ver_targets = np.vstack(targets_clusters[:i] + targets_clusters[(i+1):])
		
		
		net.init()
		error = net.train(train_inputs, train_targets, epochs=10000, show=100, goal=0.01) #, lr=0.01
		outputs = []
		for j in range(len(ver_inputs)):
			out = net.step(ver_inputs[j])
			print("INPUTS:", ver_inputs[j])
			print("OUTPUTS:", out)
			input("Press Enter to continue...")
			outputs.append(out)
		#out = net.sim(ver_inputs)
		error_fun = nl.error.MSE()
		errors = error_fun(out, ver_targets)
		train_errors.append(error)
		#print("Kfold step:", k, "error:", errors)
		input("Press Enter to continue...")
		#print(out)
		
		net.reset()
	return train_errors

[inputs, targets] = readDataset("data.csv")
nin = len(inputs[0]) 
nout = len(targets[0])
nhiddens = [20, 1]
functions = [nl.trans.LogSig(), nl.trans.LogSig()]


inputs = np.asarray(inputs)
targets = np.asarray(targets)
inputs_min = inputs.min(axis=0)
inputs_max = inputs.max(axis=0)
inputs_minmax = []
for i in range(len(inputs_max)):
	inputs_minmax.append([inputs_min[i], inputs_max[i]])

#ANN INIT
net = nl.net.newff(inputs_minmax, nhiddens, functions)
net.init()
net.trainf = nl.train.train_gd

k = 2
kFoldValidation(net, inputs, targets, k)

#print(inputs)
#net.trainf = nl.train.train_gd

#error = net.train(inputs, targets, epochs=100, show=1, goal=0.01, lr=0.01)
#print(error)
#out = net.sim(inputs)
#print(out)
#errors = nl.error.MSE()
#print(errors(error, out))






