from mlp import MLP
import trans
import csv

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


[inputs, targets] = readDataset("data.csv")
nin = len(inputs[0]) 
nout = len(targets[0])
nhiddens = [5]
inputs_min = min(inputs)
inputs_max = max(inputs)
nlayers = len(nhiddens) + 1
transfs = [trans.TanSig()] * nlayers
mlp = MLP(nin, nout, nhiddens, inputs_min, inputs_max, transfs)

learningRate = 0.001
goal = 0.01
epochs = 10000
size = len(inputs)
errors =  mlp.train(inputs[0:size/8], targets[0:size/8], 
 	goal, epochs, learningRate, 10)


