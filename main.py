import numpy as np

import csv
import copy

def readDataset(filename):
	file = open(filename)
	reader = csv.reader(file)
	header = next(reader) # the first line is the header

	# row[0] is ID - not needed
	# row[1] is output
	# row[2:] are inputs

	def readInputs(row):
		return np.array([float(param) for param in row[2:]])

	def readOutputs(row):
	    if row[1] == 'M':
	        outputs = np.array([0.0])
	    else:
	        outputs = np.array([1.0])
	    return outputs

	# read first row and initialize lists and minmax
	firstRow = next(reader)
	outputsList = [readOutputs(firstRow)]
	inputsList = [readInputs(firstRow)]

	inputsMinMax = [[value, value] for value in inputsList[0]]
	outputsMinMax = [[0.0, 1.0]]

	for row in reader:
		inputs = readInputs(row)
		outputs = readOutputs(row)

		inputsList.append(inputs)
		outputsList.append(outputs)

		for i in range(len(inputs)):
			value = inputs[i]
			minmax = inputsMinMax[i]
			minmax[0] = min(minmax[0], value)
			minmax[1] = max(minmax[1], value)
			inputsMinMax[i] = minmax

	inputsList = np.array(inputsList)
	outputsList = np.array(outputsList)

	return inputsList, outputsList, inputsMinMax, outputsMinMax


class PNN(object):
	def __init__(self, inputsMinMax, outputsMinMax, hidden):
		self.inN = len(inputsMinMax)
		self.outN = len(outputsMinMax)
		self.hidden = hidden

		self.activateFunction = np.vectorize(self.sigmoid)
		self.initializeWeights()
		self.initializeMinMax(inputsMinMax, outputsMinMax)

	# initializes weights for both hidden and output layers
	def initializeWeights(self):
		self.weightsList = []

		# initialize weights for hidden layers as random (-1, 1)
		minRand = -1
		maxRand = 1
		width = self.inN
		for k in range(len(self.hidden)):
			height = self.hidden[k]
			weights = np.random.uniform(minRand, maxRand, [height, width])
			self.weightsList.append(weights)
			width = height

		# initialize weights for output layers as zeros
		height = self.outN
		weights = np.zeros([height, width])
		self.weightsList.append(weights)

	def initializeMinMax(self, inputsMinMax, outputsMinMax):
		inputsFromMin = []
		inputsFromMax = []
		for minmax in inputsMinMax:
			inputsFromMin.append(minmax[0])
			inputsFromMax.append(minmax[1])
		self.inputsFromMin = np.array(inputsFromMin)
		self.inputsFromMax = np.array(inputsFromMax)

		outputsFromMin = []
		outputsFromMax = []
		for minmax in outputsMinMax:
			outputsFromMin.append(minmax[0])
			outputsFromMax.append(minmax[1])
		self.outputsFromMin = np.array(outputsFromMin)
		self.outputsFromMax = np.array(outputsFromMax)

		self.inputsToMin = -1
		self.inputsToMax = 1
		hiddenTotal = sum(self.hidden)
		self.outputsToMin = -np.sqrt(hiddenTotal)
		self.outputsToMax = np.sqrt(hiddenTotal)

	# maps value from one range to another
	@staticmethod
	def remap(value, fromMin, fromMax, toMin, toMax):
		return (value-fromMin) * (toMax-toMin) / (fromMax-fromMin) + toMin

	# maps inputs from dataset's range to network's range (-1, 1)
	def scaleInputs(self, inputs):
		return PNN.remap(
			inputs, 
			self.inputsFromMin, self.inputsFromMax,
			self.inputsToMin, self.inputsToMax
			)

	# maps outputs from dataset's range to network's range (-sqrt(hn), sqrt(hn))
	def scaleOutputs(self, outputs):
		return PNN.remap(
			outputs,
			self.outputsFromMin, self.outputsFromMax,
			self.outputsToMin, self.outputsToMax
			)

	# calculates list of outputs for each layer
	def forwardPropagation(self, inputs):
		outputsList = []
		for weights in self.weightsList:
			summary = weights.dot(inputs.T)
			outputs = self.activateFunction(summary.T)
			outputsList.append(outputs)
			inputs = outputs

		return outputsList

	# same as forwardPropagation, but without creating outputsList
	# at the end, it scales outputs back to the dataset's range
	def calculateOutputs(self, inputs):
		inputs = self.scaleInputs(inputs)
		for weights in self.weightsList:
			summary = weights.dot(inputs.T)
			outputs = self.activateFunction(summary.T)
			inputs = outputs

		return outputs

	# activation function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# derivative of activation function
	def calculateGradients(self, outputs):
		# sigmoid gradient
		return outputs * (1 - outputs)

	def backwardPropagation(self, inputs, expectedOutputs, outputsList, learningRate):
		# calculate deltas for output layers
		currentOutputs = outputsList[-1]
		errors = expectedOutputs - currentOutputs
		deltas = errors 
		deltasList = [deltas]
		
		# calculate deltas for hidden layers
		for weights in reversed(self.weightsList[1:]):
			deltas = deltas.dot(weights)
			deltasList.insert(0, deltas)

		# update weights
		for i in range(len(self.weightsList)):
			weights = self.weightsList[i]
			deltas = deltasList[i]
			outputs = np.array(outputsList[i])
			gradients = self.calculateGradients(outputs)

			derivative = deltas * gradients
			weightsGrowth = learningRate * (derivative.T.dot(inputs))
			weights += weightsGrowth
			self.weightsList[i] = weights

			inputs = outputs

		return errors

	# trains net once with whole inputs and expectedOutputs list
	# at the end, calculates Mean Square Error of training process
	def singleTrain(self, inputsList, expectedOutputsList, learningRate):
		errorsSum = np.zeros([1, self.outN])
		for k in range(len(inputsList)):
			inputs = inputsList[k].reshape(1, self.inN)
			inputs = self.scaleInputs(inputs)

			expectedOutputs = expectedOutputsList[k].reshape(1, self.outN)

			outputsList = self.forwardPropagation(inputs)
			errors = self.backwardPropagation(inputs, expectedOutputs, outputsList, learningRate)
			errorsSum += (errors ** 2)
			
		mse = errorsSum / len(inputsList)
		return mse

	# performs full training process, approaching goal or number of epochs
	def train(self, inputsList, expectedOutputsList, goal, epochs, learningRate, monit):
		# assert len(inputsList) == len(expectedOutputsList)
		elapsedToMonit = monit
		for i in range(epochs):
			# shuffle training set
			p = np.random.permutation(len(inputsList))
			inputsList = inputsList[p]
			expectedOutputsList = expectedOutputsList[p]

			# perform single train
			mse = self.singleTrain(inputsList, expectedOutputsList, learningRate)
			if(mse < goal):
				break

			elapsedToMonit -= 1
			if(elapsedToMonit == 0):
				print "Epoch: ", i+1," MSE: ", mse
				elapsedToMonit = monit

		print "Finished training in: ",i+1
		return mse

learningRate = 0.05
goal = 0.001
epochs = 2000
hidden = [3]
inputsList, outputsList, inputsMinMax, outputsMinMax = readDataset("data.csv")

pnn = PNN(inputsMinMax, outputsMinMax, hidden)

# train net
size = len(inputsList)
errors =  pnn.train(
	inputsList[0:size/2], outputsList[0:size/2], 
	goal, epochs, learningRate, 10
	)
