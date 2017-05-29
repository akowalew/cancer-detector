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
	outputsdata = [readOutputs(firstRow)]
	inputsdata = [readInputs(firstRow)]

	# output is binary, arleady known
	outputssize = len(outputsdata[0])
	outputsminima = [0.0]
	outputsmaxima = [1.0]

	inputssize = len(inputsdata[0])
	inputsminima = [float('Inf')] * inputssize
	inputsmaxima = [float('-Inf')] * inputssize

	minimaf = np.vectorize(min)
	maximaf = np.vectorize(max)

	for row in reader:
		inputs = readInputs(row)
		outputs = readOutputs(row)
		assert len(inputs) == inputssize
		assert len(outputs) == outputssize

		inputsdata.append(inputs)
		outputsdata.append(outputs)

		inputsminima = minimaf(inputsminima, inputs)
		inputsmaxima = maximaf(inputsmaxima, inputs)

	inputsdata = np.array(inputsdata)
	outputsdata = np.array(outputsdata)
	return {
		'inputsdata': inputsdata,
		'outputsdata': outputsdata,
		'minmax': {
			'inputsmaxima': inputsmaxima,
			'inputsminima': inputsminima,
			'outputsminima': outputsminima,
			'outputsmaxima': outputsmaxima
		}
	}


class PNN(object):
	def __init__(self, nin, nout, nhiddens = [], minmax = None, transfs = None):
		self.nin = nin
		self.nout = nout
		self.nhiddens = nhiddens
		self.initializeWeights()	

		self.nlayers = len(nhiddens) + 1 # hidden layers plus output layer
		if transfs == None:
			self.transfs = [PNN.sigmoid] * self.nlayers
		else:
			assert len(transfs) == nlayers
			self.transfs = transfs

		if minmax != None:
			assert len(minmax['inputsminima']) == nin
			assert len(minmax['inputsmaxima']) == nin
			assert len(minmax['outputsminima']) == nout	
			assert len(minmax['outputsmaxima']) == nout
			ntotalhidden = sum(nhiddens)
			self.outputsmin = -np.sqrt(ntotalhidden)
			self.outputsmax = np.sqrt(ntotalhidden)
		self.minmax = minmax

		
	# initializes weights for both nhiddens and output layers
	def initializeWeights(self):
		weightsList = []

		# initialize weights for each hidden layer as random (-1, 1)
		minRand = -1
		maxRand = 1
		width = self.nin
		for nhidden in self.nhiddens:
			height = nhidden
			weights = np.random.uniform(minRand, maxRand, [height, width])
			weightsList.append(weights)
			width = height

		# initialize weights for output layer as zeros
		outputweights = np.zeros([self.nout, width])
		weightsList.append(outputweights)

		self.weightsList = weightsList

	# maps value from one range to another
	@staticmethod
	def remap(value, fromMin, fromMax, toMin, toMax):
		return (value-fromMin) * (toMax-toMin) / (fromMax-fromMin) + toMin

	# maps inputs from dataset's range to network's range (-1, 1)
	def scaleInputs(self, inputs):
		assert self.minmax != None
		return PNN.remap(
			inputs, 
			self.minmax['inputsminima'], self.minmax['inputsmaxima'], 
			-1, 1
			)

	# maps outputs from dataset's range to network's range (-sqrt(hn), sqrt(hn))
	def scaleOutputs(self, outputs):
		assert self.minmax != None
		return PNN.remap(
			outputs,
			self.minmax['outputsminima'], self.minmax['outputsmaxima'],
			self.outputsmin, self.outputsmax
			)

	# calculates list of outputs for each layer
	def forwardPropagation(self, inputs):
		outputsList = []
		for i in range(self.nlayers):
			# sum all inputs multiplied by its weights
			weights = self.weightsList[i]
			summary = weights.dot(inputs.T)

			# pass through activate function
			transf = self.transfs[i]
			outputs = transf(summary.T)

			# save outputs and set current outputs as next inputs
			outputsList.append(outputs)
			inputs = outputs

		# TODO: scale the last output!

		return outputsList

	def backwardPropagation(self, inputs, expectedOutputs, outputsList, learningRate):
		# calculate errors for output layer
		netOutputs = outputsList[-1]
		errors = expectedOutputs - netOutputs
		
		# calculate deltas for each hidden layer 
		# starting from the last hidden layer of the net
		deltas = errors 
		deltasList = [deltas]
		for weights in reversed(self.weightsList[1:]):
			deltas = deltas.dot(weights)
			deltasList.insert(0, deltas)

		# update weights
		for i in range(self.nlayers):
			weights = self.weightsList[i]
			deltas = deltasList[i]
			outputs = outputsList[i]
			gradients = PNN.calculateGradients(outputs)

			derivative = deltas * gradients
			weightsGrowth = learningRate * (derivative.T.dot(inputs))
			weights += weightsGrowth
			self.weightsList[i] = weights

			inputs = outputs

		return errors

	# trains net once with whole inputs and expectedOutputs list
	# at the end, calculates Mean Square Error of training process
	def singleTrain(self, inputsList, expectedOutputsList, learningRate):
		errorsSum = np.zeros([1, self.nout])
		for k in range(len(inputsList)):
			inputs = inputsList[k].reshape(1, self.nin)
			expectedOutputs = expectedOutputsList[k].reshape(1, self.nout)
			
			inputs = self.scaleInputs(inputs)

			outputsList = self.forwardPropagation(inputs)
			errors = self.backwardPropagation(inputs, expectedOutputs, outputsList, learningRate)
			errorsSum += (errors ** 2)
			
		mse = errorsSum / len(inputsList)
		return mse

	# performs full training process, approaching goal or number of epochs
	def train(self, inputsList, expectedOutputsList, goal, epochs, learningRate, monit):
		elapsedToMonit = monit
		for i in range(epochs):
			# # shuffle training set
			# p = np.random.permutation(len(inputsList))
			# inputsList = inputsList[p]
			# expectedOutputsList = expectedOutputsList[p]

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

	# Simulates net's work. Calculates outputs for last layer.
	# Same as forwardPropagation, but without creating outputsList
	def simulate(self, inputs):
		inputs = self.scaleInputs(inputs)
		outputsList = forwardPropagation(inputs)
		return outputs[-1]

	# activation function
	@staticmethod
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	# derivative of activation function
	@staticmethod
	def calculateGradients(outputs):
		# sigmoid gradient
		return outputs * (1 - outputs)

# def kFoldTesting(dataset, k, testf):

dataset = readDataset("data.csv")
inputsdata = dataset['inputsdata']
outputsdata = dataset['outputsdata']
nin = len(inputsdata[0]) 
nout = len(outputsdata[0])
nhiddens = [5]
minmax = dataset['minmax']
pnn = PNN(nin, nout, nhiddens, minmax)

# train net
learningRate = 0.01
goal = 0.01
epochs = 5000
size = len(inputsdata)
errors =  pnn.train(
	inputsdata[0:size/8], outputsdata[0:size/8], 
	goal, epochs, learningRate, 10
	)
