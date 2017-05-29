import trans

import numpy as np

# MLP - Multi-Layer-Perceptron
class MLP(object):
	def __init__(self, nin, nout, nhiddens, inputs_min, inputs_max, transfs = None):
		self.nin = nin
		self.nout = nout
		self.nhiddens = nhiddens

		self.nlayers = len(nhiddens) + 1 # hidden layers plus output layer
		if transfs == None:
			self.transfs = [trans.LogSig()] * self.nlayers
		else:
			assert len(transfs) == self.nlayers
			self.transfs = transfs

		self.inputs_min = np.array(inputs_min)
		self.inputs_max = np.array(inputs_max)
		# self.outputs_min = np.array(outputs_min)
		# self.outputs_max = np.array(outputs_max)

		assert len(self.inputs_min) == nin
		assert len(self.inputs_max) == nin
		# assert len(self.outputs_min) == nout
		# assert len(self.outputs_max) == nout
		# ntotalhidden = sum(nhiddens)
		# self.outputsmin = -np.sqrt(ntotalhidden)
		# self.outputsmax = np.sqrt(ntotalhidden)

		self.init()


	def init(self):
		self.initializeWeights()	

		
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
		return MLP.remap(
			inputs, 
			self.inputs_min, self.inputs_max, 
			-1, 1
			)

	# # maps outputs from dataset's range to network's range (-sqrt(hn), sqrt(hn))
	# def scaleOutputs(self, outputs):
	# 	return MLP.remap(
	# 		outputs,
	# 		self.outputs_min, self.outputs_max,
	# 		self.outputsmin, self.outputsmax
	# 		)

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
			transf = self.transfs[i]
			gradients = transf.gradient(outputs)

			derivative = deltas * gradients
			weightsGrowth = learningRate * (derivative.T.dot(inputs))
			weights += weightsGrowth
			self.weightsList[i] = weights

			inputs = outputs

		return errors

	# trains net once with whole inputs and expectedOutputs list
	# at the end, calculates Mean Square Error of training process
	def singleTrain(self, inputsdata, targetsdata, learningRate):
		errorsSum = np.zeros([1, self.nout])
		for k in range(len(inputsdata)):
			inputs = inputsdata[k].reshape(1, self.nin)
			expectedOutputs = targetsdata[k].reshape(1, self.nout)

			inputs = self.scaleInputs(inputs)

			outputsList = self.forwardPropagation(inputs)
			errors = self.backwardPropagation(inputs, expectedOutputs, outputsList, learningRate)

			# calculate MSE
			errorsSum += (errors ** 2)
			
		totalError = errorsSum / len(inputsdata)
		return totalError

	# performs full training process, approaching goal or number of epochs
	def train(self, inputsdata, targetsdata, goal, epochs, learningRate, monitf=None):
		inputsdata = np.array(inputsdata, copy=False)
		targetsdata = np.array(targetsdata, copy=False)

		for epoch in range(1, epochs):
			# shuffle training set
			p = np.random.permutation(len(inputsdata))
			inputsdata = inputsdata[p]
			targetsdata = targetsdata[p]

			# perform single train
			error = self.singleTrain(inputsdata, targetsdata, learningRate)
			if monitf != None:
				monitf(epoch, error)

			if(error < goal):
				break

		return error

	# Simulates net's work. Calculates outputs for last layer.
	# Same as forwardPropagation, but without creating outputsList
	def simulate(self, inputs):
		inputs = self.scaleInputs(inputs)
		outputs = self.forwardPropagation(inputs)
		return outputs[-1]