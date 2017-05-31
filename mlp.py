import trans

import numpy as np
from train import GdBackpropagationTrainer

# MLP - Multi-Layer-Perceptron
class MLP(object):
	def __init__(self, nin, nout, nhiddens, inputs_min, inputs_max):
		self.nin = nin
		self.nout = nout
		self.nhiddens = nhiddens

		self.nlayers = len(nhiddens) + 1 # hidden layers plus output layer

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

		self.transfs = [trans.LogSig()] * self.nlayers

		self.trainf = GdBackpropagationTrainer()

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

	# maps outputs from dataset's range to network's range (-sqrt(hn), sqrt(hn))
	def scaleOutputs(self, outputs):
		return MLP.remap(
			outputs,
			self.outputs_min, self.outputs_max,
			self.outputsmin, self.outputsmax
			)

	def train(self, *args, **kwargs):
		return self.trainf(self, *args, **kwargs)

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

		return outputsList

	# Simulates net's work. Calculates outputs for last layer.
	# Same as forwardPropagation, but without creating outputsList
	def simulate(self, inputs):
		inputs = self.scaleInputs(inputs)
		outputs = self.forwardPropagation(inputs)
		# TODO: scale outputs if needed
		return outputs[-1]