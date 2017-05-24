import numpy as np

class PNN(object):
	def __init__(self, inN, outN, hN):
		self.inN = inN
		self.outN = outN

		layers = np.concatenate([[inN], hN, [outN]])

		self.weightsList = []

		minRand = -0.5
		maxRand = 0.5
		for k in range(layers.size - 1):
			width = layers[k]
			height = layers[k+1]
			weights = np.random.uniform(minRand, maxRand, [height, width])
			self.weightsList.append(weights)

		self.activateFunctionMapper = np.vectorize(self.sigmoid)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def forwardPropagation(self, inputs):
		outputsList = []
		for weights in self.weightsList:
			inputs = weights.dot(inputs)
			outputs = self.activateFunctionMapper(inputs)
			outputsList.append(outputs)

		return outputsList

	def calculateGradients(self, outputs):
		# sigmoid gradient
		return outputs * (1 - outputs)

	def backwardPropagation(self, inputs, targetOutputs, outputsList, learningRate):
		outputsList.insert(0, inputs)

		currentOutputs = outputsList[-1]
		errors = targetOutputs - currentOutputs
		deltas = errors.T

		# calculate deltas
		deltasList = [deltas]
		for weights in reversed(self.weightsList[1:]):
			deltas = deltas.dot(weights)
			deltasList.insert(0, deltas)

		for i in range(len(self.weightsList)):
			weights = self.weightsList[i]

			inputs = np.array(outputsList[i])
			outputs = np.array(outputsList[i+1])
			gradients = self.calculateGradients(outputs)

			deltas = deltasList[i]
			
			weightsGrowth = learningRate * ((deltas.T * gradients).dot(inputs.T))

			weights = weights + weightsGrowth
			self.weightsList[i] = weights

		return abs(errors)

	def singleTrain(self, inputs, targetOutputs, learningRate):
		outputsList = self.forwardPropagation(inputs)
		errors = self.backwardPropagation(inputs, targetOutputs, outputsList, learningRate)
		return errors

	def train(self, inputsList, targetsList, goal, epochs, learningRate, step):
		assert len(inputsList) == len(targetsList)

		elapsedToMonit = step
		for i in range(epochs):
			p = np.random.permutation(len(inputsList))
			inputsList = inputsList[p]
			targetsList = targetsList[p]

			errorsSum = []
			for k in range(len(inputsList)):
				inputs = inputsList[k].reshape(1, self.inN)
				targets = targetsList[k].reshape(1, self.outN)
				errors = pnn.singleTrain(inputs, targets, learningRate)
				if(len(errorsSum) == 0):
					errorsSum = errors
				else:
					errorsSum = errorsSum + (errors ** 2)
				
			mse = errorsSum / len(inputsList)
			if(mse < goal):
				break

			elapsedToMonit = elapsedToMonit - 1
			if(elapsedToMonit == 0):
				elapsedToMonit = step
				print "Epoch: ", i+1," Errors: ", mse

		print i+1
		return mse		


def fun(x):
	return sin(x)

inN = 1
outN = 1
hN = [3]
learningRate = 0.01
goal = 0.02
epochs = 2000
pnn = PNN(inN, outN, hN)

# generate train set
size = 50
x = np.linspace(-7, 7, size)
y = np.sin(x)

inp = x.reshape(size, 1)
tar = y.reshape(size, 1)

# train net
errors =  pnn.train(inp, tar, goal, epochs, learningRate, 10)
