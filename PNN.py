import numpy as np

class PNN(object):
	def __init__(self, inN, outN, hN):
		hN = np.concatenate([[inN], hN, [outN]])

		self.weights = []

		mean = 0.0
		stddev = 2 ** -0.5
		for k in range(hN.size - 1):
			width = hN[k+1]
			height = hN[k]
			weight = np.random.normal(mean, stddev, [height, width])
			self.weights.append(weight)

		self.activateFunctionMapper = np.vectorize(self.sigmoid)

	def forwardPropagation(self, inputs):
		outputs = []
		for w in self.weights:
			inputs = inputs.dot(w)
			output = self.activateFunctionMapper(inputs)
			outputs.append(output)

		return outputs

	def calculateGradient(self, outputs):
		# sigmoid gradient
		return outputs * (1 - outputs)

	def backwardPropagation(self, inputs, outputs, expectedOutputs, learningRate):
		outputs.insert(0, inputs)

		currentOutputs = outputs[-1]
		print "currentOutputs", currentOutputs, "expectedOutputs", expectedOutputs
		error = expectedOutputs - currentOutputs
		delta = error

		# print "expectedOutputs", expectedOutputs

		for idx in reversed(range(len(self.weights))):
			weight = self.weights[idx]
			nextDelta = weight.dot(delta)

			output = outputs[idx]
			gradient = self.calculateGradient(output)

			weightGrowth = learningRate * (output.T.dot(delta.T))

			# print "idx", idx
			# print "weight", weight
			# print "output", output
			# print "delta", delta
			# print "gradient", gradient
			# print "weightGrowth", weightGrowth
			# print "----"

			weight += weightGrowth
			self.weights[idx] += weight

			delta = nextDelta

		return error

	def singleTrain(self, inputs, expectedOutputs, learningRate):
		outputs = self.forwardPropagation(inputs)
		error = self.backwardPropagation(inputs, outputs, expectedOutputs, learningRate)
		return error

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))