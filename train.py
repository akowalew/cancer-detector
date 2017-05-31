import numpy as np

class TrainerBase(object):
	def __init__(self):
		raise "not implemented"

	def __call__(self):
		raise "not implemented"


class GdBackpropagationTrainer(TrainerBase):
	def __init__(self):
		super(TrainerBase, self).__init__()

	# performs full training process, approaching goal or number of epochs
	def __call__(self, net, inputsdata, targetsdata, goal, epochs, learn_rate, monitf=None):
		inputsdata = np.array(inputsdata, copy=False)
		targetsdata = np.array(targetsdata, copy=False)

		for epoch in range(1, epochs):
			# shuffle training set
			p = np.random.permutation(len(inputsdata))
			inputsdata = inputsdata[p]
			targetsdata = targetsdata[p]

			# perform single train
			error = self.singleTrain(net, inputsdata, targetsdata, learn_rate)
			if monitf != None:
				monitf(epoch, error)

			if(error < goal):
				break
		return error

	# trains net once with whole inputs and targets list.
	# At the end, returns error of training process
	def singleTrain(self, net, inputsdata, targetsdata, learn_rate):
		errorsSum = np.zeros([1, net.nout])
		for k in range(len(inputsdata)):
			inputs = inputsdata[k].reshape(1, net.nin)
			targets = targetsdata[k].reshape(1, net.nout)

			inputs = net.scaleInputs(inputs)

			outputsList = net.forwardPropagation(inputs)
			errors = self.backwardPropagation(net, inputs, targets, outputsList, learn_rate)

			errorsSum += (errors ** 2)
			
		# Mean Square Error
		error = errorsSum / len(inputsdata)
		return error

	def backwardPropagation(self, net, inputs, targets, outputsList, learn_rate):
		# calculate errors for output layer
		netOutputs = outputsList[-1]
		errors = targets - netOutputs
		
		# calculate deltas for each hidden layer 
		# starting from the last hidden layer of the net
		deltas = errors 
		deltasList = [deltas]
		for weights in reversed(net.weightsList[1:]):
			deltas = deltas.dot(weights)
			deltasList.insert(0, deltas)

		# update weights
		for i in range(net.nlayers):
			weights = net.weightsList[i]
			deltas = deltasList[i]
			outputs = outputsList[i]
			transf = net.transfs[i]
			gradients = transf.gradient(outputs)

			derivative = deltas * gradients
			weightsGrowth = learn_rate * (derivative.T.dot(inputs))
			weights += weightsGrowth
			net.weightsList[i] = weights

			inputs = outputs

		return errors		
