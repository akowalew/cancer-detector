import PNN
import numpy as np

# approximated function is (x+y)^2
def fun(x,y):
	return (x+y)*(x+y)

def main():
	inN = 2
	outN = 1
	hN = [3]
	learningRate = 0.01
	pnn = PNN.PNN(inN, outN, hN)
	
	# generate train set
	learnSize = 10
	learnInputs = []
	learnOutputs = []
	for k in range(learnSize):
		inputs = np.random.normal(0, 1, (1, inN))
		outputs = np.array([[fun(inputs[0][0], inputs[0][1])]])
		learnInputs.append(inputs)
		learnOutputs.append(outputs)

	# train net
	for k in range(learnSize):
		error = pnn.singleTrain(learnInputs[k], 
			learnOutputs[k], 
			learningRate)
		print "[x,y] = ", learnInputs[k], "(x+y)^2 = ", learnOutputs[k], "e = ", error


if __name__ == "__main__":
    main()