from neuron import Neuron
from network import Network

def main():
    topology = [2, 3, 1]
    net = Network(topology)
    Neuron.eta = 0.09
    Neuron.alpha = 0.015
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [0], [0], [1]] # AND gate
    while True:
        err = 0
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForward()
            net.backPropagate(outputs[i])
            err += net.getError(outputs[i])
        print("Error: ", err)
        if(err < 0.01):
            break
    while True:
        a = int(input("Enter first input : "))
        b = int(input("Enter second input: "))
        net.setInput((a, b))
        net.feedForward()
        print(net.getThresholdResults())

if(__name__ == "__main__"):
    main()
