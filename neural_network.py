import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class neuralNetwork:
    def __init__(self, inputenodes, hiddennodes, outputenodes,learningrate):
        #set number of nodes in each input, hidden & output nodes
        self.inodes = inputenodes
        self.hnodes = hiddennodes
        self.onodes = outputenodes
        self.lr = learningrate
        
        #link weight matrices wih & who
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list):
        #convert input list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into output layers
        final_inputs = np.dot(self.who, hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        
        #output layer error is the (target - actual)
        output_errors = targets - final_outputs
        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        #update the weights for the links between the hidden and output layers
        self.who += self.lr*np.dot((output_errors*final_outputs*(1 - final_outputs)), np.transpose(hidden_outputs))
        #update the weights for the links between the input and hidden layers
        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1 - hidden_outputs)), np.transpose(inputs))
        pass
    
    def query(self, inputs_list):
        #convert input list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        
        #calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into output layers
        final_inputs = np.dot(self.who, hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


input_nodes = 784
hidden_nodes = 300
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass



test_data_file = open("mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
    
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass


scorecard_array = np.asarray(scorecard)
print ("Accuracy = ", scorecard_array.sum() / scorecard_array.size)





