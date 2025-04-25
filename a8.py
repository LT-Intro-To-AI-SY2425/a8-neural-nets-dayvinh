from neural import *

# print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

# xor_training_data = ([0, 0], [0]),([0, 1], [1]),([1, 0], [1]),([1, 1], [0]),

# nn = NeuralNet(2, 2, 1)
# nn.train(xor_training_data)
# print(nn.test_with_expected(xor_training_data))

# print("<<<<<<<<<<<<<< nu XOR for Q2>>>>>>>>>>>>>>\n")

# xor_training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

# nn = NeuralNet(2, 1, 1)
# nn.train(xor_training_data)
# print(nn.test_with_expected(xor_training_data))

#print("<<<<<<<<<<<<<< nu XOR for Q1>>>>>>>>>>>>>>\n")

# xor_training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

# nn = NeuralNet(2, 1, 1)
# nn.train(xor_training_data)
# print(nn.test_with_expected(xor_training_data))


print("<<<<<<<<<<<<<< nu XOR for politics - table 2 >>>>>>>>>>>>>>\n")

xor_training_data = [([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]), ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]), ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])]

nn = NeuralNet(5, 8, 1)
nn.train(xor_training_data)
print(nn.test_with_expected(xor_training_data))

print("<<<<<<<<<<<<<< nu XOR for politics - table 3 >>>>>>>>>>>>>>\n")

xor_training_data = [([1.0, 1.0, 1.0, 0.1, 0.1]), ([0.5, 0.2, 0.1, 0.7, 0.7]), ([0.8, 0.3, 0.3, 0.3, 0.8]), ([0.8, 0.3, 0.3, 0.8, 0.3]), ([0.9, 0.8, 0.8, 0.3, 0.6])]

nn = NeuralNet(5, 8, 1)

print(nn.test(xor_training_data))

