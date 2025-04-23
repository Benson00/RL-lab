import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
import tensorflow as tf; import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
import matplotlib.pyplot as plt
from DangerousGridWorld import GridWorld


def set_same_weights(keras_model, pytorch_model):
	keras_weights = [layer.get_weights() for layer in keras_model.layers]
	with torch.no_grad():
		for i, layer in enumerate(pytorch_model.children()):
			if isinstance(layer, nn.Linear):
				layer.weight = nn.Parameter(torch.tensor(keras_weights[i][0].T, requires_grad=True))
				layer.bias = nn.Parameter(torch.tensor(keras_weights[i][1], requires_grad=True))
	
def mse(network, dataset_input, target):
	"""
	Compute the MSE loss function
	"""

	# Compute the predicted value, over time this value should
	# be more close to the expected output (i.e., target)
	predicted_value = network( dataset_input )
	
	# Compute MSE between the predicted and the expected value
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)
	
	# Return the averaged values for computational optimization
	return mse


def objective(x, y):
	"""
	Implements the following simple 2-variables function to optimize:
		2x^2 + 2xy + 2y^2 - 6x

	"""
	return 2*x**2 + 2*x*y + 2*y**2 - 6*x


def find_minimum_keras(objective_function, n_iter=5000):
	"""
	Function that find the assignements to the variables that minimize the objective function,
	exploiting TensorFlow.

	Args:
		objective_function: the objective function to minimize
		n_iter: rnumber of iteration for the gradient descent process
		
	Returns:
		x: the best assignement for variable 'x'
		y: the best assignement for variable 'y'

	"""
	
	x = tf.Variable(0.0, name='x')
	y = tf.Variable(0.0, name='y')
	optimizer = tf.keras.optimizers.SGD( learning_rate=0.001 )
	#
	# YOUR CODE HERE!
	#

	for _ in range(n_iter):

		with tf.GradientTape() as tape:
			objective = objective_function(x,y)
			grad = tape.gradient(objective, [x,y])
			optimizer.apply_gradients( zip(grad, [x, y]) )

	return x.numpy(), y.numpy()
	

def find_minimum_torch(objective_function, n_iter=5000):
	"""
	Function that find the assignements to the variables that minimize the objective function,
	exploiting TensorFlow.

	Args:
		objective_function: the objective function to minimize
		n_iter: rnumber of iteration for the gradient descent process
		
	Returns:
		x: the best assignement for variable 'x'
		y: the best assignement for variable 'y'

	"""
	
	x = torch.tensor([0.0], requires_grad=True)
	y = torch.tensor([0.0], requires_grad=True)

	optimizer = optim.SGD([x, y], lr=0.001)
	#
	# YOUR CODE HERE!
	#
	for _ in range(5000):  
		objective = objective_function(x, y)
		optimizer.zero_grad()
		objective.backward()
		optimizer.step()
	return x.detach().numpy().item(), y.detach().numpy().item()

	
def create_DNN_keras(nInputs, nOutputs, nLayer, nNodes):
	"""
	Function that generates a neural network with Keras and specific parameters.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	Returns:
		model: the generated tensorflow model

	"""
	
	# Initialize the neural network
	model = Sequential()
	
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu"))
	model.add(Dense(nNodes, activation="relu")) 
	model.add(Dense(nNodes, activation="relu")) 
	model.add(Dense(nOutputs, activation="linear")) 
	return model


class TorchModel(nn.Module):
	"""
	Class that generates a neural network with PyTorch and specific parameters.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	"""

	# Initialize the neural network
	def __init__(self, nInputs, nOutputs, nLayer, nNodes):
		super(TorchModel, self).__init__()
		self.fc1 = nn.Linear(nInputs, nNodes)
		self.fc2 = nn.Linear(nNodes, nNodes)
		self.fc3 = nn.Linear(nNodes, nNodes)
		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.output(x)
		return x
	
	
def collect_random_trajectories(env, num_episodes=10):
	"""
	Function that collect a dataset from the environment with an iterative
	interaction process

	Args:
		env: the environment in the gym-like format on which collect the data
		num_episodes: number of episodes to perform in the environment
		
	Returns:
		memory_buffer: an array with the collected data

	"""
	
	memory_buffer = []

	for _ in range(num_episodes):
		state = env.random_initial_state()
		# uniform policy
		policy = [[1 / env.action_space for _ in range(env.action_space)] for _ in range(env.observation_space)]
		memory_buffer = env.sample_episode( policy, initial_state=state)
		#print(memory_buffer)

		
	return np.array(memory_buffer)


def train_DNN_keras(model, memory_buffer, epoch=20):

	"""
	Function that perform the gradient descent training loop based on the data collected;
	the objective is to generate a Keras neural network able to predict the reward of a state 
	given in input.

	Args:
		model: the initial model before the training phase
		memory_buffer: an array with the collected data
		epoch: number of gradient descent iteration
		
	Returns:
		model: Keras trained model

	"""
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

	# # Preprocess data
	dataset_input = np.vstack(memory_buffer[:, 1])
	target = np.vstack(memory_buffer[:, 2])

	model.compile(optimizer=optimizer,
                  loss='mse',  # Mean Squared Error, suitable for regression tasks
                  metrics=['mae'])  # Mean Absolute Error

    # Train the model
	model.fit(dataset_input, target, epochs=epoch, verbose=1)

	return model

	
def train_DNN_torch(model, memory_buffer, epoch=20):
	"""
	Function that perform the gradient descent training loop based on the data collected;
	the objective is to generate a PyTorch neural network able to predict the reward of a state 
	given in input.

	Args:
		model: the initial model before the training phase
		memory_buffer: an array with the collected data
		epoch: number of gradient descent iteration
		
	Returns:
		model: PyTorch trained model

	"""
	
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()

	# Preprocess data
	dataset_input = np.vstack(memory_buffer[:, 1])
	target = np.vstack(memory_buffer[:, 2])

	# Convert memory buffer to NumPy array
	memory_buffer = np.array(memory_buffer, dtype=object)

    # Extract inputs (states) and targets (rewards)
	dataset_input = np.vstack(memory_buffer[:, 1])
	target = np.vstack(memory_buffer[:, 2])

    # Convert to PyTorch tensors
	dataset_input = torch.tensor(dataset_input, dtype=torch.float32)
	target = torch.tensor(target, dtype=torch.float32)

    # Training loop
	model.train()
	for _ in range(epoch):
		optimizer.zero_grad()
		output = model(dataset_input)
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()

	return model


	

def main():
	print( "\n************************************************" )
	print( "*  Welcome to the sixth lesson of the RL-Lab!  *" )
	print( "*    (Tensorflow-PyTorch and Neural Networks)    *" )
	print( "**************************************************" )

	# PART 1) Non Linear Optimization
	x, y = find_minimum_keras(objective)
	print(f"\nA) The global minimum of the function: '2x^2 + 2xy + 2y^2 - 6x' using Keras is:")
	print(f"\t<x:{np.round(x, 2)}, y:{np.round(y, 2)}> with value {np.round(objective(x, y), 2)}")

	x, y = find_minimum_torch( objective )
	print(f"\nA) The global minimum of the function: '2x^2 + 2xy + 2y^2 - 6x' using PyTorch is:")
	print(f"\t<x:{np.round(x, 2)}, y:{np.round(y, 2)}> with value {np.round(objective(x, y), 2)}\n")

	# PART 2) Creating a Deep Neural Network using Keras and PyTorch
	print("\nB) Showing the deep neural networks structure:")
	dnn_model_keras = create_DNN_keras(nInputs=1, nOutputs=1, nLayer=2, nNodes=8)
	dnn_model_keras.summary()

	dnn_model_torch = TorchModel(nInputs=1, nOutputs=1, nLayer=2, nNodes=8)
	print(dnn_model_torch)

	# set the same weights and biases for the DNNs
	try:
		print("\nPre-conversion forward propagation of the value -1.4")
		print("Keras output: ", np.round(dnn_model_keras(np.array([[-1.4]])).numpy().item(),4))
		print("PyTorch output: ", np.round(dnn_model_torch(torch.tensor([-1.4])).item(),4))
		set_same_weights(dnn_model_keras, dnn_model_torch)
		print("Post-conversion forward propagation of the value -1.4")
		print("Keras output: ", np.round(dnn_model_keras(np.array([[-1.4]])).numpy().item(),4))
		print("PyTorch output: ", np.round(dnn_model_torch(torch.tensor([-1.4])).item(),4))
	except:
		print("Your Keras and PyTorch models are not the same! Check your functions and retry...")

	
	# PART 3) A Standard DRL Loop
	print("\nC) Collect a dataset from the interaction with the environment")
	env = GridWorld()
	memory_buffer = collect_random_trajectories(env, num_episodes=10)
	print(memory_buffer.shape)
	inp = np.array([[0], [48]])

	# PART 4) Train the DNN to predict the reward of given the state
	print("\nD) Training a DNN to predict the reward of a state:")
	
	out = dnn_model_keras(inp).numpy()
	print("Pre Training Reward Prediction Keras-PyTorch model: ")
	print(f"\tstate {inp[0][0]} => reward: {out[0][0]} ")
	print(f"\tstate {inp[1][0]} => reward: {out[1][0]} ")
	

	trained_dnn_model_keras = train_DNN_keras(dnn_model_keras, memory_buffer, epoch=2000)
	out = trained_dnn_model_keras(inp).numpy()
	print("Post Training Keras Reward Prediction:")
	print(f"\tstate {inp[0][0]} => reward: {out[0][0]} ")
	print(f"\tstate {inp[1][0]} => reward: {out[1][0]} ")

	trained_dnn_model_torch = train_DNN_torch(dnn_model_torch, memory_buffer, epoch=2000)
	out = trained_dnn_model_torch(torch.from_numpy(inp).type(torch.float)).detach().numpy()
	print("Post Training PyTorch Reward Prediction:")
	print(f"\tstate {inp[0][0]} => reward: {out[0][0]} ")
	print(f"\tstate {inp[1][0]} => reward: {out[1][0]} ")


if __name__ == "__main__":
	main()	
