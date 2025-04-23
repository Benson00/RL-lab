import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import pandas as pd
import numpy as np


def createDNN_keras(nInputs, nOutputs, nLayer, nNodes):
	"""
	Function that generates a neural network with the given requirements.

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
	model.add(Dense(nNodes, activation='relu', input_shape=(nInputs,)))
	for _ in range(nLayer - 1): 
		model.add(Dense(nNodes, activation = 'relu'))
	model.add(Dense(nOutputs, activation='linear'))
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
		layers = []
		layers.append(nn.Linear(nInputs, nNodes))
		layers.append(nn.ReLU())

		# add the hidden states
		for _ in range(nLayer-1):
			layers.append(nn.Linear(nNodes,nNodes))
			layers.append(nn.ReLU())
		
		layers.append(nn.Linear(nNodes, nOutputs))
		self.model = nn.Sequential(*layers)


	def forward(self, x):
		return self.model(x)


def mse(network, dataset_input, target):
	"""
	Compute the MSE loss function

	"""
	# Compute the predicted value, over time this value should
	# looks more like to the expected output (i.e., target)
	predicted_value = network(dataset_input)
	
	# Compute MSE between the predicted value and the expected labels
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)
	
	# Return the averaged values for computational optimization
	return mse


def training_loop(env, neural_net, updateRule, keras=True, eps=1.0, updates=1, episodes=100):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	#TODO: initialize the optimizer 
	if keras:
		optimizer = optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
	else:
		optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-3)

	 
	rewards_list, memory_buffer = [], collections.deque( maxlen=1000 )
	averaged_rewards = []
	for ep in range(episodes):

		#TODO: reset the environment and obtain the initial state
		
		state = env.reset()[0]
		ep_reward = 0
		while True:
	
			#TODO: select the action to perform exploiting an epsilon-greedy strategy
			if np.random.normal() < eps: 
				action = env.action_space.sample()
			else:
				# take q_values and then take the max for the action
				q_values = neural_net.predict(np.array([state])) if keras else neural_net(torch.tensor(state, dtype=torch.float32))
				action = int(torch.argmax(q_values).item()) if not keras else np.argmax(q_values)

			#TODO: update epsilon value
			eps *= 0.99

			#TODO: Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			memory_buffer.append([state,action,reward,next_state,done])
			ep_reward += reward

			# Perform the actual training
			for _ in range(updates):
				#TODO: call the update rule...
				DQNupdate(neural_net, keras, memory_buffer, optimizer)
				

			#TODO: modify the exit condition for the episode
			if done:
				break

			#TODO: update the current state
			state = next_state

		# Update the reward list to return
		rewards_list.append(ep_reward)
		averaged_rewards.append(np.mean(rewards_list))
		print( f"episode {ep:2d}: mean reward: {averaged_rewards[-1]:3.2f}, eps: {eps:3.2f}" )

	# Close the enviornment and return the rewards list
	env.close()
	return averaged_rewards


def DQNupdate(neural_net, keras, memory_buffer, optimizer, batch_size=32, gamma=0.99):
    """
    Main update rule for the DQN process. Extract data from the memory buffer and update 
    the network computing the gradient.
    """

    # If we don't have enough samples in memory_buffer, exit
    if len(memory_buffer) < batch_size: 
        return 

    # Sample random indices from the memory buffer
    indices = np.random.randint(len(memory_buffer), size=batch_size)

    for idx in indices: 
        # Extract data from the buffer
        state, action, reward, next_state, done = memory_buffer[idx]

        if keras:
            # Convert to tensors
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)

            # Predict Q-values for current and next states
            q_values = neural_net(state_tensor).numpy()  # Convert tensor to numpy for modification
            next_q_values = neural_net(next_state_tensor).numpy()

            # Compute target Q-value
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(next_q_values)

            # Update the Q-value for the selected action
            q_values[0, action] = target

            # Convert updated Q-values back to tensor
            target_tensor = tf.convert_to_tensor(q_values, dtype=tf.float32)

            # Compute loss and backpropagation
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(neural_net(state_tensor) - target_tensor))

            gradients = tape.gradient(loss, neural_net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables))

        else:  # PyTorch implementation
            # Convert to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.long)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            done_tensor = torch.tensor(done, dtype=torch.float32)

            # Predict Q-values
            q_values = neural_net(state_tensor)
            next_q_values = neural_net(next_state_tensor).detach()  # No gradient tracking

            # Compute target Q-value
            if done:
                target = reward_tensor
            else:
                target = reward_tensor + gamma * torch.max(next_q_values)

            # Compute loss and perform backpropagation
            loss = nn.functional.mse_loss(q_values[0, action_tensor], target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



def main():
	print( "\n************************************************" )
	print( "*  Welcome to the seventh lesson of the RL-Lab!   *" )
	print( "*               (Deep Q-Network)                 *" )
	print( "**************************************************\n" )

	training_steps = 50
	
	# setting DNN configuration
	nInputs=4
	nOutputs=2
	nLayer=2
	nNodes=32 

	print("\nTraining torch model...\n")
	rewards_torch = []
	for _ in range(10):
		env = gymnasium.make("CartPole-v1", render_mode="human")
		neural_net_torch = TorchModel(nInputs, nOutputs, nLayer, nNodes)
		rewards_torch.append(training_loop(env, neural_net_torch, DQNupdate, keras=False, episodes=training_steps))

	print("\nTraining keras model...\n")
	rewards_keras = []
	for _ in range(10):
		env = gymnasium.make("CartPole-v1")#, render_mode="human" )
		neural_net_keras = createDNN_keras(nInputs, nOutputs, nLayer, nNodes)
		rewards_keras.append(training_loop(env, neural_net_keras, DQNupdate, keras=True, episodes=training_steps))


	# plotting the results
	t = list(range(0, training_steps))

	data = {'Environment Step': [], 'Mean Reward': []}
	for _, rewards in enumerate(rewards_torch):
		for step, reward in zip(t, rewards):
			data['Environment Step'].append(step)
			data['Mean Reward'].append(reward)
	df_torch = pd.DataFrame(data)

	data_keras = {'Environment Step': [], 'Mean Reward': []}
	for _, rewards in enumerate(rewards_keras):
		for step, reward in zip(t, rewards):
			data_keras['Environment Step'].append(step)
			data_keras['Mean Reward'].append(reward)
	df_keras = pd.DataFrame(data_keras)

	# Plotting
	sns.set_style("darkgrid")
	plt.figure(figsize=(8, 6))  # Set the figure size
	sns.lineplot(data=df_torch, x='Environment Step', y='Mean Reward', label='torch', errorbar='se')
	sns.lineplot(data=df_keras, x='Environment Step', y='Mean Reward', label='keras', errorbar='se')

	# Add title and labels
	plt.title('Comparison Keras and PyTorch on CartPole-v1')
	plt.xlabel('Episodes')
	plt.ylabel('Mean Reward')

	# Show legend
	plt.legend()

	# Show plot
	plt.show()
	plt.savefig('comparison.pdf')


if __name__ == "__main__":
	main()	

