import torch  # Import the PyTorch library for tensor operations and neural network training
import torch.optim as optim  # Import optimization algorithms, specifically Adam in this case
import json  # Import the json module to read JSON data from a file
from ai_model.model import EnemyAIModel  # Import the custom enemy AI model from the ai_model package

# Load collision data from a JSON file
with open("collision_data.json", "r") as f:
    data = json.load(f)  # Load the JSON data into the 'data' variable

# Prepare training data
# This assumes the data has 'player_position', 'enemy_position', and 'distance'
inputs = []  # Initialize an empty list to store input features
outputs = []  # Initialize an empty list to store the expected outputs

# Iterate over each entry in the loaded data
for entry in data:
    inputs.append(entry['player_position'] + entry['enemy_position'])  # Concatenate player and enemy positions as input features
    outputs.append(entry['distance'])  # Use distance between player and enemy as the output label

# Convert input and output lists to PyTorch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)  # Convert input features to a tensor of type float32
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)  # Convert output labels to a tensor of type float32

# Define model, loss function, and optimizer
input_size = len(inputs[0])  # Determine the size of the input layer based on the length of the first input
hidden_size = 64  # Define the number of hidden units in the hidden layer
output_size = 1  # Define the size of the output layer, which is 1 for predicting distance

model = EnemyAIModel(input_size, hidden_size, output_size)  # Instantiate the EnemyAIModel with the specified sizes
criterion = torch.nn.MSELoss()  # Mean Squared Error loss function for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001

# Training loop
epochs = 1000  # Define the number of training epochs
for epoch in range(epochs):  # Loop through each epoch
    optimizer.zero_grad()  # Zero out the gradients from the previous iteration
    output = model(inputs_tensor)  # Perform a forward pass through the model with the input data
    loss = criterion(output, outputs_tensor)  # Compute the loss between the predicted output and the actual output
    loss.backward()  # Backpropagate the loss to compute gradients
    optimizer.step()  # Update model parameters using the computed gradients

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')  # Print the epoch number and current loss value