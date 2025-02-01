import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the data
x = torch.tensor([[3.0, 8.0], [4.0, 5.0], [5.0, 7.0], [6.0, 3.0], [2.0, 1.0]])
y = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7])

# Define the model, loss function, and optimizer
model = nn.Linear(2, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training the model
Losses = []
epochs = 100

for i in range(epochs):
    optimizer.zero_grad()  # Zero the gradients from the previous step
    y_pred = model(x)  # Forward pass
    loss = criterion(y_pred, y)  # Compute loss
    Losses.append(loss.item())  # Store the loss for plotting
    loss.backward()  # Backward pass to compute gradients
    optimizer.step()  # Update model parameters using the optimizer
    print(f"epoch = {i} -----> Loss = {loss.item()}")

# Plot the loss curve
plt.plot(Losses)
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.show()

# Now, to plot the data and regression line/plane:
# First, prepare the data for the 3D plot
x_data = x.numpy()  # Convert tensor to numpy array for plotting
y_data = y.numpy()

# Plot the data points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the actual data points
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, color='blue', label='Data points')

# Generate values for the plane
x1_vals = np.linspace(x_data[:, 0].min(), x_data[:, 0].max(), 10)
x2_vals = np.linspace(x_data[:, 1].min(), x_data[:, 1].max(), 10)
X1, X2 = np.meshgrid(x1_vals, x2_vals)  # Create a grid for X1 and X2
X1_flat = X1.flatten()
X2_flat = X2.flatten()

# Use the model to predict y values based on X1 and X2
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient tracking for inference
    inputs = torch.tensor(np.column_stack([X1_flat, X2_flat]), dtype=torch.float32)
    y_pred = model(inputs).numpy()  # Predicted y values

# Reshape y_pred back to the meshgrid shape
Y_pred = y_pred.reshape(X1.shape)

# Plot the regression plane
ax.plot_surface(X1, X2, Y_pred, color='red', alpha=0.5, label='Regression Plane')

# Labels and legend
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.legend()

plt.show()
