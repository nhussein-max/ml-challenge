import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define the model architecture
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(133, 133)
        self.linear2 = torch.nn.Linear(133, 133)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Initialize the model and load the saved weights
model = Model()
model.load_state_dict(torch.load("model.pth"))

# Get the weights and biases of the layer
l1_weights = model.linear1.weight
l1_weights_inv = torch.linalg.inv(l1_weights)
l1_bias = model.linear1.bias

l2_weights = model.linear2.weight
l2_weights_inv = torch.linalg.inv(l2_weights)
l2_bias = model.linear2.bias


# Read scrambled flag
flag = cv2.imread('out.png', 0)
image_tensor = torch.from_numpy(flag).to(torch.float32)

# Reverse Path
model_input = image_tensor
layer1_undo = torch.matmul((model_input-l2_bias), l2_weights_inv.T)
original_output = torch.matmul((layer1_undo-l1_bias), l1_weights_inv.T)


# Convert tensors back to numpy arrays for visualization
original_image_np = image_tensor.numpy()
reverse_image_np = original_output.detach().numpy()

tensor_min = original_output.min()
tensor_max = original_output.max()
# Map values to the range 0-255
mapped_tensor = ((original_output - tensor_min) / (tensor_max - tensor_min)) * 255
mapped_np = mapped_tensor.detach().numpy()

rows = 2
# Plot images side by side using matplotlib
plt.figure(figsize=(15, 5))
plt.subplot(rows, 1, 1)
plt.title('Original Image Tensor')
plt.imshow(original_image_np, cmap='gray')
plt.axis('off')

plt.subplot(rows, 1, 2)
plt.title('After Reverse Path')
plt.imshow(reverse_image_np, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
