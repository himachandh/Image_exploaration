import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage import color

# Load the image information into a numpy nd-array
image = imread('scenery.png')

# Print the number of pixels in the given image
height, width, _ = image.shape
num_pixels = height * width
print("Number of pixels:", num_pixels)

# Separate the image into red, green, and blue channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Create a new image with only the red channel
red_image = np.zeros_like(image)
red_image[:, :, 0] = image[:, :, 0]

# Create a new image with only the green channel
green_image = np.zeros_like(image)
green_image[:, :, 1] = image[:, :, 1]

# Create a new image with only the blue channel
blue_image = np.zeros_like(image)
blue_image[:, :, 2] = image[:, :, 2]

# Display the red, green, and blue channel images and find the maximum intensity values and their indices in each channel
red_max_value = np.max(red_channel)
red_max_index = np.unravel_index(np.argmax(red_channel), red_channel.shape)

print("Red: Maximum intensity value =", red_max_value, "at index", red_max_index)

fig, axes = plt.subplots(1, 1, figsize=(4, 4))
axes.imshow(red_image)
axes.set_title('Red Channel')
axes.axis('off')
plt.tight_layout()
plt.show()

green_max_value = np.max(green_channel)
green_max_index = np.unravel_index(np.argmax(green_channel), green_channel.shape)

print("Green: Maximum intensity value =", green_max_value, "at index", green_max_index)

fig, axes = plt.subplots(1, 1, figsize=(4, 4))
axes.imshow(green_image)
axes.set_title('Green Channel')
axes.axis('off')
plt.tight_layout()
plt.show()

blue_max_value = np.max(blue_channel)
blue_max_index = np.unravel_index(np.argmax(blue_channel), blue_channel.shape)

print("Blue: Maximum intensity value =", blue_max_value, "at index", blue_max_index)

fig, axes = plt.subplots(1, 1, figsize=(4, 4))
axes.imshow(blue_image)
axes.set_title('Blue Channel')
axes.axis('off')
plt.tight_layout()
plt.show()



# Convert the image to grayscale and calculate mean values
grayscale_image = color.rgb2gray(image)
print("Mean of original image:", np.mean(image))
print("Mean of grayscale image:", np.mean(grayscale_image))

# Flip the image horizontally
flipped_horizontal = np.flip(image, axis=0)

# Flip the image vertically
flipped_vertical = np.flip(image, axis=1)

# Plot the Grayscale image
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[1].imshow(grayscale_image, cmap='gray')
axes[1].set_title("Grayscale Image (Mean: {:.2f})".format(np.mean(grayscale_image)))

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

# Plot the horiznntally flipped image
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[1].imshow(flipped_horizontal)
axes[1].set_title("Horizontally Flipped Image")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

# Plot the vertically flipped image
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[1].imshow(flipped_vertical)
axes[1].set_title("Vertically Flipped Image")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

