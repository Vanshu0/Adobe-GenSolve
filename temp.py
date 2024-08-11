import pandas as pd
import numpy as np
import cv2

# Read the CSV file into a pandas DataFrame
# Assume that the CSV file has pixel values with no headers and one pixel value per row
csv_file = 'problems\isolated.csv'
data = pd.read_csv(csv_file, header=None)

# Convert DataFrame to a NumPy array
pixel_values = data.values

# Determine the shape of the image
# For example, if the image is 28x28 pixels (common in MNIST dataset), reshape it accordingly
image_height, image_width = 96, 58
image_array = pixel_values.reshape((image_height, image_width))

# Convert the pixel values to an 8-bit integer type
image_array = image_array.astype(np.uint8)

# Display the image using OpenCV
cv2.imshow('Image', image_array)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()