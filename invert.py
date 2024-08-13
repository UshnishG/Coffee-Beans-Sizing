import cv2
import numpy as np

# Load the image
image = cv2.imread('test8.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Invert the grayscale image
inverted = cv2.bitwise_not(gray)

# Apply binary thresholding
_, thresholded = cv2.threshold(inverted, 130, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a black background of the same size as the image
black_background = np.zeros_like(image)

# Fill contours with white
cv2.drawContours(black_background, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Draw contours in green for marking
cv2.drawContours(black_background, contours, -1, (0, 255, 0), thickness=2)  # Green color for contour edges

# Save the result with contours marked
cv2.imwrite('result_with_contours8.png', black_background)

