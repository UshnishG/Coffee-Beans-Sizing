import cv2
import numpy as np

# Conversion factor
PX_TO_MM = 14

# Load the image with contours
image_with_contours = cv2.imread('result_with_contours7.png')

# Convert to grayscale
gray = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thresholded = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through each contour
for contour in contours:
    # Get the convex hull of the contour
    hull = cv2.convexHull(contour)
    
    # Find the minimum enclosing rectangle for the convex hull
    rect = cv2.minAreaRect(hull)
    (cx, cy), (width, height), _ = rect
    
    # Calculate the shortest diagonal
    shortest_diagonal_px = min(width, height)
    shortest_diagonal_mm = shortest_diagonal_px / PX_TO_MM
    
    # Annotate the shortest diagonal (as width) on the image
    text_diag = f'Width: {shortest_diagonal_mm:.2f} mm'
    
    # Calculate the position for the text
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        text_position = (cX - 50, cY - 10)
    else:
        text_position = (int(cx - width / 2), int(cy - height / 2) - 10)
    
    # Annotate the diagonal on the image using contours
    cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)
    cv2.putText(image_with_contours, text_diag, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# Save the result with annotated shortest diagonals and contours
cv2.imwrite('result_with_annotations7.png', image_with_contours)
