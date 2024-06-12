import cv2

# Step 1: Image acquisition
image = cv2.imread('document.jpg')

# Step 2: Image cropping and rotation (if required)
# Specify the coordinates of the ROI to crop (x, y, width, height)
roi = (100, 100, 800, 600)
cropped_image = image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

# Step 3: Image resizing and normalization
resized_image = cv2.resize(cropped_image, (800, 600))

# Step 4: Noise reduction
denoised_image = cv2.medianBlur(resized_image, 5)

# Step 5: Image enhancement
enhanced_image = cv2.equalizeHist(denoised_image)

# Step 6: Binarization
gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Display the preprocessed image
cv2.imshow('Preprocessed Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
