import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
# Make sure image files are in the same directory as your script
img = np.array([
cv2.imread("N0001E1.jpg"),
cv2.imread("N0001E2.jpg"),
cv2.imread("N0001E3.jpg"),
cv2.imread("N0001N1.jpg"),
cv2.imread("N0001N2.jpg"),
cv2.imread("N0001N3.jpg"),
cv2.imread("N0001S1.jpg"),
cv2.imread("N0001S2.jpg"),
cv2.imread("N0001S3.jpg"),
cv2.imread("N0001W1.jpg"),
cv2.imread("N0001W2.jpg"),
cv2.imread("N0001W3.jpg"),
])

# Mike's Images
img_W = np.array([
cv2.imread("W0001.jpg"),
cv2.imread("W0002.jpg"),
cv2.imread("W0003.jpg"),
])

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Threshold filter and crop
def process(image,cutoff,crop_size,new_size,apply_threshold_filter=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,filtered_img = cv2.threshold(img_gray, cutoff, 255, cv2.THRESH_TOZERO)
    x, y, w, h = cv2.boundingRect(filtered_img)
    center = ((2*x+w)/2, (2*y+h)/2)
    s = (crop_size, crop_size)
    # Fixed size crop
    if apply_threshold_filter:
        cropped_img = cv2.getRectSubPix(filtered_img, s, center)
    else:
        cropped_img = cv2.getRectSubPix(img_gray, s, center)
    processed_img = cv2.resize(cropped_img, (new_size,new_size), interpolation = cv2.INTER_AREA)
    return processed_img

#outputs 224x224 image
final_size = 224

#240x240 crop
size = 240
img_processed = np.zeros((len(img),final_size,final_size),dtype=np.uint8)
for i in range(len(img)):
    # Cutoff may need to be higher depending on background brightness
    img_processed[i] = process(img[i],50,size,final_size)
    #show_image(img_processed[i])

#960x960 crop
size = 960
img_processed_W = np.zeros((len(img_W),final_size,final_size),dtype=np.uint8)
for i in range(len(img_W)):
    img_processed_W[i] = process(img_W[i], 70, size,final_size)
    #show_image(img_processed_W[i])

# Save the processed image
#cv2.imwrite("N0001E1p.jpg",img_processed[0])
#cv2.imwrite("N0001E2p.jpg",img_processed[1])
#cv2.imwrite("N0001E3p.jpg",img_processed[2])
#cv2.imwrite("N0001N1p.jpg",img_processed[3])
#cv2.imwrite("N0001N2p.jpg",img_processed[4])
#cv2.imwrite("N0001N3p.jpg",img_processed[5])
#cv2.imwrite("N0001S1p.jpg",img_processed[6])
#cv2.imwrite("N0001S2p.jpg",img_processed[7])
#cv2.imwrite("N0001S3p.jpg",img_processed[8])
#cv2.imwrite("N0001W1p.jpg",img_processed[9])
#cv2.imwrite("N0001W2p.jpg",img_processed[10])
#cv2.imwrite("N0001W3p.jpg",img_processed[11])
#cv2.imwrite("W0001p.jpg",img_processed_W[0])
#cv2.imwrite("W0002p.jpg",img_processed_W[1])
#cv2.imwrite("W0003p.jpg",img_processed_W[2])
