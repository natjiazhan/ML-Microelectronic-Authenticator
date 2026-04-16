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
def process(image,cutoff,img_size):
    _,filtered_img = cv2.threshold(image, cutoff, 255, cv2.THRESH_TOZERO)
    x, y, w, h = cv2.boundingRect(filtered_img)
    center_x = (2*x+w)//2
    center_y = (2*y+h)//2
    s = img_size//2
    if center_y-s < 0:
        center_y -= center_y-s
    if center_y+s > image.shape[1]:
        center_y -= center_y+s-image.shape[1]
    # Fixed size crop
    cropped_img = filtered_img[center_y-s:center_y+s,center_x-s:center_x+s]
    return cropped_img

#240x240 image
size = 240
img_gray = np.zeros((len(img),480,640),dtype=np.uint8)
img_processed = np.zeros((len(img),size,size),dtype=np.uint8)
for i in range(len(img)):
    img_gray[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
    # Cutoff may need to be higher depending on background brightness
    img_processed[i] = process(img_gray[i],50,size)
    show_image(img_processed[i])
size = 960
img_gray_W = np.zeros((len(img_W),1080,1920),dtype=np.uint8)
img_processed_W = np.zeros((len(img_W),size,size),dtype=np.uint8)
for i in range(len(img_W)):
    img_gray_W[i] = cv2.cvtColor(img_W[i], cv2.COLOR_BGR2GRAY)
    img_processed_W[i] = process(img_gray_W[i], 70, size)
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