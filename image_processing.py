import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
# Make sure image files are in the same directory as the script

image_name = ["N0005","N0006"]
# Number of sets of images per sample, for each image name you want to process
number_of_images = [3,3]

read_array = []
write_array = []
for i in range(len(image_name)):
    for j in range(4*number_of_images[i]):
        if j//number_of_images[i] == 0:
            direction = "E"
        elif j//number_of_images[i] == 1:
            direction = "N"
        elif j//number_of_images[i] == 2:
            direction = "S"
        elif j//number_of_images[i] == 3:
            direction = "W"
        read_array.append(image_name[i]+direction+str(j%number_of_images[i]+1)+".jpg")
        write_array.append(image_name[i]+direction+str(j%number_of_images[i]+1)+"P"+".jpg")

img = np.zeros((len(read_array),480,640,3),dtype=np.uint8)
for i in range(len(read_array)):
    img[i] = cv2.imread(read_array[i])

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
    img_processed[i] = process(img[i],80,size,final_size)
    #show_image(img_processed[i])

#960x960 crop
size = 960
img_processed_W = np.zeros((len(img_W),final_size,final_size),dtype=np.uint8)
for i in range(len(img_W)):
    img_processed_W[i] = process(img_W[i], 80, size,final_size)
    #show_image(img_processed_W[i])

# Save the processed image
#for i in range(len(write_array)):
    #img[i] = cv2.imwrite(write_array[i],img_processed[i])
