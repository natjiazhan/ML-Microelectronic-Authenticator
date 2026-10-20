import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load images
raw_dir = Path("Query_raw")
processed_dir = Path("Query_processed")
processed_dir.mkdir(exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

read_array = []
write_array = []

for path in sorted(raw_dir.iterdir()):
    if path.is_file() and path.suffix.lower() in image_extensions:
        read_array.append(str(path))
        write_array.append(str(processed_dir / f"{path.stem}{path.suffix}"))

img = []
for i in range(len(read_array)):
    img.append(cv2.imread(read_array[i]))

def combine(image1,image2,image3,image4,threshold = 20):
    image1 = process(image1,threshold)
    image2 = process(image2,threshold)
    image3 = process(image3,threshold)
    image4 = process(image4,threshold)
    a1 = cv2.add(image1,image2)
    a2 = cv2.add(image3,image4)
    combined = cv2.add(a1,a2)
    #combined = cv2.threshold(combined, 1, 255, cv2.THRESH_BINARY)[1]
    return combined

def process(image,cutoff):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,filtered_img = cv2.threshold(img_gray, cutoff, 255, cv2.THRESH_TOZERO)
    return filtered_img

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def get_center(image):
    moments = cv2.moments(image)
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    return center_x, center_y

def crop(image,x,y,crop_size,new_size):
    s = (crop_size, crop_size)
    # Fixed size crop
    center = (x,y)
    cropped_img = cv2.getRectSubPix(image, s, center)
    processed_img = cv2.resize(cropped_img, (new_size,new_size), interpolation = cv2.INTER_AREA)
    return processed_img

def old_process(image,cutoff,crop_size,new_size,apply_threshold_filter=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _,filtered_img = cv2.threshold(blur, cutoff, 255, cv2.THRESH_TOZERO)
    _, filtered_img_bg = cv2.threshold(img_gray, 20, 255, cv2.THRESH_TOZERO)
    x, y, w, h = cv2.boundingRect(filtered_img)
    center = ((2*x+w)/2, (2*y+h)/2)
    s = (crop_size, crop_size)
    # Fixed size crop
    if apply_threshold_filter:
        cropped_img = cv2.getRectSubPix(filtered_img_bg, s, center)
    else:
        cropped_img = cv2.getRectSubPix(img_gray, s, center)
    processed_img = cv2.resize(cropped_img, (new_size,new_size), interpolation = cv2.INTER_AREA)
    return processed_img

#outputs 224x224 image
final_size = 224
#crop size
size = 240
img_processed = np.zeros((len(img),final_size,final_size),dtype=np.uint8)
if len(img) > 4:
    for i in range(len(img)//4):
        combined_img = combine(img[i*4],img[i*4+1],img[i*4+2],img[i*4+3],40)
        x,y = get_center(combined_img)
        #show_image(combined_img)
        for j in range(0,4):
            img_filtered = process(img[i*4+j],20)
            img_processed[i*4+j] = crop(img_filtered,x,y,size,final_size)
            #show_image(img_processed[i*4+j])
else:
    for i in range(len(img)):
        img_processed[i] = old_process(img[i], 50, size, final_size, True)

for i in range(len(write_array)):
    cv2.imwrite(write_array[i],img_processed[i])
