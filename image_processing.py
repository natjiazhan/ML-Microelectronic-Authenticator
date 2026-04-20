import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load images
raw_dir = Path("Raw")
processed_dir = Path("Processed")
processed_dir.mkdir(exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

read_array = []
write_array = []

for path in sorted(raw_dir.iterdir()):
    if path.is_file() and path.suffix.lower() in image_extensions:
        read_array.append(str(path))
        write_array.append(str(processed_dir / f"{path.stem}P{path.suffix}"))

img = []
for i in range(len(read_array)):
    img.append(cv2.imread(read_array[i]))

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

img_processed = np.zeros((len(img),final_size,final_size),dtype=np.uint8)
for i in range(len(img)):
    file_name = Path(read_array[i]).name

    if file_name.startswith("N"):
        size = 240
    elif file_name.startswith("W"):
        size = 960
    else:
        size = 240

    # Cutoff may need to be higher depending on background brightness
    img_processed[i] = process(img[i],80,size,final_size)
    #show_image(img_processed[i])

# Save the processed image
for i in range(len(write_array)):
    cv2.imwrite(write_array[i],img_processed[i])