import cv2
import numpy as np
from pathlib import Path
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load images
img_dir = Path("Images")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

read_array = []

for path in sorted(img_dir.iterdir()):
    if path.is_file() and path.suffix.lower() in image_extensions:
        read_array.append(str(path))

img = []
sampleID = []
direction = []
for i in range(len(read_array)):
    img.append(cv2.imread(read_array[i]))
    sampleID.append(os.path.basename(read_array[i])[1:5])
    direction.append(os.path.basename(read_array[i])[5])
classes = sorted(set(sampleID))
class_dict = {label: i for i, label in enumerate(classes)}
# targets
Y = [class_dict[i] for i in sampleID]

# Creates new images with small variations
def image_gen(image):
    h, w = image.shape[:2]
    rand_angle = random.uniform(-10,10)
    M_rot = cv2.getRotationMatrix2D((w/2, h/2),rand_angle,1.0)
    M_rot[0,2] += random.randint(-5,5)
    M_rot[1,2] += random.randint(-5,5)
    gen_image = cv2.warpAffine(image, M_rot,(w, h),cv2.BORDER_CONSTANT,0)
    return gen_image

# Generate augmented dataset
new_images = 20
for i in range(len(img)):
    for j in range(new_images):
        new_image = image_gen(img[i])
        img.append(new_image)
        Y = np.append(Y,Y[i])

# split and prep datasets
img = np.array(img)
img = np.mean(img,axis=3)
mean = img.mean()
std = img.std()
img = (img-mean)/std
# 70/30 training/test split
np.random.seed(1)
idx = np.random.permutation(len(img))
train_size = int(len(img)*0.7)
train_idx = idx[:train_size]
test_idx = idx[train_size:]
# rearrange data to fit pytorch format
X_train = torch.from_numpy(img[train_idx]).float().unsqueeze(1)
X_test = torch.from_numpy(img[test_idx]).float().unsqueeze(1)
Y_train = torch.from_numpy(Y[train_idx]).long()
Y_test = torch.from_numpy(Y[test_idx]).long()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # out_features is the number of classes
        self.fc = nn.Linear(64, 10)
    # forward propagation
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

cnn = CNN()
lossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.002)
epochs = 100
batch_size = 8

mode = input("Train model (0) or Predict (1)")
if mode == '0':
    for i in range(epochs):
        for j in range(0, len(X_train), batch_size):
        # train in batches
            inputs = X_train[j:j + batch_size]
            labels = Y_train[j:j + batch_size]
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()
        if i % 10 == 9:
            print(i, loss.item())
    # save weights
    torch.save(cnn.state_dict(), "CNN_weights.pth")
    print("Finished Training")
    y_pred = torch.argmax(cnn(X_test), dim=1)
    errors = 0
    for i in range(len(y_pred)):
        if y_pred[i] != Y_test[i]:
            errors += 1
    accuracy = 1 - errors / len(y_pred)
    print(y_pred)
    print(Y_test)
    print("Accuracy: ", accuracy)
elif mode == '1':
    cnn.load_state_dict(torch.load("CNN_weights.pth"))
    cnn.eval()
    img_dir = Path("Test_image")
    test_img = cv2.imread(next(img_dir.glob("*.jpg")))
    test_img = np.array(test_img)
    test_img = np.mean(test_img, axis=2)
    test_img = (test_img - mean) / std
    test_img = torch.from_numpy(test_img).float().unsqueeze(0).unsqueeze(0)
    y_pred = torch.argmax(cnn(test_img), dim=1)
    prob = torch.softmax(cnn(test_img), dim=1).detach().cpu().numpy()
    inv_dict = {value: key for key, value in class_dict.items()}
    print("Sample ID: ", inv_dict[int(y_pred)])
    print("Confidence: ", np.max(prob))
else:
    print("Invalid Input")