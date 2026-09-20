
import cv2
import numpy as np
from pathlib import Path
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

img_dir = Path("Images")
test_img_dir = Path("Test_image")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

embedding_size = 128
epochs = 50
batch_size = 16
pairs_per_epoch = 4000
learning_rate = 0.001
margin = 1.0

# Replicate split
train_replicates = {1, 2, 3, 4, 5, 6, 7}
validation_replicates = {8}
test_replicates = {9, 10}

# Number of augmented versions generated from each TRAINING image
new_images = 5

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# ---------------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------------

read_array = []

for path in sorted(img_dir.iterdir()):
    if path.is_file() and path.suffix.lower() in image_extensions:
        read_array.append(str(path))

img = []
sampleID = []
replicate = []
direction = []

filename_pattern = re.compile(r"D(\d+)R(\d+)A(\d+)", re.IGNORECASE)

for i in range(len(read_array)):
    image = cv2.imread(read_array[i])

    if image is None:
        print("Could not read:", read_array[i])
        continue

    name = Path(read_array[i]).stem
    match = filename_pattern.fullmatch(name)

    if not match:
        raise ValueError(
            f"Unexpected filename format: {name}\n"
            "Expected format such as D1R1A1.jpg"
        )

    dot_num, rep_num, angle_num = match.groups()

    img.append(image)
    sampleID.append(int(dot_num))
    replicate.append(int(rep_num))
    direction.append(int(angle_num))

print("Images loaded:", len(img))
print("Dots found:", sorted(set(sampleID)))

# ---------------------------------------------------------
# IMAGE AUGMENTATION
# ---------------------------------------------------------

def image_gen(image):
    h, w = image.shape[:2]

    rand_angle = random.uniform(-10, 10)

    M_rot = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        rand_angle,
        1.0
    )

    M_rot[0, 2] += random.randint(-5, 5)
    M_rot[1, 2] += random.randint(-5, 5)

    gen_image = cv2.warpAffine(
        image,
        M_rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return gen_image

# ---------------------------------------------------------
# SPLIT ORIGINAL IMAGES BY REPLICATE
# ---------------------------------------------------------

train_img = []
train_ids = []

val_img = []
val_ids = []

test_img = []
test_ids = []

for i in range(len(img)):

    if replicate[i] in train_replicates:
        train_img.append(img[i])
        train_ids.append(sampleID[i])

    elif replicate[i] in validation_replicates:
        val_img.append(img[i])
        val_ids.append(sampleID[i])

    elif replicate[i] in test_replicates:
        test_img.append(img[i])
        test_ids.append(sampleID[i])

print("Original training images:", len(train_img))
print("Validation images:", len(val_img))
print("Test images:", len(test_img))

# ---------------------------------------------------------
# AUGMENT TRAINING DATA ONLY
# ---------------------------------------------------------

augmented_train_img = list(train_img)
augmented_train_ids = list(train_ids)

for i in range(len(train_img)):
    for j in range(new_images):
        new_image = image_gen(train_img[i])
        augmented_train_img.append(new_image)
        augmented_train_ids.append(train_ids[i])

train_img = augmented_train_img
train_ids = augmented_train_ids

print("Training images after augmentation:", len(train_img))

# ---------------------------------------------------------
# CONVERT TO GRAYSCALE
# ---------------------------------------------------------

def convert_to_gray(image_list):
    gray_array = []

    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_array.append(gray)

    return np.array(gray_array, dtype=np.float32)

train_img = convert_to_gray(train_img)
val_img = convert_to_gray(val_img)
test_img = convert_to_gray(test_img)

# ---------------------------------------------------------
# NORMALIZE USING TRAINING DATA ONLY
# ---------------------------------------------------------

mean = train_img.mean()
std = train_img.std()

if std == 0:
    raise ValueError("Training image standard deviation is zero.")

train_img = (train_img - mean) / std
val_img = (val_img - mean) / std
test_img = (test_img - mean) / std

print("Training mean:", mean)
print("Training std:", std)

# ---------------------------------------------------------
# SIAMESE PAIR DATASET
# ---------------------------------------------------------

class SiameseDataset(Dataset):

    def __init__(self, images, labels, pairs_per_epoch):

        self.images = images
        self.labels = np.array(labels)
        self.pairs_per_epoch = pairs_per_epoch

        self.classes = sorted(set(labels))

        self.class_indices = {}

        for label in self.classes:
            self.class_indices[label] = np.where(
                self.labels == label
            )[0]

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, index):

        same_class = random.random() < 0.5

        if same_class:

            selected_class = random.choice(self.classes)

            indices = self.class_indices[selected_class]

            if len(indices) < 2:
                raise ValueError(
                    f"Dot {selected_class} does not have enough images "
                    "to make a positive pair."
                )

            idx1, idx2 = random.sample(
                list(indices),
                2
            )

            label = 1.0

        else:

            class1, class2 = random.sample(
                self.classes,
                2
            )

            idx1 = random.choice(
                self.class_indices[class1]
            )

            idx2 = random.choice(
                self.class_indices[class2]
            )

            label = 0.0

        image1 = torch.from_numpy(
            self.images[idx1]
        ).float().unsqueeze(0)

        image2 = torch.from_numpy(
            self.images[idx2]
        ).float().unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image1, image2, label

# ---------------------------------------------------------
# CNN EMBEDDING NETWORK
# ---------------------------------------------------------

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Siamese output: embedding instead of class scores
        self.fc = nn.Linear(64, embedding_size)

    # forward propagation
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Normalize the embedding so cosine similarity is meaningful
        x = F.normalize(x, p=2, dim=1)

        return x

# ---------------------------------------------------------
# CONTRASTIVE LOSS
# ---------------------------------------------------------

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        distance = F.pairwise_distance(output1, output2)

        positive_loss = label * distance.pow(2)

        negative_loss = (
            (1 - label)
            * torch.clamp(
                self.margin - distance,
                min=0
            ).pow(2)
        )

        return (positive_loss + negative_loss).mean()

# ---------------------------------------------------------
# MODEL SETUP
# ---------------------------------------------------------

cnn = CNN()
lossFunc = ContrastiveLoss(margin=margin)
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

# ---------------------------------------------------------
# HELPER: EMBEDDING
# ---------------------------------------------------------

def get_embedding(model, image):
    model.eval()

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image)

    return embedding

# ---------------------------------------------------------
# HELPER: COSINE SIMILARITY
# ---------------------------------------------------------

def compare_embeddings(embedding1, embedding2):
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

# ---------------------------------------------------------
# VALIDATION / TEST PAIR EVALUATION
# ---------------------------------------------------------

def evaluate_pairs(model, images, labels, num_pairs=1000):

    dataset = SiameseDataset(
        images,
        labels,
        num_pairs
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    same_scores = []
    different_scores = []

    model.eval()

    with torch.no_grad():

        for image1, image2, label in loader:

            output1 = model(image1)
            output2 = model(image2)

            similarities = F.cosine_similarity(
                output1,
                output2
            )

            for score, pair_label in zip(
                similarities.cpu().numpy(),
                label.cpu().numpy()
            ):

                if pair_label == 1:
                    same_scores.append(float(score))
                else:
                    different_scores.append(float(score))

    return same_scores, different_scores

# ---------------------------------------------------------
# FIND BEST VALIDATION THRESHOLD
# ---------------------------------------------------------

def find_best_threshold(same_scores, different_scores):

    all_scores = same_scores + different_scores

    minimum = min(all_scores)
    maximum = max(all_scores)

    thresholds = np.linspace(
        minimum,
        maximum,
        500
    )

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:

        correct = 0
        total = 0

        for score in same_scores:
            if score >= threshold:
                correct += 1
            total += 1

        for score in different_scores:
            if score < threshold:
                correct += 1
            total += 1

        accuracy = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy

# ---------------------------------------------------------
# TRAIN OR PREDICT
# ---------------------------------------------------------

mode = input("Train model (0) or Predict (1): ")

# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

if mode == '0':

    train_dataset = SiameseDataset(
        train_img,
        train_ids,
        pairs_per_epoch
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for i in range(epochs):

        cnn.train()
        running_loss = 0.0

        for image1, image2, labels in train_loader:

            optimizer.zero_grad()

            output1 = cnn(image1)
            output2 = cnn(image2)

            loss = lossFunc(
                output1,
                output2,
                labels
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        if i % 5 == 4:
            print("Epoch:", i + 1, "Loss:", avg_loss)

    # -----------------------------------------------------
    # VALIDATION THRESHOLD
    # -----------------------------------------------------

    same_scores, different_scores = evaluate_pairs(
        cnn,
        val_img,
        val_ids,
        num_pairs=2000
    )

    threshold, validation_accuracy = find_best_threshold(
        same_scores,
        different_scores
    )

    print("Validation threshold:", threshold)
    print("Validation pair accuracy:", validation_accuracy)

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    same_test_scores, different_test_scores = evaluate_pairs(
        cnn,
        test_img,
        test_ids,
        num_pairs=2000
    )

    correct = 0
    total = 0

    for score in same_test_scores:
        if score >= threshold:
            correct += 1
        total += 1

    for score in different_test_scores:
        if score < threshold:
            correct += 1
        total += 1

    test_accuracy = correct / total

    print("Final test pair accuracy:", test_accuracy)
    print("Average same-dot similarity:", np.mean(same_test_scores))
    print("Average different-dot similarity:", np.mean(different_test_scores))

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    torch.save(
        {
            "model_state": cnn.state_dict(),
            "mean": float(mean),
            "std": float(std),
            "threshold": float(threshold),
            "embedding_size": embedding_size
        },
        "Siamese_CNN_model.pth"
    )

    print("Finished Training")
    print("Saved as Siamese_CNN_model.pth")

# ---------------------------------------------------------
# PREDICT / COMPARE TWO IMAGES
# ---------------------------------------------------------

elif mode == '1':

    checkpoint = torch.load(
        "Siamese_CNN_model.pth",
        map_location="cpu"
    )

    cnn.load_state_dict(checkpoint["model_state"])
    cnn.eval()

    saved_mean = checkpoint["mean"]
    saved_std = checkpoint["std"]
    threshold = checkpoint["threshold"]

    test_paths = []

    for path in sorted(test_img_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in image_extensions:
            test_paths.append(path)

    if len(test_paths) != 2:
        raise ValueError(
            "Put exactly TWO images in the Test_image folder.\n"
            "The program will compare those two images."
        )

    processed_images = []

    for path in test_paths:

        test_image = cv2.imread(str(path))

        if test_image is None:
            raise ValueError(f"Could not read {path}")

        test_image = cv2.cvtColor(
            test_image,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        test_image = (
            test_image - saved_mean
        ) / saved_std

        processed_images.append(test_image)

    embedding1 = get_embedding(cnn, processed_images[0])
    embedding2 = get_embedding(cnn, processed_images[1])

    similarity = compare_embeddings(
        embedding1,
        embedding2
    )

    print("Image 1:", test_paths[0].name)
    print("Image 2:", test_paths[1].name)
    print("Cosine similarity:", similarity)
    print("Decision threshold:", threshold)

    if similarity >= threshold:
        print("Prediction: SAME DOT")
    else:
        print("Prediction: DIFFERENT DOT")

else:
    print("Invalid Input")
