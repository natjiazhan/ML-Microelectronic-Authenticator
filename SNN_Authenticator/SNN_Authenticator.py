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
from datetime import date

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

img_dir = Path("Training_database")
one_to_one_dir = Path("one_to_one_auth")
test_img_dir = Path("Query_processed")
auth_data_dir = Path("Auth_database")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

select_weight = Path("saved_weights/2026-09-27.pth")

embedding_size = 128
epochs = 60
batch_size = 32
pairs_per_epoch = 8000
learning_rate = 0.003
margin = 1.0
set_threshold = 0.85
use_set_threshold = True

# Replicate split
train_replicates = {1, 2, 3, 4, 5, 6, 7}
validation_replicates = {8}
test_replicates = {9, 10}

# Number of augmented versions generated from each TRAINING image
new_images = 4

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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

    M_rot[0, 2] += random.randint(-8, 8)
    M_rot[1, 2] += random.randint(-8, 8)

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
train_direction = []

val_img = []
val_ids = []
val_direction = []

test_img = []
test_ids = []
test_direction = []

for i in range(len(img)):

    if replicate[i] in train_replicates:
        train_img.append(img[i])
        train_ids.append(sampleID[i])
        train_direction.append(direction[i])

    elif replicate[i] in validation_replicates:
        val_img.append(img[i])
        val_ids.append(sampleID[i])
        val_direction.append(direction[i])

    elif replicate[i] in test_replicates:
        test_img.append(img[i])
        test_ids.append(sampleID[i])
        test_direction.append(direction[i])

print("Original training images:", len(train_img))
print("Validation images:", len(val_img))
print("Test images:", len(test_img))

# ---------------------------------------------------------
# AUGMENT TRAINING DATA ONLY
# ---------------------------------------------------------

augmented_train_img = list(train_img)
augmented_train_ids = list(train_ids)
augmented_train_direction = list(train_direction)

for i in range(len(train_img)):
    for j in range(new_images):
        new_image = image_gen(train_img[i])
        augmented_train_img.append(new_image)
        augmented_train_ids.append(train_ids[i])
        augmented_train_direction.append(train_direction[i])
train_img = augmented_train_img
train_ids = augmented_train_ids
train_direction = augmented_train_direction

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
    def __init__(self, images, sample_ids, directions, pairs_per_epoch):
        self.images = images
        self.sample_ids = np.array(sample_ids)
        self.directions = np.array(directions)
        self.pairs_per_epoch = pairs_per_epoch
        self.classes = sorted(set(zip(self.sample_ids, self.directions)))
        # Store image indices for each (dot, direction) combination
        self.class_indices = {}
        for class_id in self.classes:
            dot_id, direction = class_id
            indices = np.where((self.sample_ids == dot_id) & (self.directions == direction))[0]
            self.class_indices[class_id] = indices

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, index):
        # 50% positive, 50% negative
        same_class = random.random() < 0.5
        if same_class:
            # Choose a (dot, direction) combination
            selected_class = random.choice(self.classes)
            indices = self.class_indices[selected_class]
            if len(indices) < 2:
                raise ValueError(
                    f"Class {selected_class} has fewer than 2 images.")
            # Two images from SAME dot AND SAME direction
            idx1, idx2 = random.sample(list(indices), 2)
            label = 1.0

        else:
            class1 = random.choice(self.classes)
            dot1, direction1 = class1
            possible_classes = [
                class2
                for class2 in self.classes
                if class2[0] != dot1
                   and class2[1] == direction1
            ]

            if not possible_classes:
                raise ValueError(f"No negative class found for "f"dot {dot1}, direction {direction1}")
            class2 = random.choice(possible_classes)
            idx1 = random.choice(self.class_indices[class1])
            idx2 = random.choice(self.class_indices[class2])
            label = 0.0

        image1 = torch.from_numpy(self.images[idx1]).float().unsqueeze(0)
        image2 = torch.from_numpy(self.images[idx2]).float().unsqueeze(0)
        label = torch.tensor(label,dtype=torch.float32)
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
        negative_loss = ((1 - label) * torch.clamp(self.margin - distance,min=0).pow(2))
        return (positive_loss + negative_loss).mean()

# ---------------------------------------------------------
# MODEL SETUP
# ---------------------------------------------------------

cnn = CNN().to(device)
lossFunc = ContrastiveLoss(margin=margin)
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

# ---------------------------------------------------------
# HELPER: EMBEDDING
# ---------------------------------------------------------

def get_embedding(model, image):
    model.eval()
    # Get the device where the model is located
    device = next(model.parameters()).device

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0).unsqueeze(0)

    # Move image to the same device as the model
    image = image.to(device)

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

def evaluate_pairs(
        model,
        query_images,
        query_ids,
        query_directions,
        reference_images=None,
        reference_ids=None,
        reference_directions=None,
        num_pairs=1000
):
    model.eval()

    same_dataset = reference_images is None

    if same_dataset:
        reference_images = query_images
        reference_ids = query_ids
        reference_directions = query_directions

    reference_classes = {}

    for i, (dot_id, direction) in enumerate(zip(reference_ids, reference_directions)):
        key = (int(dot_id), int(direction))
        if key not in reference_classes:
            reference_classes[key] = []
        reference_classes[key].append(i)
    all_classes = list(reference_classes.keys())
    same_scores = []
    different_scores = []

    for _ in range(num_pairs):
        query_idx = random.randrange(len(query_images))
        query_id = int(query_ids[query_idx])
        query_direction = int(query_directions[query_idx])
        query_class = (query_id, query_direction)
        same_class = random.random() < 0.5

        # ------------------------------------------------
        # POSITIVE PAIR
        # ------------------------------------------------

        if same_class:
            if query_class not in reference_classes:
                continue
            possible_indices = reference_classes[query_class]
            # Prevent image from being paired with itself
            if same_dataset:
                possible_indices = [
                    i for i in possible_indices
                    if i != query_idx
                ]
            if not possible_indices:
                continue
            reference_idx = random.choice(possible_indices)
            label = 1

        # ------------------------------------------------
        # NEGATIVE PAIR
        # ------------------------------------------------

        else:

            valid_classes = [
                key for key in all_classes
                if key[0] != query_id
                and key[1] == query_direction
            ]

            if not valid_classes:
                continue

            selected_class = random.choice(valid_classes)

            reference_idx = random.choice(
                reference_classes[selected_class]
            )

            label = 0

        # ------------------------------------------------
        # EMBEDDINGS
        # ------------------------------------------------

        image1 = torch.from_numpy(
            query_images[query_idx]
        ).float().unsqueeze(0).unsqueeze(0).to(device)

        image2 = torch.from_numpy(
            reference_images[reference_idx]
        ).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():

            output1 = model(image1)
            output2 = model(image2)

            similarity = F.cosine_similarity(
                output1,
                output2
            ).item()

        if label == 1:
            same_scores.append(similarity)
        else:
            different_scores.append(similarity)

    return same_scores, different_scores



# ---------------------------------------------------------
# FIND BEST VALIDATION THRESHOLD
# ---------------------------------------------------------

def find_best_threshold(same_scores, different_scores):

    all_scores = same_scores + different_scores

    minimum = min(all_scores)
    maximum = max(all_scores)

    thresholds = np.linspace(minimum, maximum, 500)

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

mode = input("*Select Mode*\nTrain model (0)\n1 to 1 match (1)\nDatabase search (2)\nEnter Input: ")

# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

if mode == '0':

    train_dataset = SiameseDataset(
        train_img,
        train_ids,
        train_direction,
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
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            output1 = cnn(image1)
            output2 = cnn(image2)

            loss = lossFunc(output1, output2, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print("Epoch:", i + 1, "Loss:", avg_loss)

    # -----------------------------------------------------
    # VALIDATION THRESHOLD
    # -----------------------------------------------------

    same_scores, different_scores = evaluate_pairs(
        cnn,
        val_img,
        val_ids,
        val_direction,
        train_img,
        train_ids,
        train_direction,
        num_pairs=2000
    )

    threshold, validation_accuracy = find_best_threshold(
        same_scores,
        different_scores)

    print("Validation threshold:", threshold)
    print("Validation pair accuracy:", validation_accuracy)

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------

    same_test_scores, different_test_scores = evaluate_pairs(
        cnn,
        test_img,
        test_ids,
        test_direction,
        train_img,
        train_ids,
        train_direction,
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

    true_positive = sum(score >= threshold for score in same_test_scores)
    false_negative = sum(score < threshold for score in same_test_scores)
    false_positive = sum(score >= threshold for score in different_test_scores)
    true_negative = sum(score < threshold for score in different_test_scores)
    far = false_positive / (false_positive + true_negative)
    frr = false_negative / (true_positive + false_negative)

    print("Final test pair accuracy:", test_accuracy)
    print("False acceptance rate:", far)
    print("False rejection rate:", frr)
    print("Average same-dot similarity:", np.mean(same_test_scores))
    print("Average different-dot similarity:", np.mean(different_test_scores))

    # -----------------------------------------------------
    # SAVE WEIGHTS
    # -----------------------------------------------------

    save_dir = Path("saved_weights")
    date = str(date.today())
    # Find an available filename
    save_name = save_dir / f"{date}.pth"
    counter = 1
    while save_name.exists():
        save_name = save_dir / f"{date}-{counter}.pth"
        counter += 1

    torch.save(
        {
            "model_state": cnn.state_dict(),
            "mean": float(mean),
            "std": float(std),
            "threshold": float(threshold),
            "embedding_size": embedding_size
        },
        save_name
    )

    print("Finished Training")
    print("Saved as " + str(save_name))

# ---------------------------------------------------------
# PREDICT / COMPARE TWO IMAGES
# ---------------------------------------------------------

elif mode == '1':

    checkpoint = torch.load(
        select_weight,
        map_location=device)

    cnn.load_state_dict(checkpoint["model_state"])
    cnn.eval()
    saved_mean = checkpoint["mean"]
    saved_std = checkpoint["std"]
    if not use_set_threshold:
        threshold = checkpoint["threshold"]
        print("Using Auto Threshold")
    else:
        threshold = set_threshold
        print("Using Manual Threshold")
    test_paths = []

    for path in sorted(one_to_one_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in image_extensions:
            test_paths.append(path)

    if len(test_paths) != 2:
        raise ValueError(
            "Put exactly TWO images in the one_to_one_auth folder.\n"
            "The program will compare those two images.")

    processed_images = []

    for path in test_paths:

        test_image = cv2.imread(str(path))

        if test_image is None:
            raise ValueError(f"Could not read {path}")
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        test_image = (test_image - saved_mean) / saved_std
        processed_images.append(test_image)

    embedding1 = get_embedding(cnn, processed_images[0])
    embedding2 = get_embedding(cnn, processed_images[1])

    similarity = compare_embeddings(embedding1, embedding2)

    print("Image 1:", test_paths[0].name)
    print("Image 2:", test_paths[1].name)
    print("Cosine similarity:", similarity)
    print("Decision threshold:", threshold)
    if similarity >= threshold:
        print("Prediction: SAME DOT")
    else:
        print("Prediction: DIFFERENT DOT")

# ---------------------------------------------------------
# Database Search
# ---------------------------------------------------------

elif mode == '2':

    checkpoint = torch.load(
        select_weight,
        map_location=device
    )

    cnn.load_state_dict(checkpoint["model_state"])
    cnn.eval()

    saved_mean = checkpoint["mean"]
    saved_std = checkpoint["std"]

    if not use_set_threshold:
        threshold = checkpoint["threshold"]
        print("Using Auto Threshold")
    else:
        threshold = set_threshold
        print("Using Manual Threshold")

    test_paths = []
    auth_paths = []

    for path in sorted(test_img_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in image_extensions:
            test_paths.append(path)

    for path in sorted(auth_data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in image_extensions:
            auth_paths.append(path)

    # The test image folder should contain exactly 4 images:
    if len(test_paths) != 4:
        raise ValueError(
            f"Expected exactly 4 test images in {test_img_dir}, "
            f"but found {len(test_paths)}."
        )


    database_pattern = re.compile(r"D(\d+)R(\d+)A(\d+)",re.IGNORECASE)

    # -----------------------------------------------------
    # SORT TEST IMAGES BY ANGLE
    # -----------------------------------------------------

    test_by_direction = {}

    for i, path in enumerate(test_paths):
        angle = (i+1)
        test_by_direction[angle] = path

    test_embeddings = {}

    for angle in [1, 2, 3, 4]:

        path = test_by_direction[angle]

        test_image = cv2.imread(str(path))

        if test_image is None:
            raise ValueError(f"Could not read {path}")

        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        test_image = (test_image - saved_mean) / saved_std
        test_embeddings[angle] = get_embedding(cnn,test_image)

    print("\nTest images:")

    for angle in [1, 2, 3, 4]:
        print(
            f"  A{angle}: "
            f"{test_by_direction[angle].name}"
        )

    database = {}

    for path in auth_paths:
        match = database_pattern.fullmatch(path.stem)
        if not match:
            raise ValueError(
                f"Unexpected database filename: {path.name}\n"
                "Expected format such as D0001R1A1.jpg"
            )
        dot_num, rep_num, angle_num = match.groups()
        dot_id = int(dot_num)
        replicate_id = int(rep_num)
        angle = int(angle_num)
        if angle not in {1, 2, 3, 4}:
            continue
        # Candidate is identified by D + R
        candidate_id = (dot_id, replicate_id)

        if candidate_id not in database:
            database[candidate_id] = {}

        database[candidate_id][angle] = path

    complete_database = {}

    for candidate_id, images in database.items():

        missing = {
            1, 2, 3, 4
        } - set(images.keys())

        if not missing:
            complete_database[candidate_id] = images
        else:
            print(
                f"Skipping D{candidate_id[0]}R{candidate_id[1]} "
                f"(missing A{sorted(missing)})"
            )

    print(
        "Total database samples:",
        len(complete_database)
    )

    similarity_scores = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    candidate_ids = []

    for candidate_id, candidate_images in complete_database.items():
        candidate_ids.append(candidate_id)

        for angle in [1, 2, 3, 4]:
            path = candidate_images[angle]
            auth_image = cv2.imread(str(path))

            if auth_image is None:
                raise ValueError(
                    f"Could not read {path}"
                )
            auth_image = cv2.cvtColor(auth_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            auth_image = (auth_image - saved_mean) / saved_std
            auth_embedding = get_embedding(cnn, auth_image)
            similarity = compare_embeddings(test_embeddings[angle], auth_embedding)
            similarity_scores[angle].append(similarity)

    # -----------------------------------------------------
    # SUM THE FOUR SIMILARITY SCORES
    # -----------------------------------------------------

    total_scores = []

    for i, candidate_id in enumerate(candidate_ids):

        total_similarity = (
            similarity_scores[1][i]
            + similarity_scores[2][i]
            + similarity_scores[3][i]
            + similarity_scores[4][i]
        )

        total_scores.append(total_similarity)

    # -----------------------------------------------------
    # CREATE ALL-FOUR-ABOVE-THRESHOLD LIST
    # -----------------------------------------------------

    all_four_match = []

    for i in range(len(candidate_ids)):

        matches_all_four = (
            similarity_scores[1][i] >= threshold
            and similarity_scores[2][i] >= threshold
            and similarity_scores[3][i] >= threshold
            and similarity_scores[4][i] >= threshold
        )

        all_four_match.append(matches_all_four)

    # -----------------------------------------------------
    # SORT DATABASE BY TOTAL SIMILARITY
    # -----------------------------------------------------

    ranked_indices = sorted(
        range(len(candidate_ids)),
        key=lambda i: total_scores[i],
        reverse=True
    )

    # -----------------------------------------------------
    # PRINT TOP FIVE
    # -----------------------------------------------------

    num_results = min(5, len(ranked_indices))

    print("\n" + "=" * 70)
    print("TOP DATABASE MATCHES")
    print("=" * 70)

    for rank in range(num_results):

        i = ranked_indices[rank]
        dot_id, replicate_id = candidate_ids[i]

        print(
            f"\nRank {rank + 1}: "
            f"D{dot_id:04d}R{replicate_id}"
        )

        print(
            f"  A1 similarity: "
            f"{similarity_scores[1][i]:.4f}"
        )

        print(
            f"  A2 similarity: "
            f"{similarity_scores[2][i]:.4f}"
        )

        print(
            f"  A3 similarity: "
            f"{similarity_scores[3][i]:.4f}"
        )

        print(
            f"  A4 similarity: "
            f"{similarity_scores[4][i]:.4f}"
        )

        print(
            f"  Total similarity: "
            f"{total_scores[i]:.4f}"
        )

        print(
            f"  Authentication Pass: "
            f"{all_four_match[i]}"
        )


    matching_candidates = [
        candidate_ids[i]
        for i in range(len(candidate_ids))
        if all_four_match[i]
    ]

    print("\n" + "=" * 70)
    print("DATABASE MATCH")
    print("=" * 70)

    if matching_candidates:
        for dot_id, replicate_id in matching_candidates:
            print(f"D{dot_id:04d}")
    else:
        print("No database match")

else:
    print("Invalid Input")