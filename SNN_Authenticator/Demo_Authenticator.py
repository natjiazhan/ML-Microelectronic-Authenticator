import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
test_img_dir = Path("Query_processed")
auth_data_dir = Path("Auth_database")
raw_dir = Path("Query_raw")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

select_weight = Path("saved_weights/2026-10-21.pth")
embedding_cache_path = Path("auth_database_embeddings.pt")

set_threshold = 0.87
use_set_threshold = True
reset_query = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------------

p_read_array = []
write_array = []

for path in sorted(raw_dir.iterdir()):
    if path.is_file() and path.suffix.lower() in image_extensions:
        p_read_array.append(str(path))
        write_array.append(str(test_img_dir / f"{path.stem}{path.suffix}"))

p_img = []
for i in range(len(p_read_array)):
    p_img.append(cv2.imread(p_read_array[i]))

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

#outputs 224x224 image
final_size = 224
#crop size
size = 240
img_processed = np.zeros((len(p_img),final_size,final_size),dtype=np.uint8)
if len(p_img) == 4:
    for i in range(len(p_img)//4):
        combined_img = combine(p_img[i*4],p_img[i*4+1],p_img[i*4+2],p_img[i*4+3],40)
        x,y = get_center(combined_img)
        for j in range(0,4):
            img_filtered = process(p_img[i*4+j],20)
            img_processed[i*4+j] = crop(img_filtered,x,y,size,final_size)
elif len(p_img) != 0:
    raise ValueError (
        f"Expected exactly 4 images in the Query_raw"
        )
for i in range(len(write_array)):
    cv2.imwrite(write_array[i],img_processed[i])
# ---------------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------------

read_array = []
for path in sorted(auth_data_dir.iterdir()):
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

test_img = []
test_ids = []
test_direction = []
for i in range(len(img)):
    test_img.append(img[i])
    test_ids.append(sampleID[i])
    test_direction.append(direction[i])

# ---------------------------------------------------------
# CONVERT TO GRAYSCALE
# ---------------------------------------------------------

def convert_to_gray(image_list):
    gray_array = []
    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_array.append(gray)
    return np.array(gray_array, dtype=np.float32)

test_img = convert_to_gray(test_img)

# ---------------------------------------------------------
# NORMALIZE USING TRAINING DATA ONLY
# ---------------------------------------------------------

mean = test_img.mean()
std = test_img.std()

if std == 0:
    raise ValueError("Training image standard deviation is zero.")

test_img = (test_img - mean) / std

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

embedding_size = 128
batch_size = 32

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
# Database Search
# ---------------------------------------------------------

start_time = time.perf_counter()
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
    print("Using Auto Threshold: " + str(threshold))
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


# -----------------------------------------------------
# LOAD OR COMPUTE DATABASE EMBEDDINGS
# -----------------------------------------------------

# Create a signature that changes if the model checkpoint,
# normalization values, or database files change.
database_file_info = []

for path in sorted(auth_data_dir.iterdir()):
    if path.is_file() and path.suffix.lower() in image_extensions:
        stat = path.stat()
        database_file_info.append(
            (
                str(path.resolve()),
                stat.st_mtime_ns,
                stat.st_size
            )
        )

checkpoint_stat = select_weight.stat()

cache_signature = (
    str(select_weight.resolve()),
    checkpoint_stat.st_mtime_ns,
    checkpoint_stat.st_size,
    float(saved_mean),
    float(saved_std),
    tuple(database_file_info)
)

cached_data = None

if embedding_cache_path.exists():
    try:
        cached_data = torch.load(
            embedding_cache_path,
            map_location="cpu",
            weights_only=True
        )

        if cached_data["signature"] != cache_signature:
            print("Database or model changed. Rebuilding embedding cache.")
            cached_data = None
        else:
            print("Using cached database embeddings.")

    except (OSError, RuntimeError, KeyError, ValueError, EOFError):
        print("Embedding cache is invalid. Rebuilding it.")
        cached_data = None

# Rebuild the cache if it does not exist or is outdated.
if cached_data is None:
    print("Computing database embeddings...")

    database_embeddings = {}
    database_batch_images = []
    database_batch_keys = []

    # Collect images in candidate/angle order.
    for candidate_id, candidate_images in complete_database.items():
        for angle in [1, 2, 3, 4]:
            path = candidate_images[angle]
            auth_image = cv2.imread(str(path))

            if auth_image is None:
                raise ValueError(f"Could not read {path}")

            auth_image = cv2.cvtColor(auth_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            auth_image = (auth_image - saved_mean) / saved_std
            database_batch_images.append(auth_image)
            database_batch_keys.append((candidate_id, angle))

    cnn.eval()

    with torch.no_grad():
        for start in range(
            0, len(database_batch_images), batch_size
        ):
            batch_images = np.stack(
                database_batch_images[start:start + batch_size]
            )

            batch_tensor = torch.from_numpy(
                batch_images
            ).float().unsqueeze(1).to(device)

            batch_embeddings = cnn(batch_tensor).cpu()

            batch_keys = database_batch_keys[
                start:start + batch_size
            ]

            for key, embedding in zip(
                batch_keys, batch_embeddings
            ):
                candidate_id, angle = key

                if candidate_id not in database_embeddings:
                    database_embeddings[candidate_id] = {}

                database_embeddings[candidate_id][angle] = (
                    embedding.clone()
                )

    # Save embeddings on the CPU to keep the cache portable.
    embedding_cache_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    torch.save(
        {
            "signature": cache_signature,
            "embeddings": database_embeddings
        },
        embedding_cache_path
    )

    print("Saved database embedding cache.")

else:
    database_embeddings = cached_data["embeddings"]

# -----------------------------------------------------
# COMPUTE SIMILARITIES USING CACHED EMBEDDINGS
# -----------------------------------------------------

candidate_ids = list(complete_database.keys())

similarity_scores = {
    1: [],
    2: [],
    3: [],
    4: []
}

for candidate_id in candidate_ids:
    for angle in [1, 2, 3, 4]:
        auth_embedding = database_embeddings[
            candidate_id
        ][angle].to(device)

        similarity = compare_embeddings(
            test_embeddings[angle],
            auth_embedding.unsqueeze(0)
        )

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
end_time = time.perf_counter()
execution_time = end_time - start_time
print("Execution time: " + str(execution_time))

# -----------------------------------------------------
# Reset Query
# -----------------------------------------------------
if reset_query:
    for file_path in test_img_dir.iterdir():
        if file_path.is_file():
            file_path.unlink()
    for file_path in raw_dir.iterdir():
        if file_path.is_file():
            file_path.unlink()