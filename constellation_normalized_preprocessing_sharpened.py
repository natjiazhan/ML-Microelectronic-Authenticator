
"""
Normalized Optical-PUF Constellation Preprocessing Pipeline
-----------------------------------------------------------

Purpose:
    Detect each printed optical-PUF dot, normalize its position and scale,
    preserve the full-resolution optical structure, and optionally create
    a CNN-sized enhanced image.

Default outputs for each input image:
    *_registered_768.jpg   -> normalized, centered, minimally altered
    *_enhanced_768.jpg     -> registered + CLAHE + mild denoising
    *_sharpened_768.jpg    -> enhanced + mild luminance-only sharpening

    *_registered_448.jpg   -> registered branch resized for CNN testing
    *_enhanced_448.jpg     -> enhanced branch resized for CNN testing
    *_sharpened_448.jpg    -> sharpened branch resized for CNN testing

Optional troubleshooting outputs:
    *_debug.jpg        -> original image with detected circle/center
    *_mask.jpg         -> binary mask used during detection

Diagnostics are kept in the code but DISABLED by default.
Set SAVE_DIAGNOSTICS = True near the top of the script if needed.

Recommended raw folder structure:

    Raw_Constellations/
    ├── Dot_0001/
    │   ├── angle_1.jpg
    │   ├── angle_2.jpg
    │   ├── angle_3.jpg
    │   └── angle_4.jpg
    ├── Dot_0002/
    │   └── ...
    └── ...

The same folder structure is recreated in the output folder.

Requirements:
    pip install opencv-python numpy

Tkinter is included with most standard Windows Python installations.
"""

from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import cv2
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

# Full-resolution normalized image size.
# 768 preserves considerably more fine mica detail than 224.
NORMALIZED_SIZE = 768

# CNN input image size.
# 448 is a good starting point for retaining fine structure.
CNN_SIZE = 448

# Fraction of half-width occupied by the physical dot.
# 0.88 means the dot diameter is about 88% of the image width,
# leaving a small, consistent black border.
DOT_DIAMETER_FRACTION = 0.88

# Save masks/debug images only when troubleshooting.
SAVE_DIAGNOSTICS = False

# Conservative image enhancement.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Mild denoising values. Keep these low so mica texture is not erased.
DENOISE_LUMINANCE = 3
DENOISE_COLOR = 3

# Mild luminance-only sharpening.
SHARPEN_AMOUNT = 0.7
SHARPEN_SIGMA = 1.2


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def apply_rgb_clahe(image_bgr,
                    clip_limit=CLAHE_CLIP_LIMIT,
                    tile_grid=CLAHE_TILE_GRID):
    """
    Apply CLAHE to luminance only, preserving RGB/color relationships.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )

    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )


def sharpen_luminance(image_bgr,
                      amount=SHARPEN_AMOUNT,
                      sigma=SHARPEN_SIGMA):
    """
    Apply mild unsharp masking to the luminance channel only.

    This sharpens flake boundaries while reducing the chance of creating
    false color fringes in the blue/green/red reflections.
    """

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    blurred_l = cv2.GaussianBlur(
        l_channel,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    sharpened_l = cv2.addWeighted(
        l_channel,
        1.0 + amount,
        blurred_l,
        -amount,
        0
    )

    sharpened_lab = cv2.merge(
        (sharpened_l, a_channel, b_channel)
    )

    return cv2.cvtColor(
        sharpened_lab,
        cv2.COLOR_LAB2BGR
    )


def enhance_image(image_bgr):
    """
    Conservative enhancement intended to normalize local contrast without
    destroying the fine spatial/color information in the mica reflections.
    """
    enhanced = apply_rgb_clahe(image_bgr)

    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced,
        None,
        h=DENOISE_LUMINANCE,
        hColor=DENOISE_COLOR,
        templateWindowSize=7,
        searchWindowSize=21
    )

    return enhanced


# ============================================================
# DOT DETECTION
# ============================================================

def detect_constellation_region(image_bgr):
    """
    Detect the main optical-PUF dot region.

    Returns:
        cx, cy       detected center
        radius       detected radius
        mask         binary detection mask
    """
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    background_level = np.percentile(
        blurred,
        25
    )

    threshold_value = max(
        background_level + 12,
        np.percentile(blurred, 65)
    )

    _, mask = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Connect nearby reflective regions.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    # Remove small isolated regions.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError(
            "No constellation region was detected."
        )

    height, width = gray.shape

    image_center = np.array(
        [width / 2.0, height / 2.0],
        dtype=np.float32
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 0.002 * width * height:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(
            contour
        )

        distance_from_center = np.linalg.norm(
            np.array([cx, cy], dtype=np.float32)
            - image_center
        )

        # Prefer large regions near the center of the photo.
        score = area / (
            1.0 + 0.002 * distance_from_center
        )

        candidates.append(
            (score, cx, cy, radius, contour)
        )

    if not candidates:
        raise RuntimeError(
            "Only small/noisy bright regions were detected."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    _, cx, cy, radius, contour = candidates[0]

    # Refine center/radius from the selected connected region.
    selected_mask = np.zeros_like(gray)

    cv2.drawContours(
        selected_mask,
        [contour],
        -1,
        255,
        thickness=-1
    )

    ys, xs = np.where(
        selected_mask > 0
    )

    if len(xs) > 0:
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        distances = np.sqrt(
            (xs - cx) ** 2
            + (ys - cy) ** 2
        )

        # Avoid one stray point making the dot radius too large.
        radius = float(
            np.percentile(
                distances,
                98
            )
        )

    return cx, cy, radius, mask


# ============================================================
# NORMALIZATION / REGISTRATION
# ============================================================

def normalize_dot(image_bgr,
                  cx,
                  cy,
                  radius,
                  output_size=NORMALIZED_SIZE,
                  diameter_fraction=DOT_DIAMETER_FRACTION,
                  mask_outside=True):
    """
    Normalize each physical dot to the same:
        - image center
        - apparent diameter
        - square canvas size

    This is scale/translation normalization only.
    It does NOT rotate the image.

    Rotation registration can be added later after we determine whether
    orientation differences are significant and whether a stable rotation
    reference exists.
    """

    if radius <= 0:
        raise ValueError("Detected radius must be greater than zero.")

    target_center = (output_size - 1) / 2.0
    target_radius = (output_size * diameter_fraction) / 2.0

    scale = target_radius / radius

    # Affine transform:
    # x' = scale*x + tx
    # y' = scale*y + ty
    tx = target_center - scale * cx
    ty = target_center - scale * cy

    matrix = np.array(
        [
            [scale, 0.0, tx],
            [0.0, scale, ty]
        ],
        dtype=np.float32
    )

    registered = cv2.warpAffine(
        image_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    if mask_outside:
        circular_mask = np.zeros(
            (output_size, output_size),
            dtype=np.uint8
        )

        cv2.circle(
            circular_mask,
            (int(round(target_center)), int(round(target_center))),
            int(round(target_radius)),
            255,
            thickness=-1
        )

        registered = cv2.bitwise_and(
            registered,
            registered,
            mask=circular_mask
        )

    return registered


# ============================================================
# SINGLE IMAGE PROCESSING
# ============================================================

def process_image(input_path,
                  output_dir,
                  normalized_size=NORMALIZED_SIZE,
                  cnn_size=CNN_SIZE):

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {input_path}"
        )

    # Step 1: Detect dot.
    cx, cy, radius, detection_mask = detect_constellation_region(
        image
    )

    # Step 2: Standardize center + apparent diameter.
    registered = normalize_dot(
        image,
        cx,
        cy,
        radius,
        output_size=normalized_size,
        diameter_fraction=DOT_DIAMETER_FRACTION,
        mask_outside=True
    )

    # Step 3: Apply conservative enhancement at full normalized resolution.
    enhanced = enhance_image(
        registered
    )

    # Re-apply a clean circular mask after CLAHE/denoising so processing
    # does not create small values outside the physical dot.
    target_center = int(round((normalized_size - 1) / 2.0))
    target_radius = int(round(
        normalized_size * DOT_DIAMETER_FRACTION / 2.0
    ))

    clean_circle = np.zeros(
        (normalized_size, normalized_size),
        dtype=np.uint8
    )

    cv2.circle(
        clean_circle,
        (target_center, target_center),
        target_radius,
        255,
        thickness=-1
    )

    enhanced = cv2.bitwise_and(
        enhanced,
        enhanced,
        mask=clean_circle
    )

    # Step 4: Create a sharpened experimental branch.
    sharpened = sharpen_luminance(
        enhanced,
        amount=SHARPEN_AMOUNT,
        sigma=SHARPEN_SIGMA
    )

    # Re-apply the same clean circular mask after sharpening.
    sharpened = cv2.bitwise_and(
        sharpened,
        sharpened,
        mask=clean_circle
    )

    # Step 5: Create same-size CNN inputs for controlled comparison.
    registered_cnn = cv2.resize(
        registered,
        (cnn_size, cnn_size),
        interpolation=cv2.INTER_AREA
    )

    enhanced_cnn = cv2.resize(
        enhanced,
        (cnn_size, cnn_size),
        interpolation=cv2.INTER_AREA
    )

    sharpened_cnn = cv2.resize(
        sharpened,
        (cnn_size, cnn_size),
        interpolation=cv2.INTER_AREA
    )

    stem = input_path.stem

    registered_path = (
        output_dir / f"{stem}_registered_{normalized_size}.jpg"
    )

    enhanced_path = (
        output_dir / f"{stem}_enhanced_{normalized_size}.jpg"
    )

    sharpened_path = (
        output_dir / f"{stem}_sharpened_{normalized_size}.jpg"
    )

    registered_cnn_path = (
        output_dir / f"{stem}_registered_{cnn_size}.jpg"
    )

    enhanced_cnn_path = (
        output_dir / f"{stem}_enhanced_{cnn_size}.jpg"
    )

    sharpened_cnn_path = (
        output_dir / f"{stem}_sharpened_{cnn_size}.jpg"
    )

    cv2.imwrite(
        str(registered_path),
        registered,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    # cv2.imwrite(
    #     str(enhanced_path),
    #     enhanced,
    #     [cv2.IMWRITE_JPEG_QUALITY, 95]
    # )

    # cv2.imwrite(
    #     str(sharpened_path),
    #     sharpened,
    #     [cv2.IMWRITE_JPEG_QUALITY, 95]
    # )

    # cv2.imwrite(
    #     str(registered_cnn_path),
    #     registered_cnn,
    #     [cv2.IMWRITE_JPEG_QUALITY, 95]
    # )

    # cv2.imwrite(
    #     str(enhanced_cnn_path),
    #     enhanced_cnn,
    #     [cv2.IMWRITE_JPEG_QUALITY, 95]
    # )

    # cv2.imwrite(
    #     str(sharpened_cnn_path),
    #     sharpened_cnn,
    #     [cv2.IMWRITE_JPEG_QUALITY, 95]
    # )

    # --------------------------------------------------------
    # OPTIONAL DIAGNOSTICS
    # --------------------------------------------------------
    # These remain available in the script but are disabled
    # by default so the dataset is not cluttered.
    # Set SAVE_DIAGNOSTICS = True near the top to enable them.
    # --------------------------------------------------------

    if SAVE_DIAGNOSTICS:
        debug_image = image.copy()

        cv2.circle(
            debug_image,
            (int(round(cx)), int(round(cy))),
            int(round(radius)),
            (0, 255, 0),
            4
        )

        cv2.circle(
            debug_image,
            (int(round(cx)), int(round(cy))),
            8,
            (0, 0, 255),
            -1
        )

        debug_path = (
            output_dir / f"{stem}_debug.jpg"
        )

        mask_path = (
            output_dir / f"{stem}_mask.jpg"
        )

        cv2.imwrite(
            str(debug_path),
            debug_image
        )

        cv2.imwrite(
            str(mask_path),
            detection_mask
        )

    print(
        f"Processed: {input_path.name}"
    )
    print(
        f"  Detected center: ({cx:.1f}, {cy:.1f})"
    )
    print(
        f"  Detected radius: {radius:.1f} px"
    )


# ============================================================
# FOLDER PROCESSING
# ============================================================

def process_folder(input_folder,
                   output_folder,
                   normalized_size=NORMALIZED_SIZE,
                   cnn_size=CNN_SIZE):

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    files = [
        path
        for path in input_folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in supported_extensions
        )
    ]

    if not files:
        messagebox.showwarning(
            "No Images Found",
            "No supported image files were found in the selected folder."
        )
        return

    print(
        f"\nFound {len(files)} image(s).\n"
    )

    successful = 0
    failed = 0

    for index, file in enumerate(
        sorted(files),
        start=1
    ):
        print(
            f"[{index}/{len(files)}] {file}"
        )

        try:
            relative_parent = (
                file.parent.relative_to(
                    input_folder
                )
            )

            image_output_folder = (
                output_folder
                / relative_parent
            )

            process_image(
                file,
                image_output_folder,
                normalized_size=normalized_size,
                cnn_size=cnn_size
            )

            successful += 1

        except Exception as exc:
            failed += 1
            print(
                f"  FAILED: {exc}"
            )

    summary = (
        "Processing complete.\n\n"
        f"Images found: {len(files)}\n"
        f"Successfully processed: {successful}\n"
        f"Failed: {failed}\n\n"
        f"Normalized image size: {normalized_size} x {normalized_size}\n"
        f"CNN image size: {cnn_size} x {cnn_size}\n"
        f"Diagnostics saved: {SAVE_DIAGNOSTICS}\n\n"
        f"Output folder:\n{output_folder}"
    )

    print(
        "\n" + summary
    )

    messagebox.showinfo(
        "Constellation Processing Complete",
        summary
    )


# ============================================================
# FOLDER SELECTION GUI
# ============================================================

def select_folder(title):
    folder = filedialog.askdirectory(
        title=title
    )

    if folder:
        return Path(folder)

    return None


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    root = Tk()
    root.withdraw()

    print(
        "Normalized Constellation Image Preprocessing"
    )
    print(
        "--------------------------------------------\n"
    )

    input_folder = select_folder(
        "Select Folder Containing Raw Constellation Images"
    )

    if input_folder is None:
        print(
            "No input folder selected. Program cancelled."
        )
        root.destroy()
        return

    output_folder = select_folder(
        "Select Folder for Processed Constellation Images"
    )

    if output_folder is None:
        print(
            "No output folder selected. Program cancelled."
        )
        root.destroy()
        return

    try:
        if (
            input_folder.resolve()
            == output_folder.resolve()
        ):
            messagebox.showerror(
                "Invalid Folder Selection",
                "The input and output folders must be different."
            )
            root.destroy()
            return
    except Exception:
        pass

    print(
        f"Input folder:\n{input_folder}\n"
    )
    print(
        f"Output folder:\n{output_folder}\n"
    )

    process_folder(
        input_folder,
        output_folder,
        normalized_size=NORMALIZED_SIZE,
        cnn_size=CNN_SIZE
    )

    root.destroy()


if __name__ == "__main__":
    main()
