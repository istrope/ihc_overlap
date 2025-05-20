import re
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from skimage import filters, morphology
from scipy.stats import pearsonr
from collections import defaultdict

# Configure logging
logging.basicConfig(
    filename="colocalization_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Function to extract MAX_NUMBER (imaging session)
def get_max_number(filename):
    match = re.match(r"(MAX_\d+)", filename)
    return match.group(1) if match else "UNKNOWN"

# Function to parse file structure
def parse_file_structure(file_path):
    session_files = defaultdict(lambda: defaultdict(list))
    current_directory = ""

    with open(file_path, "r") as f:
        file_contents = f.readlines()

    for line in file_contents:
        line = line.strip().strip("'")

        if line.endswith("/"):
            current_directory = line
        else:
            full_file_path=f'{current_directory}{line}'
            session = get_max_number(line)

            if session != 'UNKNOWN':
                session_files[session]['files'].append(full_file_path)

    return session_files

def compute_global_thresholds(session_files):
    ch1_values, ch2_values = [], []

    for files in session_files.values():
        for file in files["files"]:
            if "hpg only" in file.lower():  # Ensure case-insensitive matching
                img = tiff.imread(file)
                ch1_values.append(filters.threshold_otsu(img[0]))
                ch2_values.append(filters.threshold_otsu(img[1]))

    # Calculate the average thresholds, applying the scaling factor to ch1
    if ch1_values:  # Check if the list is not empty to avoid ZeroDivisionError
        avg_ch1_threshold = np.mean(ch1_values) * 0.85
    else:
        avg_ch1_threshold = None  # Handle cases where no hpg_only files are found

    if ch2_values:  # Check if the list is not empty
        avg_ch2_threshold = np.mean(ch2_values)
    else:
        avg_ch2_threshold = None  # Handle cases where no hpg_only files are found

    return avg_ch1_threshold, avg_ch2_threshold

# Apply thresholding
def apply_threshold(image, threshold):
    return image * (image > threshold)

# Normalize per image after filtering
def normalize_per_image(image):
    max_intensity = np.max(image)
    return image / (max_intensity + 1e-6)

# Create nucleus and cytoplasmic masks
def create_nucleus_and_cyto_masks(dapi_channel):
    dapi_threshold = filters.threshold_otsu(dapi_channel)
    nucleus_mask = dapi_channel > dapi_threshold
    cyto_mask = ~nucleus_mask
    cyto_mask = morphology.remove_small_holes(cyto_mask, area_threshold=50)
    return nucleus_mask, cyto_mask

# Compute Pearson Correlation Coefficient
def compute_pearson_correlation(ch1, ch2):
    return pearsonr(ch1.flatten(), ch2.flatten())[0]

# Compute Percent Colocalization
def compute_percent_colocalization(ch1, ch2, mask, area_mask):
    ch1_masked = np.where(mask, ch1, 0)
    ch2_masked = np.where(mask, ch2, 0)

    overlap = np.logical_and(ch1_masked > 0, ch2_masked > 0)
    overlap_area = np.sum(overlap)

    total_area = np.sum(np.logical_and(area_mask, mask))

    return (overlap_area / total_area) * 100 if total_area != 0 else np.nan

# Save visualization images
def plot_filtered_visualization(file_path, ch1_norm, ch2_norm, dapi_filtered, ch1_thresh, ch2_thresh):
    plot_file = re.sub('Batch_1', 'Batch_1_out', file_path)
    plot_file = re.sub('.tif', '.png', plot_file)

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(ch1_norm, cmap='Reds')
    axes[0].set_title("Normalized Channel 1")

    axes[1].imshow(ch2_norm, cmap='Greens')
    axes[1].set_title("Normalized Channel 2")

    axes[2].imshow(dapi_filtered, cmap='Blues')
    axes[2].set_title("Filtered DAPI (Nucleus Mask)")

    overlay = np.dstack((ch1_thresh, ch2_thresh, np.zeros_like(ch1_thresh)))
    axes[3].imshow(overlay)
    axes[3].set_title("Colocalization Overlay")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    logging.info(f"Saved filtered visualization: {plot_file}")

def process_image(file_path, global_thresholds):
    filename = os.path.basename(file_path)
    session = get_max_number(filename)

    try:
        logging.info(f"Processing image: {filename}, Session: {session}")

        img = tiff.imread(file_path)
        ch1_raw, ch2_raw, dapi = img[0], img[1], img[2]

        ch1_thresh = apply_threshold(ch1_raw, global_thresholds[0])
        ch2_thresh = apply_threshold(ch2_raw, global_thresholds[1])

        ch1_norm = normalize_per_image(ch1_thresh)
        ch2_norm = normalize_per_image(ch2_thresh)

        nucleus_mask, cyto_mask = create_nucleus_and_cyto_masks(dapi)

        pearson_corr = compute_pearson_correlation(ch1_thresh, ch2_thresh)
        percent_coloc_nucleus = compute_percent_colocalization(
            ch1_thresh, ch2_thresh, nucleus_mask, np.logical_or(ch1_thresh > 0, ch2_thresh > 0)
        )
        percent_coloc_cyto = compute_percent_colocalization(ch1_thresh, ch2_thresh, cyto_mask, np.logical_or(ch1_thresh > 0, ch2_thresh > 0))

        plot_filtered_visualization(file_path, ch1_norm, ch2_norm, dapi * nucleus_mask, ch1_thresh, ch2_thresh)

        return {
            "MAX_NUMBER": session,
            "File Basename": filename,
            "Pearson Correlation": pearson_corr,
            "Percent Coloc Nucleus": percent_coloc_nucleus,
            "Percent Coloc Cytoplasm": percent_coloc_cyto,
            "Ch1 Threshold Used": global_thresholds[0],
            "Ch2 Threshold Used": global_thresholds[1]
        }
    except Exception as e:
        logging.error(f"Error processing image {filename}: {str(e)}")
        return {}

# Define and parse the dataset
file_path = "Image_files.txt"
session_files = parse_file_structure(file_path)

# Calculate global thresholds
global_thresholds = compute_global_thresholds(session_files)

# Process all images
results = []
for session, data in session_files.items():
    for file_path in data["files"]:
        result = process_image(file_path, global_thresholds)
        results.append(result)

# Save results
df = pd.DataFrame(results)
df.to_csv("colocalization_results.csv", index=False)
logging.info("Saved results to colocalization_results.csv")
print("Colocalization analysis completed.")

