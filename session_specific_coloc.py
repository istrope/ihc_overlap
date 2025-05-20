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

    condition_keywords = {
        "hpg only": "hpg only",
        "chx and tig": "chx + tig",
        "chx 100ug": "chx",
        "tig 50um": "tig",
        "tig 8um": "tig low dose",
        "no hpg": "no hpg",
        "plus chx": "chx treated",
        "plus chx veh": "chx vehicle",
        "plus chx crb": "chx + crb",
        "mitofuncat crb with tig": "mito crb + tig",
        "no chx": "no chx",
        "si_chemo": "si chemo",
        "si only": "si only",
        "oxa veh": "oxa vehicle",
        "oxa crb": "oxa + crb",
        "nc veh": "nc vehicle",
        "nc crb": "nc + crb"
    }

    with open(file_path, "r") as f:
        file_contents = f.readlines()

    for line in file_contents:
        line = line.strip().strip("'")

        if line.endswith("/"):
            current_directory = f"{line}"
            continue

        file_path = f"{current_directory}{line}"
        session = get_max_number(line)
        if session is None:
            continue

        condition = "unknown"
        lowercase_filename = line.lower()
        for keyword, mapped_condition in condition_keywords.items():
            if keyword in lowercase_filename:
                condition = mapped_condition
                break

        if condition == 'unknown':
            continue

        session_files[session][condition].append(file_path)

    return session_files

# Compute thresholds for each session using HPG-only images
def get_hpg_thresholds(control_files, scaling_ch1=0.85, scaling_ch2=1):
    thresholds = {}
    all_ch1_thresholds, all_ch2_thresholds = [], []

    for session, files in control_files.items():
        ch1_values, ch2_values = [], []
        for file in files:
            img = tiff.imread(file)
            ch1_values.append(filters.threshold_otsu(img[0]))
            ch2_values.append(filters.threshold_otsu(img[1]))
        
        if ch1_values and ch2_values:
            thresholds[session] = (
                np.mean(ch1_values) * scaling_ch1, 
                np.mean(ch2_values) * scaling_ch2
            )
            all_ch1_thresholds.extend(ch1_values)
            all_ch2_thresholds.extend(ch2_values)

    avg_ch1_threshold = np.mean(all_ch1_thresholds) * scaling_ch1 if all_ch1_thresholds else None
    avg_ch2_threshold = np.mean(all_ch2_thresholds) * scaling_ch2 if all_ch2_thresholds else None

    return thresholds, avg_ch1_threshold, avg_ch2_threshold

# Apply thresholding
def apply_threshold(image, threshold):
    return image * (image > threshold)

# Normalize per image after filtering
def normalize_per_image(image):
    max_intensity = np.max(image)
    return image / (max_intensity + 1e-6)

# Create nucleus and cytoplasmic (mitochondria) masks
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
    # Apply the mask to both channels before computing overlap
    ch1_masked = np.where(mask, ch1, 0)
    ch2_masked = np.where(mask, ch2, 0)

    # Calculate overlap only within the masked region
    overlap = np.logical_and(ch1_masked > 0, ch2_masked > 0)
    overlap_area = np.sum(overlap)

    # Apply area_mask to compute the total area within the mask for calculation
    total_area = np.sum(np.logical_and(area_mask, mask))

    # Return percentage of overlap relative to the total area in the mask
    return (overlap_area / total_area) * 100 if total_area != 0 else np.nan

# Save visualization images in Batch1_Out
def plot_filtered_visualization(file_path, ch1_norm, ch2_norm, dapi_filtered, ch1_thresh, ch2_thresh):
    plot_file = re.sub('Batch_1', 'Batch_1_out', file_path)
    plot_file = re.sub('.tif', '.png', plot_file)

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(ch1_norm, cmap='Reds')
    axes[0].set_title("Normalized HPG-Azide 494")

    axes[1].imshow(ch2_norm, cmap='Greens')
    axes[1].set_title("Normalized COX4 488")

    axes[2].imshow(dapi_filtered, cmap='Blues')
    axes[2].set_title("Filtered DAPI (Nucleus Mask)")

    overlay = np.dstack((ch1_thresh, ch2_thresh, np.zeros_like(ch1_thresh)))
    axes[3].imshow(overlay)
    axes[3].set_title("Colocalization Overlay")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    logging.info(f"Saved filtered visualization: {plot_file}")

# Process a single image
def process_image(file_path, condition, ref_thresholds, avg_thresholds):
    filename = os.path.basename(file_path)
    session = get_max_number(filename)

    try:
        ref_ch1_threshold, ref_ch2_threshold = avg_thresholds

        logging.info(f"Processing image: {filename}, Session: {session}, Condition: {condition}")

        img = tiff.imread(file_path)
        ch1_raw, ch2_raw, dapi = img[0], img[1], img[2]

        ch1_thresh = apply_threshold(ch1_raw, ref_ch1_threshold)
        ch2_thresh = apply_threshold(ch2_raw, ref_ch2_threshold)

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
            "Condition": condition,
            "Pearson Correlation": pearson_corr,
            "Percent Coloc Nucleus": percent_coloc_nucleus,
            "Percent Coloc Cytoplasm": percent_coloc_cyto,
            "Ch1 Threshold Used": ref_ch1_threshold,
            "Ch2 Threshold Used": ref_ch2_threshold,
            "Normalization Reference": "Session-Specific HPG" if session in ref_thresholds else "Global Average HPG"
        }
    except:
        return {
            "MAX_NUMBER": session,
            "File Basename": filename,
            "Condition": condition,
            "Pearson Correlation": "NA",
            "Percent Coloc Nucleus": "NA",
            "Percent Coloc Cytoplasm": "NA",
            "Ch1 Threshold Used": "NA",
            "Ch2 Threshold Used": "NA",
            "Normalization Reference": "NA"
        }

def process_all_sessions(session_files):

    control_files = {session: files["hpg only"] for session, files in session_files.items() if "hpg only" in files}

    # Get threshold references
    ref_thresholds, avg_ch1_threshold, avg_ch2_threshold = get_hpg_thresholds(control_files)

    results = []

    # Process images session by session
    for session, files in session_files.items():
        logging.info(f'Starting Analysis on session: {session}')
        for condition, file_list in files.items():

            for file in file_list:
                result = process_image(file, condition, ref_thresholds, (avg_ch1_threshold, avg_ch2_threshold))
                results.append(result)
        logging.info(f'Finished Analysis on session: {session}')

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("colocalization_results.csv", index=False)
    logging.info("Saved results to colocalization_results.csv")
    print("Colocalization analysis completed.")

# Define and parse the dataset
file_path = "Image_files.txt"
session_files = parse_file_structure(file_path)

# Process all images
process_all_sessions(session_files)

