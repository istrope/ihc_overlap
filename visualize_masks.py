import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import re
import logging
from skimage import filters, morphology

# Configure logging
logging.basicConfig(
    filename="colocalization_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Normalize image intensity to 0-1 range
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Create a mitochondria mask by filtering out nuclear (DAPI) regions
def create_mito_mask(dapi_channel):
    dapi_threshold = filters.threshold_otsu(dapi_channel)
    nucleus_mask = dapi_channel > dapi_threshold  # DAPI-positive areas

    # Invert nucleus mask to create a mitochondria mask
    mito_mask = ~nucleus_mask  # Everything *except* the nucleus is mitochondria/cytoplasm

    # Optional: Remove small holes to clean up the mask
    mito_mask = morphology.remove_small_holes(mito_mask, area_threshold=50)

    return mito_mask

# Function to visualize the mitochondria mask applied to Channel 1
def plot_mito_mask_on_ch1(file_path, ch1, dapi, mito_mask):
    plot_file = re.sub('.tif', '_mito_signal.png', file_path)

    # Apply the mitochondria mask to Channel 1 (HPG-Azide 494)
    ch1_mito = ch1 * mito_mask  

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(ch1, cmap='Reds')
    axes[0].set_title("Raw HPG-Azide 494 (Protein Synthesis)")

    axes[1].imshow(dapi, cmap='Blues')
    axes[1].set_title("Raw DAPI (Nucleus)")

    axes[2].imshow(mito_mask, cmap='gray')
    axes[2].set_title("Mitochondria Mask (Non-DAPI Regions)")

    axes[3].imshow(ch1_mito, cmap='Reds')
    axes[3].set_title("HPG Signal in Mitochondria Only")

    plt.savefig(plot_file)
    plt.close()
    logging.info(f"Saved mitochondria-masked signal visualization: {plot_file}")

# Process images and generate mitochondria-masked signal
def process_image_with_mito_mask(file_path):
    logging.info(f"Processing image for mitochondria-masked signal: {file_path}")

    img = tiff.imread(file_path)
    
    # Extract fluorescence channels
    ch1 = img[0]  # HPG-Azide 494 (Red - Protein synthesis)
    dapi = img[2] # DAPI (Blue - Nucleus)

    # Normalize
    ch1_norm = normalize_image(ch1)
    dapi_norm = normalize_image(dapi)

    # Create mitochondria mask (non-DAPI areas)
    mito_mask = create_mito_mask(dapi_norm)

    # Save visualization of HPG-Azide 494 in mitochondria only
    plot_mito_mask_on_ch1(file_path, ch1_norm, dapi_norm, mito_mask)

# Batch process images to generate mitochondria-masked signal plots
def process_batch_mito_masks(image_files):
    for file in image_files:
        process_image_with_mito_mask(file)
    logging.info("Mitochondria-masked signal visualization for all images completed.")

image_files = ["Test/MAX_29-no hpg.tif", 
               "Test/MAX_6-HPG only.tif", 
               "Test/MAX_26-hpg + both inihibitors.tif",
               "Test/MAX_19- hpg + mitoribosome inhibitor.tif", 
               "Test/MAX_13-hpg + cytoribosome inhibitor.tif"]

process_batch_mito_masks(image_files)

# Example usage





