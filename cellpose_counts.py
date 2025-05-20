from cellpose import models, core, io, plot
import tifffile as tiff
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

cellpose_model_dapi = models.CellposeModel(pretrained_model = 'nuclei_segment/weights.pt',gpu=True)
cellpose_model_sam = models.CellposeModel(gpu=True)

files = []
counts_sam = []
counts_dapi = []
num = []

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img = img - np.min(img)
    if np.max(img) > 0:
        img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    return img

def cellpose_SAM(img):
    img = normalize_to_uint8(img)
    img_smooth = gaussian_filter(img,sigma=1.5)
    masks,flows,styles = cellpose_model_sam.eval([img_smooth],diameter=150)
    return masks,flows,styles

def cellpose_dapi(img):
    img = normalize_to_uint8(img)
    img_smooth = guassian_filter(img,sigma=5)
    masks,flows,styles = cellpose_model_dapi.eval([img_smooth],diameter=50)
    return np.max(masks)

for file in os.listdir('Batch_1/controls'):
    print(f'Processing file: {file}')
    img = tiff.imread('Batch_1/controls/' + file)
    dapi = img[2]
    masks,flows,_ = cellpose_SAM(dapi)
    
    counts_sam.append(np.max(masks[0]))
    #counts_dapi.append(cellpose_dapi(dapi))
    files.append(str(file))
    max_num = file.split('-')[0]
    max_num = int(max_num.split('_')[1])
    num.append(max_num)
    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, dapi, masks[0], flows[0][0])
    plt.tight_layout()
    plt.savefig(f'cellpose_masks/{max_num}.png')
    

df = pd.DataFrame({
    'file':files,
    #'nuclei_dapi':counts_dapi,
    'nuclei_sam':counts_sam,
    'MAX_NUM':num
    })
df_sorted = df.sort_values(by='MAX_NUM')
df_sorted.to_csv('cell_counts_controls.csv')    

