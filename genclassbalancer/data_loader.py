# data_loader.py
import os
import numpy as np
from PIL import Image

def load_masks(mask_dir):
    """
    Загружает маски из директории.
    
    Parameters:
    mask_dir (str): Путь к директории с масками
    
    Returns:
    list of np.array: Список масок в виде numpy массивов
    """
    masks = []
    for filename in os.listdir(mask_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            mask_path = os.path.join(mask_dir, filename)
            mask = Image.open(mask_path)
            mask = np.array(mask)
            masks.append(mask)
    
    return masks
  
