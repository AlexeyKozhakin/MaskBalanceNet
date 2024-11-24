# data_loader.py
import os
import numpy as np
from PIL import Image


def load_masks_dict(mask_dir):
    """
    Загружает маски из директории и возвращает словарь с названиями файлов и масками.

    Parameters:
    mask_dir (str): Путь к директории с масками

    Returns:
    dict: Словарь, где ключ - это название файла, а значение - маска в виде numpy массива
    """
    masks_dict = {}
    for filename in os.listdir(mask_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            mask_path = os.path.join(mask_dir, filename)
            mask = Image.open(mask_path)
            masks_dict[filename] = np.array(mask)

    return masks_dict

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
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            mask_path = os.path.join(mask_dir, filename)
            mask = Image.open(mask_path)
            mask = np.array(mask)
            masks.append(mask)
    
    return masks

def load_images(image_dir):
    """
    Загружает изображения из заданного каталога.
    
    Parameters:
    image_dir (str): Путь к каталогу с изображениями
    
    Returns:
    list of np.ndarray: Список изображений
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = np.array(Image.open(img_path))
        images.append(img)
    dic_img = {image_file:image for image_file, image in zip(image_files, images)}
    return dic_img



