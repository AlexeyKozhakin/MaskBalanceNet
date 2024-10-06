# utils.py
import numpy as np
from .class_colors import class_colors_hessinheim  # Импортируем словарь с цветами классов

def calculate_class_distribution(masks, num_classes, class_colors):
    """
    Считает распределение классов по маскам.
    
    Parameters:
    masks (list of arrays): Список масок (каждая маска - это изображение с индексами классов)
    num_classes (int): Общее количество классов
    class_colors (dict): Словарь цветов классов {класс: (R, G, B)}
    
    Returns:
    dict: Распределение классов {класс: количество пикселей для этого класса}
    """
    class_counts = np.zeros(num_classes, dtype=int)

    for mask in masks:
        # Превращаем маску в одноцветные изображения (RGB) для подсчета классов
        for class_id, color in class_colors.items():
            class_counts[class_id] += np.sum(np.all(mask == color, axis=-1))

    distribution = {i: class_counts[i] for i in range(num_classes)}
    return distribution
