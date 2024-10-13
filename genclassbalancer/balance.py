import os
import cv2
from .utils import calculate_class_distribution
from .genetic import genetic_algorithm

def balance_dataset(masks, images, class_ids, output_dir, class_colors):
    """
    Балансировка датасета изображений по классам.

    Parameters:
    masks (list): Список масок изображений
    images (list): Список изображений
    class_ids (list): Список идентификаторов классов, которые нужно сбалансировать
    output_dir (str): Директория для сохранения сбалансированных изображений
    class_colors (dict): Словарь цветов классов

    Returns:
    list: Сбалансированные маски и изображения
    """
    # Считаем распределение классов по маскам
    class_distribution_matrix = calculate_class_distribution(masks, class_colors)

    # Уменьшаем матрицу по выбранным классам
    reduced_matrix = class_distribution_matrix[class_ids]

    # Запускаем генетический алгоритм
    best_solution = genetic_algorithm(reduced_matrix)

    # Отбор сбалансированных изображений
    balanced_masks = [masks[i] for i in range(len(masks)) if best_solution[i]]
    balanced_images = [images[i] for i in range(len(images)) if best_solution[i]]

    # Создание подкаталогов для сохранения результатов
    segment_dir = os.path.join(output_dir, 'segment')
    original_dir = os.path.join(output_dir, 'original')

    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)

    if not os.path.exists(original_dir):
        os.makedirs(original_dir)

    # Сохранение масок
    for idx, mask in enumerate(balanced_masks):
        cv2.imwrite(os.path.join(segment_dir, f"balanced_mask_{idx}.png"), mask)

    # Сохранение изображений
    for idx, image in enumerate(balanced_images):
        cv2.imwrite(os.path.join(original_dir, f"balanced_image_{idx}.png"), image)

    return balanced_masks, balanced_images
