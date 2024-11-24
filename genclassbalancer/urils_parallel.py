import numpy as np
import pandas as pd
from multiprocessing import Pool
from genclassbalancer.class_colors import class_colors_stpls3d
from genclassbalancer.data_loader import load_masks_dict

import numpy as np
import pandas as pd
from multiprocessing import Pool

def process_mask(mask_item, class_colors):
    """
    Обрабатывает одну маску для вычисления количества пикселей каждого класса.

    Parameters:
    mask_item (tuple): Кортеж (имя_файла, маска)
    class_colors (dict): Словарь цветов классов {класс: (R, G, B)}

    Returns:
    list: Список с именем файла и количеством пикселей для каждого класса.
    """
    file_name, mask = mask_item
    class_pixel_counts = [np.sum(np.all(mask == color, axis=-1)) for color in class_colors.values()]
    return [file_name] + class_pixel_counts

def calculate_class_distribution_parallel(masks_dict, class_colors, output_csv_path, num_workers = 1):
    """
    Считает распределение классов по маскам и сохраняет результаты в CSV файл параллельно.

    Parameters:
    masks_dict (dict): Словарь, где ключ - это имя файла, а значение - маска (RGB-изображение)
    class_colors (dict): Словарь цветов классов {класс: (R, G, B)}
    output_csv_path (str): Путь для сохранения CSV файла

    Returns:
    np.ndarray: Матрица, где строки - классы, столбцы - маски, а значения - количество пикселей данного класса.
    """
    # Преобразуем словарь в список для передачи в Pool
    masks_list = list(masks_dict.items())

    # Параллельная обработка масок
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_mask, [(mask_item, class_colors) for mask_item in masks_list])

    # Создаем DataFrame с названиями классов
    column_names = ['file_name'] + [f'class_{class_id}' for class_id in class_colors.keys()]
    df = pd.DataFrame(results, columns=column_names)

    # Сохраняем DataFrame в CSV файл
    df.to_csv(output_csv_path, index=False)

    # Создаем матрицу распределения классов
    distribution_matrix = df.iloc[:, 1:].values.T  # Преобразуем DataFrame в np.ndarray

    return distribution_matrix



if __name__ == '__main__':
    import time
    # Путь к выходному CSV файлу
    output_csv_path = 'class_distribution.csv'
    class_colors = class_colors_stpls3d
    mask_dir = r'D:\data\data_for_training\data_training_stpl3d_64_512\train\segment'
    masks_dict = load_masks_dict(mask_dir)
    # Запускаем параллельную функцию для подсчета распределения классов
    start = time.time()
    distribution_matrix = calculate_class_distribution_parallel(masks_dict,
                                                                class_colors,
                                                                output_csv_path,
                                                                num_workers=7)
    end = time.time()
    print(round(end-start))