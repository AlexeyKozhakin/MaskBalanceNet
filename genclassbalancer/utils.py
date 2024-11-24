# utils.py
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np



def process_mask(args):
    """
    Функция для обработки одной маски. Вызывается в параллельных процессах.

    Parameters:
    args (tuple): Кортеж из имени файла, маски и цветов классов.

    Returns:
    list: Статистика для одного файла в формате [имя файла, кол-во пикселей для каждого класса].
    """
    filename, mask, class_colors = args
    num_classes = len(class_colors)
    class_counts = np.zeros(num_classes, dtype=int)

    # Считаем количество пикселей для каждого класса
    c_id = 0
    for class_id, color in class_colors.items():
        class_counts[c_id] += np.sum(np.all(mask == color, axis=-1))
        c_id += 1

    return [filename] + class_counts.tolist()


def calculate_class_distribution_stat_file_parallel(masks_dict, class_colors, num_processes=1, output_csv='stat_mask.csv'):
    """
    Параллельное вычисление распределения классов по маскам и сохранение статистики в CSV файл.

    Parameters:
    masks_dict (dict): Словарь, где ключ - название файла, значение - маска в виде numpy массива
    class_colors (dict): Словарь цветов классов {класс: (R, G, B)}
    output_csv (str): Путь к выходному CSV-файлу, по умолчанию 'stat_mask.csv'

    Returns:
    pd.DataFrame: DataFrame с распределением классов для каждой маски
    """
    # Подготовка данных для параллельной обработки
    tasks = [(filename, mask, class_colors) for filename, mask in masks_dict.items()]


    # Параллельная обработка масок
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_mask, tasks)

    # Создаем DataFrame
    num_classes = len(class_colors)
    columns = ['filename'] + [f'class_{i}' for i in range(num_classes)]
    df = pd.DataFrame(results, columns=columns)

    # Сохраняем DataFrame в CSV
    df.to_csv(output_csv, index=False)
    print(f"Статистика сохранена в файл: {output_csv}")

    return df

def calculate_class_distribution_stat_file(masks_dict, class_colors, output_csv='stat_mask.csv'):
    """
    Считает распределение классов по маскам и сохраняет статистику в CSV файл.

    Parameters:
    masks_dict (dict): Словарь, где ключ - название файла, значение - маска в виде numpy массива
    class_colors (dict): Словарь цветов классов {класс: (R, G, B)}
    output_csv (str): Путь к выходному CSV-файлу, по умолчанию 'stat_mask.csv'

    Returns:
    pd.DataFrame: DataFrame с распределением классов для каждой маски
    """
    num_classes = len(class_colors)
    stats = []

    for filename, mask in masks_dict.items():
        class_counts = np.zeros(num_classes, dtype=int)

        # Считаем количество пикселей для каждого класса
        c_id = 0
        for class_id, color in class_colors.items():
            class_counts[c_id] += np.sum(np.all(mask == color, axis=-1))
            c_id+=1

        # Сохраняем статистику в формате: [имя файла, кол-во пикселей для каждого класса]
        stats.append([filename] + class_counts.tolist())

    # Создаем DataFrame
    columns = ['filename'] + [f'class_{i}' for i in range(num_classes)]
    df = pd.DataFrame(stats, columns=columns)

    # Сохраняем DataFrame в CSV
    df.to_csv(output_csv, index=False)
    print(f"Статистика сохранена в файл: {output_csv}")

    return df

def calculate_class_distribution(masks_dict, class_colors, output_csv_path):
    """
    Считает распределение классов по маскам и сохраняет результаты в CSV файл.

    Parameters:
    masks_dict (dict): Словарь, где ключ - это имя файла, а значение - маска (RGB-изображение)
    class_colors (dict): Словарь цветов классов {класс: (R, G, B)}
    output_csv_path (str): Путь для сохранения CSV файла

    Returns:
    np.ndarray: Матрица, где строки - классы, столбцы - маски, а значения - количество пикселей данного класса.
    """
    num_classes = len(class_colors)
    num_masks = len(masks_dict)

    # Матрица: строки - классы, столбцы - маски
    distribution_matrix = np.zeros((num_classes, num_masks), dtype=int)
    file_names = list(masks_dict.keys())

    # Создаём список для данных, которые пойдут в CSV
    csv_data = []

    for mask_idx, (file_name, mask) in enumerate(masks_dict.items()):
        # Считаем количество пикселей для каждого класса
        class_pixel_counts = []
        for class_id, color in class_colors.items():
            # Подсчёт пикселей, которые соответствуют цвету данного класса
            pixel_count = np.sum(np.all(mask == color, axis=-1))
            distribution_matrix[class_id, mask_idx] = pixel_count
            class_pixel_counts.append(pixel_count)

        # Добавляем строку в CSV: имя файла + пиксели для каждого класса
        csv_data.append([file_name] + class_pixel_counts)

    # Создаём DataFrame с названиями классов как заголовки
    column_names = ['file_name'] + [f'class_{class_id}' for class_id in class_colors.keys()]
    df = pd.DataFrame(csv_data, columns=column_names)

    # Сохраняем DataFrame в CSV файл
    df.to_csv(output_csv_path, index=False)

    return distribution_matrix


def calculate_class_distribution_stat(masks, class_colors):
    """
    Считает распределение классов по маскам.
    
    Parameters:
    masks (list of arrays): Список масок (каждая маска - это изображение с индексами классов)
    num_classes (int): Общее количество классов
    class_colors (dict): Словарь цветов классов {класс: (R, G, B)}
    
    Returns:
    dict: Распределение классов {класс: количество пикселей для этого класса}
    """
    num_classes = len(class_colors)
    class_counts = np.zeros(num_classes, dtype=int)

    for mask in masks:
        # Превращаем маску в одноцветные изображения (RGB) для подсчета классов
        for class_id, color in class_colors.items():
            class_counts[class_id] += np.sum(np.all(mask == color, axis=-1))
    N = sum(class_counts)
    distribution = {i: class_counts[i] / N for i in range(num_classes)}
    print('sum = ',sum(distribution.values()))
    return distribution

def calculate_overall_class_distribution(df):
    """
    Считает общее распределение классов по всему датасету.

    Parameters:
    df (pd.DataFrame): DataFrame, где первый столбец — название файла, а остальные столбцы — количество пикселей для каждого класса.

    Returns:
    dict: Распределение классов для всего датасета в виде словаря {класс: процент от общего числа пикселей}.
    """
    # Суммируем количество пикселей по каждому классу (игнорируя первый столбец с названиями файлов)
    total_pixels_per_class = df.iloc[:, 1:].sum()

    # Вычисляем общее количество пикселей во всём датасете
    total_pixels = total_pixels_per_class.sum()

    # Рассчитываем долю (процент) для каждого класса
    class_distribution = {col: (total_pixels_per_class[col] / total_pixels) * 100 for col in df.columns[1:]}

    return class_distribution