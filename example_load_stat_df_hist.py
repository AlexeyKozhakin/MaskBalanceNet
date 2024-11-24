# Использования - начальное распределение
from genclassbalancer.data_loader import load_masks_dict
from genclassbalancer.utils import calculate_class_distribution_stat_file_parallel, calculate_overall_class_distribution
from genclassbalancer.visualizer import plot_class_distribution, save_class_distribution_table
from genclassbalancer.class_colors import class_colors_stpls3d  # Импортируем словарь с цветами классов
import json
import time


if __name__ == '__main__':
    # Чтение конфигурации из JSON
    with open("config.json", "r") as f:
        config = json.load(f)

    # Использование переменных
    mask_dir = config["MASK_DIR"]

    class_colors = class_colors_stpls3d
    # 1. Загрузка масок
    masks_dict = load_masks_dict(mask_dir)
    # 2. Расчёт распределения классов
    num_classes = len(class_colors)  # Указываем количество классов на основе длины словаря
    start = time.time()
    df = calculate_class_distribution_stat_file_parallel(masks_dict, class_colors, num_processes=7, output_csv='stat_mask.csv')
    end = time.time()
    time_processing = end - start
    print(f'Time processing {round(time_processing)}s')
    class_distribution = calculate_overall_class_distribution(df)

    print(class_distribution)

    # 3. Визуализация и сохранение результатов
    output_dir = './output'
    plot_class_distribution(class_distribution, f'{output_dir}/class_distribution.png')
    save_class_distribution_table(class_distribution, f'{output_dir}/class_distribution.csv')

