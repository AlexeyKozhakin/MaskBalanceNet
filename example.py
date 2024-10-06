# README пример использования

from genclassbalancer.data_loader import load_masks
from genclassbalancer.data_loader import load_masks
from genclassbalancer.utils import calculate_class_distribution
from genclassbalancer.visualizer import plot_class_distribution, save_class_distribution_table
from genclassbalancer.class_colors import class_colors_hessinheim  # Импортируем словарь с цветами классов

# 1. Загрузка масок
mask_dir = r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing\data\data_for_training\data_training_toronto3d\train\segment'
masks = load_masks(mask_dir)

# 2. Расчёт распределения классов
num_classes = len(class_colors_hessinheim)  # Указываем количество классов на основе длины словаря
class_distribution = calculate_class_distribution(masks, num_classes, class_colors_hessinheim)

# 3. Визуализация и сохранение результатов
output_dir = './output'
plot_class_distribution(class_distribution, f'{output_dir}/class_distribution.png')
save_class_distribution_table(class_distribution, f'{output_dir}/class_distribution.csv')

