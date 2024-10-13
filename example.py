# Использования - начальное распределение

from genclassbalancer.data_loader import load_masks
from genclassbalancer.data_loader import load_masks
from genclassbalancer.utils import calculate_class_distribution_stat, calculate_overall_class_distribution
from genclassbalancer.visualizer import plot_class_distribution, save_class_distribution_table
from genclassbalancer.class_colors import class_colors_hessinheim, class_colors_test_1, class_colors_test_2  # Импортируем словарь с цветами классов
from genclassbalancer.class_colors import class_colors_toronto3D  # Импортируем словарь с цветами классов
from genclassbalancer.class_colors import class_colors_stpls3d  # Импортируем словарь с цветами классов
import pandas as pd

#class_colors = class_colors_stpls3d
class_colors = class_colors_test_2
# 1. Загрузка масок
#mask_dir = r'D:\data\data_for_training\data_training_stpl3d\train\segment'
#mask_dir = r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing\data\data_for_training\data_training_hessingeim\train\segment'
#mask_dir = "./output_balanced_dataset/segment"
mask_dir = r'data\data_test_2'
masks = load_masks(mask_dir)
print(masks)
# 2. Расчёт распределения классов
num_classes = len(class_colors)  # Указываем количество классов на основе длины словаря
#class_distribution = calculate_class_distribution_stat(masks, class_colors)
df = pd.read_csv('optimized_dataset.csv')
class_distribution = calculate_overall_class_distribution(df)

print(class_distribution)

# 3. Визуализация и сохранение результатов
output_dir = './output'
plot_class_distribution(class_distribution, f'{output_dir}/class_distribution.png')
save_class_distribution_table(class_distribution, f'{output_dir}/class_distribution.csv')

