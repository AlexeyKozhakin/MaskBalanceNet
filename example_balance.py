# Пример использования
from genclassbalancer.data_loader import load_images, load_masks
from genclassbalancer.balance import balance_dataset
from genclassbalancer.class_colors import class_colors_hessinheim, class_colors_stpls3d
from genclassbalancer.icbo import evaluate_class_imbalance
from genclassbalancer.utils import calculate_class_distribution

from genclassbalancer.class_colors import class_colors_hessinheim,class_colors_test_2  # Импортируем словарь с цветами классов

#class_colors = class_colors_test_2
#class_colors = class_colors_hessinheim
class_colors = class_colors_stpls3d
# 1. Загрузка масок и изображений
mask_dir = (r'C:\Users\alexe\PycharmProjects'
                 r'\lidar-to-cnn-preprocessing\data\data_for_training\data_training_stpl3d\train\segment')
#image_dir_anch  = r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing\data\data_for_training\data_training_stpl3d\train\original'
# mask_dir = (r'C:\Users\alexe\PycharmProjects'
#             r'\lidar-to-cnn-preprocessing\data'
#             r'\data_for_training\data_training_hessingeim\train\segment')
# image_dir = (r'C:\Users\alexe\PycharmProjects'
#              r'\lidar-to-cnn-preprocessing\data'
#              r'\data_for_training\data_training_hessingeim\train\original')
#mask_dir =  r'data\data_test_4'
output_dir = './output_balanced_dataset'
# augm
mask_dir_augm = ''
image_dir_augm = ''
masks = load_images(mask_dir)
print(masks)
# images = load_images(image_dir)
#
# # 2. Список классов, которые нужно сбалансировать
# class_ids = [0, 1, 2, 4, 15, 18]  # Например, классы 0 - Низкая растительность, 2 - Транспортное средство, 4 - Крыша
#
# # Балансировка датасета и сохранение результата
# balanced_masks, balanced_images = balance_dataset(masks, images, class_ids, output_dir, class_colors_stpls3d)
class_distribution = calculate_class_distribution(masks, class_colors,'stat_img.csv')
# print(class_distribution)
# nb = evaluate_class_imbalance(class_distribution)
# print(nb)