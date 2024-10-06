# MaskBalanceNet

```
# README пример использования

from genclassbalancer.data_loader import load_masks
from genclassbalancer.utils import calculate_class_distribution
from genclassbalancer.visualizer import plot_class_distribution, save_class_distribution_table

# 1. Загрузка масок
mask_dir = './path_to_masks'
masks = load_masks(mask_dir)

# 2. Расчёт распределения классов
num_classes = 5  # Укажите количество классов, присутствующих в масках
class_distribution = calculate_class_distribution(masks, num_classes)

# 3. Визуализация и сохранение результатов
output_dir = './output'
plot_class_distribution(class_distribution, f'{output_dir}/class_distribution.png')
save_class_distribution_table(class_distribution, f'{output_dir}/class_distribution.csv')
```