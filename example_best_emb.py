from genclassbalancer.icbo import reduce_dataset_to_balance
import pandas as pd
# Загрузка CSV файла
df = pd.read_csv('stat_img.csv')

# Определение целевых параметров
target_size = 10  # Целевой размер датасета
target_imbalance = 0.05  # Целевой уровень дисбаланса (например, 5%)

# Запуск оптимизации
optimized_df = reduce_dataset_to_balance(df,
                                         target_size=target_size,
                                         target_imbalance=target_imbalance,
                                         removal_fraction=0.05,
                                         balance_classes=['class_0', 'class_1',
                                                          'class_2', 'class_4',
                                                          'class_15', 'class_18'])

# Сохранение оптимизированного датасета
optimized_df.to_csv('optimized_dataset.csv', index=False)
