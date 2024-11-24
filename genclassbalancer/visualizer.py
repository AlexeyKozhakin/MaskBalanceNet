# visualizer.py
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_class_distribution(distribution, output_path=None):
    """
    Строит график распределения классов и сохраняет его как изображение.
    
    Parameters:
    distribution (dict): Распределение классов {класс: количество пикселей}
    output_path (str): Путь для сохранения графика (если None, график просто отображается)
    """
    classes = list(distribution.keys())
    counts = list(distribution.values())

    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Pixel Count')
    plt.title('Class Distribution')
    # Добавляем метки на оси X
    plt.xticks(ticks=classes, labels=[f'{ind}' for ind, cls in enumerate(classes)])

    if output_path:
        # Проверка, существует ли директория, и создание её, если нет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        print(f"График сохранён как {output_path}")
    else:
        plt.show()

def save_class_distribution_table(distribution, output_path):
    """
    Сохраняет распределение классов в виде таблицы (CSV).
    
    Parameters:
    distribution (dict): Распределение классов {класс: количество пикселей}
    output_path (str): Путь для сохранения таблицы
    """
    # Проверка, существует ли директория, и создание её, если нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(list(distribution.items()), columns=['Class', 'Pixel Count'])
    df.to_csv(output_path, index=False)
    print(f"Таблица сохранена как {output_path}")
