import pandas as pd
import numpy as np
from genclassbalancer.icbo import hellinger_distance, evaluate_class_imbalance



def calculate_hellinger_distance(input_csv_path, output_csv_path):
    """
    Загружает CSV файл, вычисляет распределение классов и рассчитывает расстояние Хеллингера.

    Parameters:
    input_csv_path (str): Путь к входному CSV файлу
    output_csv_path (str): Путь для сохранения результата в CSV файл
    """
    # Загружаем данные из CSV
    df = pd.read_csv(input_csv_path)

    # Извлекаем названия классов (все столбцы, кроме первого)
    class_columns = df.columns[1:]

    # Считаем общее распределение по всем классам
    total_distribution = df[class_columns].sum(axis=0)

    # Приводим к вероятностному распределению
    total_distribution_prob = total_distribution / total_distribution.sum()

    # Список для хранения результатов
    distances = []

    # Рассчитываем расстояние Хеллингера для каждой строки
    for index, row in df.iterrows():
        # Извлекаем распределение для данной строки
        row_distribution = row[class_columns]

        # Приводим к вероятностному распределению
        row_distribution_prob = row_distribution / row_distribution.sum()

        # Вычисляем расстояние Хеллингера
        distance = hellinger_distance(row_distribution_prob, total_distribution_prob)
        distances.append(distance)

    # Добавляем результаты в DataFrame
    df['hellinger_distance'] = distances

    # Сохраняем результаты в новый CSV файл
    df.to_csv(output_csv_path, index=False)

    print(f"Расстояния Хеллингера записаны в {output_csv_path}")

def process_csv_and_calculate_distances(input_csv_path):
    """
    Читает CSV файл, сортирует строки по hellinger_distance,
    а затем последовательно добавляет изображения, вычисляя расстояние Хеллингера.

    Parameters:
    input_csv_path (str): Путь к входному CSV файлу
    """
    # Загружаем данные из CSV
    df = pd.read_csv(input_csv_path)

    # Сортируем DataFrame по столбцу 'hellinger_distance'
    df_sorted = df.sort_values(by='hellinger_distance')

    # Список для хранения расстояний Хеллингера для каждой итерации
    hellinger_distances = []

    # Инициализируем переменную для хранения текущего распределения классов
    current_distribution = np.zeros(len(df_sorted.columns) - 1)  # Избегаем первого столбца (названия файлов)

    # Обработка строк
    for index, row in df_sorted.iterrows():
        # Извлекаем распределение классов для текущего изображения
        current_distribution += row[1:]  # Учитываем все столбцы, кроме первого (названия файлов)

        # Приводим к вероятностному распределению
        current_distribution_prob = current_distribution / current_distribution.sum()

        # Вычисляем расстояние Хеллингера от равномерного распределения
        distance = evaluate_class_imbalance(current_distribution_prob)
        hellinger_distances.append(distance)

        # Добавляем рассчитанное расстояние в DataFrame
        df_sorted.at[index, 'combined_hellinger_distance'] = distance

    # Сохраняем результаты в новый CSV файл
    output_csv_path = "sorted_with_combined_distances.csv"
    df_sorted.to_csv(output_csv_path, index=False)

    print(f"Результаты сохранены в {output_csv_path}")

#calculate_hellinger_distance('stat_img.csv', 'stat_img_hellinger.csv')
process_csv_and_calculate_distances('stat_img_hellinger.csv')