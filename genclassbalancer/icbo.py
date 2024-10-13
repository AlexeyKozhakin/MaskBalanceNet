import numpy as np


def hellinger_distance(p, q):
    # Проверка нормализации
    sm = [(pi**0.5-qi**0.5)**2 for pi, qi in zip(p,q)]
    d = (sum(sm)/2)**(0.5)
    return d



def evaluate_class_imbalance(current_distribution):
    """
    Оценка дисбаланса классов в маске с использованием дистанции Хеллингера.

    Parameters:
    mask (numpy.ndarray): Маска (изображение с классами).
    class_colors (dict): Словарь цветов классов.

    Returns:
    float: Дистанция Хеллингера между текущим распределением и равномерным распределением.
    """
    if isinstance(current_distribution, dict):
        num_classes = len(current_distribution.values())
        uniform_distribution = {key: 1 / num_classes
                                for key in current_distribution.keys()}
        cd = list(current_distribution.values())
    else:
        num_classes = len(current_distribution)
        uniform_distribution = {key: 1 / num_classes
                                for key in current_distribution}
        cd = list(current_distribution)

    # Целевое равномерное распределение


    # Вычисляем дистанцию Хеллингера
    return hellinger_distance(cd, list(uniform_distribution.values()))



def reduce_dataset_to_balance(df, target_size=None, target_imbalance=None,
                              removal_fraction=0.0, balance_classes=None):
    """
    Уменьшает датасет, удаляя строки, чтобы минимизировать дисбаланс по указанным классам.

    Parameters:
    df (pd.DataFrame): DataFrame с первым столбцом — названиями файлов, и остальными — содержанием классов.
    evaluate_class_imbalance (function): Функция для оценки дисбаланса (например, расстояние Хеллингера).
    target_size (int, optional): Конечный размер датасета.
    target_imbalance (float, optional): Целевое значение метрики дисбаланса.
    removal_fraction (float, optional): Доля строк для удаления за один шаг (0 - удалять одну строку, по умолчанию 0.0).
    balance_classes (list, optional): Список классов, по которым будет производиться балансировка.

    Returns:
    pd.DataFrame: Оптимизированный датасет с минимальным дисбалансом.
    """

    # Проверяем, что target_size и target_imbalance имеют значение
    if target_size is None and target_imbalance is None:
        raise ValueError("Необходимо указать хотя бы одно из условий завершения: target_size или target_imbalance.")

    # Если balance_classes не указан, используем все классы кроме первого столбца (название файла)
    if balance_classes is None:
        balance_classes = df.columns[1:]  # Используем все столбцы, кроме названия файла

    current_size = df.shape[0]

    # Цикл продолжается, пока датасет больше целевого размера или дисбаланс больше целевого значения
    while current_size > target_size or evaluate_class_imbalance(
            df[balance_classes].sum() / df[balance_classes].sum().sum()) > target_imbalance:
        imbalance_scores = []

        # Внешний цикл проходит по каждой строке
        for idx in df.index:
            # Создаем временный DataFrame без текущей строки
            temp_df = df.drop(index=idx)

            # Рассчитываем суммарное распределение классов для оставшихся строк
            total_class_distribution = temp_df[balance_classes].sum()

            # Нормализуем распределение (относительное распределение по каждому классу)
            relative_distribution = total_class_distribution / total_class_distribution.sum()

            # Вычисляем метрику дисбаланса на основе относительного распределения
            imbalance_score = evaluate_class_imbalance(relative_distribution)

            # Сохраняем индекс строки и её метрику дисбаланса
            imbalance_scores.append((idx, imbalance_score))

        # Сортируем строки по возрастанию метрики дисбаланса
        imbalance_scores.sort(key=lambda x: x[1])

        # Рассчитываем количество строк для удаления
        if removal_fraction > 0:
            num_to_remove = max(1, int(current_size * removal_fraction))
        else:
            num_to_remove = 1  # Если не указан процент, удаляем одну строку

        # Удаляем строки с минимальными метриками дисбаланса
        rows_to_remove = [item[0] for item in imbalance_scores[:num_to_remove]]

        # Удаляем строки из DataFrame
        df = df.drop(index=rows_to_remove)
        current_size -= num_to_remove

        print(f"Удалено строк: {num_to_remove}, текущий размер: {current_size}, "
              f"метрика баланса: {evaluate_class_imbalance(df[balance_classes].sum() / df[balance_classes].sum().sum())}")

        # Проверяем условия выхода из цикла
        if target_size is not None and current_size <= target_size:
            break
        if target_imbalance is not None and imbalance_scores[0][1] <= target_imbalance:
            break

    return df


