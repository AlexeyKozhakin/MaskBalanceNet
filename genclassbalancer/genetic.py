import numpy as np
import random


def fitness(solution, class_distribution_matrix):
    """
    Функция фитнеса для оценки сбалансированности подвыборки.

    Parameters:
    solution (list of bool): Массив выбора картинок (True если картинка выбрана, False если нет)
    class_distribution_matrix (np.ndarray): Матрица распределения классов по маскам

    Returns:
    float: Оценка фитнеса (чем меньше значение, тем лучше)
    """
    selected_columns = class_distribution_matrix[:, solution]
    actual_distribution = np.sum(selected_columns, axis=1)
    return np.sum(np.abs(np.max(actual_distribution) - np.min(actual_distribution)))


def select_parents(population, fitness_scores, num_parents):
    """Выбор родителей для генерации следующего поколения."""
    parents_idx = np.argsort(fitness_scores)[:num_parents]
    return [population[i] for i in parents_idx]


def crossover(parent1, parent2):
    """Операция кроссовера между двумя родителями."""
    crossover_point = random.randint(0, len(parent1) - 1)
    return np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])


def mutate(solution, mutation_rate=0.01):
    """Мутация решения с заданной вероятностью."""
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = not solution[i]
    return solution


def genetic_algorithm(class_distribution_matrix, population_size=100, num_generations=200, mutation_rate=0.01):
    """
    Реализация генетического алгоритма для отбора сбалансированного подмножества картинок.

    Parameters:
    class_distribution_matrix (np.ndarray): Матрица распределения классов по маскам

    Returns:
    list: Лучшее найденное решение
    """
    num_masks = class_distribution_matrix.shape[1]

    # Инициализация популяции
    population = [np.random.choice([True, False], size=num_masks) for _ in range(population_size)]

    for generation in range(num_generations):
        fitness_scores = [fitness(ind, class_distribution_matrix) for ind in population]

        # Выбор родителей
        parents = select_parents(population, fitness_scores, num_parents=population_size // 2)

        # Генерация следующего поколения
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1 if i + 1 < len(parents) else 0]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        population = parents + next_generation

        # Оценка фитнеса
        best_fitness = min(fitness_scores)
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

    # Возвращаем лучшее решение
    best_solution_idx = np.argmin([fitness(ind, class_distribution_matrix) for ind in population])
    return population[best_solution_idx]
