import os
import json
import shutil
from pathlib import Path

def load_config(config_path):
    """Загружает конфигурацию из JSON файла."""
    with open(config_path, 'r') as f:
        return json.load(f)

def ensure_folder_exists(folder_path):
    """Создает каталог, если он не существует."""
    os.makedirs(folder_path, exist_ok=True)

def copy_files(config):
    """Копирует файлы из input_folder в output_folder согласно конфигурации."""
    input_folder = Path(config["input_folder"])
    output_folder = Path(config["output_folder"])
    augmentation_targets = config["augmentation_targets"]

    # Создаем выходные каталоги segment и original
    segment_output = output_folder / "segment"
    original_output = output_folder / "original"
    ensure_folder_exists(segment_output)
    ensure_folder_exists(original_output)

    for file_name in augmentation_targets.keys():
        # Пути к файлам в исходных папках
        segment_file = input_folder / "segment" / file_name
        original_file = input_folder / "original" / file_name

        # Проверяем, что файлы существуют
        if not segment_file.exists():
            print(f"Файл {segment_file} не найден. Пропускаем.")
            continue
        if not original_file.exists():
            print(f"Файл {original_file} не найден. Пропускаем.")
            continue

        # Копируем файлы в выходные папки
        shutil.copy(segment_file, segment_output / file_name)
        shutil.copy(original_file, original_output / file_name)
        print(f"Файл {file_name} успешно скопирован.")

def main():
    # Путь к файлу конфигурации
    config_path = "config_balance.json"

    # Загружаем конфигурацию
    config = load_config(config_path)

    # Выполняем копирование
    copy_files(config)

if __name__ == "__main__":
    main()
