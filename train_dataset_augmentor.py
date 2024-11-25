import os
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import random
import json
import cv2


# Аугментации
def original(img):
    return img


def flip_x(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def flip_y(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def rotate_90(img):
    return img.rotate(90)


def rotate_180(img):
    return img.rotate(180)


def rotate_270(img):
    return img.rotate(270)


def flip_xy(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)


import math
import cv2
import numpy as np
from PIL import Image

def rotate_image(image, angle):
    """
    Rotates an image (NumPy format) around its center and returns the rotated image.
    """
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Get rotation matrix and perform rotation
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )
    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Get coordinates of corners and calculate new bounds
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    x_coords = [pt[0] for pt in rotated_coords]
    y_coords = [pt[1] for pt in rotated_coords]

    new_w = int(abs(max(x_coords) - min(x_coords)))
    new_h = int(abs(max(y_coords) - min(y_coords)))

    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    rotated = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return rotated


def largest_rotated_rect(w, h, angle):
    """
    Finds the largest rectangle that can fit inside a rotated rectangle.
    """
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_h) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
    """
    Crops the given image to the specified width and height around its center.
    """
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rotate_and_crop(img, angle):
    """
    Rotates and crops the image to remove black triangles.
    """
    # Convert PIL image to OpenCV format
    img_np = np.array(img)
    if img_np.ndim == 2:  # Grayscale
        img_cv = img_np
    else:  # RGB
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Rotate the image
    rotated = rotate_image(img_cv, angle)

    # Calculate largest rectangle inside the rotated image
    height, width = img_cv.shape[:2]
    new_width, new_height = largest_rotated_rect(width, height, math.radians(angle))

    # Crop the image around the center
    cropped = crop_around_center(rotated, int(new_width), int(new_height))

    # Convert back to PIL format
    if img_np.ndim == 2:  # Grayscale
        return Image.fromarray(cropped)
    else:  # RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb)



def random_rotate_and_crop(img):
    angle = random.uniform(-45, 45)  # Случайный угол поворота
    return rotate_and_crop(img, angle)


# Аугментации в словаре
augmentations = {
    "random_rotate_and_crop": random_rotate_and_crop,
}

# Загружаем конфигурацию из config.json
with open('config_balance.json', 'r') as f:
    config = json.load(f)

input_folder = config["input_folder"]
output_folder = config["output_folder"]
num_workers = config["num_workers"]
augmentation_targets = config["augmentation_targets"]

# Создание папки для результата
os.makedirs(output_folder, exist_ok=True)


# Функция для обработки одного изображения
def process_image(file_name, count, input_folder, output_folder):
    original_path = os.path.join(input_folder, "original", file_name)
    mask_path = os.path.join(input_folder, "segment", file_name)

    # Убедимся, что файлы существуют
    if not os.path.exists(original_path) or not os.path.exists(mask_path):
        print(f"Файл {file_name} отсутствует в исходной папке!")
        return

    # Открываем оригинал и маску
    original = Image.open(original_path)
    mask = Image.open(mask_path)

    # Определяем количество аугментаций
    augmentations_needed = count
    augmented_count = 0
    aug_idx = 0

    while augmented_count < augmentations_needed:
        # Случайно выбираем аугментацию
        aug_name, aug_func = random.choice(list(augmentations.items()))

        # Применяем аугментацию
        aug_original = aug_func(original)
        aug_mask = aug_func(mask)

        # Сохраняем с уникальными именами
        aug_original_name = f"{Path(file_name).stem}_{aug_name}_{aug_idx}.png"
        aug_mask_name = f"{Path(file_name).stem}_{aug_name}_{aug_idx}.png"

        aug_original.save(os.path.join(output_folder, "original", aug_original_name))
        aug_mask.save(os.path.join(output_folder, "segment", aug_mask_name))

        augmented_count += 1
        aug_idx += 1


# Основной процесс
def main():
    # Создаем подпапки в train_augm
    os.makedirs(os.path.join(output_folder, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "segment"), exist_ok=True)

    # Параллельная обработка
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = [
            executor.submit(process_image, file_name, count, input_folder, output_folder)
            for file_name, count in augmentation_targets.items()
        ]
        for task in tasks:
            task.result()

    print("Все изображения успешно обработаны!")


if __name__ == "__main__":
    main()
