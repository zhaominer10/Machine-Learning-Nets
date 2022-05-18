import os
import random
from shutil import copy, rmtree


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    random.seed(0)
    split_rate = 0.1

    cwd = os.getcwd()
    original_flower_path = os.path.join(cwd, 'flower_photos')
    assert os.path.exists(original_flower_path), "path '{}' dose not exists.".format(original_flower_path)

    flower_class = [ cla for cla in os.listdir(original_flower_path) if
                     os.path.isdir(os.path.join(original_flower_path, cla)) ]

    # 训练集文件路径
    train_root = os.path.join(cwd, 'train')
    mk_file(train_root)
    for cla in flower_class:
        mk_file(os.path.join(train_root, cla))

    # 验证集文件路径
    val_root = os.path.join(cwd, 'val')
    mk_file(val_root)
    for cla in flower_class:
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(original_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        val_samples = random.sample(images, k=int(num * split_rate))
        for index, image in enumerate(images):
            if image in val_samples:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
