from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = os.path.join(data_root, "dataset", "flower_data")
train_dir = os.path.join(image_path, "train")
validation_dir = os.path.join(image_path, "val")
assert os.path.exists(train_dir), "can not find {}".format(train_dir)
assert os.path.exists(validation_dir), "can not find {}".format(validation_dir)

if not os.path.exists("weights"):
    os.makedirs("weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 10

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode="categorical")

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')

# number of training samples
# number of validation samples
total_train = train_data_gen.n
total_val = val_data_gen.n
print("using {} images for training, {} images for validation.".format(total_train, total_val))

# get class dict
class_indices = train_data_gen.class_indices
print(class_indices)

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# show training image examples
sample_train_images, sample

