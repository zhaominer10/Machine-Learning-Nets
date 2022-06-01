import os
import sys
import json

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

from model_bn import GoogLeNet


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "dataset", "flower_data")
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("weights"):
        os.makedirs("weights")

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 30

    def pre_function(img: np.ndarray):
        img = img / 255.
        img = img - [ 0.485, 0.456, 0.406 ]
        img = img / [ 0.229, 0.224, 0.225 ]

        return img

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                               horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_train = train_data_gen.n
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    # get class dict
    class_indices = train_data_gen.class_indices
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    model = GoogLeNet(im_height=im_height, im_width=im_width, num_classes=5, aux_logits=True)
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    model.summary()

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(_images, _labels):
        with tf.GradientTape() as tape:
            aux1, aux2, output = model(_images, training=True)
            loss1 = loss_object(_labels, aux1)
            loss2 = loss_object(_labels, aux2)
            loss3 = loss_object(_labels, output)
            loss = loss1 * 0.3 + loss2 * 0.3 + loss3
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(_labels, output)

    @tf.function
    def val_step(_images, _labels):
        _, _, output = model(_images, training=False)
        loss = loss_object(_labels, output)

        val_loss(loss)
        val_accuracy(_labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(range(total_train // batch_size), file=sys.stdout)
        for step in train_bar:
            images, labels = next(train_data_gen)
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # validate
        val_bar = tqdm(range(total_val // batch_size), file=sys.stdout)
        for step in val_bar:
            val_images, val_labels = next(val_data_gen)
            val_step(val_images, val_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save_weights("./weights/myGoogLeNet.ckpt")


if __name__ == '__main__':
    main()
