from keras.preprocessing.image import ImageDataGenerator
from model import vgg
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    image_path = os.path.join(data_root, 'dataset', 'flower_data')
    train_dir = os.path.join(image_path, 'train')
    val_dir = os.path.join(image_path, 'val')
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(val_dir), "cannot find {}".format(val_dir)

    if not os.path.exists('weights'):
        os.makedirs('weights')

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10

    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
    val_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    val_data_gen = val_image_generator.flow_from_directory(directory=val_dir,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')

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

    model = vgg(model_name='vgg16', im_height=im_height, im_width=im_width, num_classes=5)
    model.summary()

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[ "Accuracy" ])

    callbacks = [ tf.keras.callbacks.ModelCheckpoint(filepath='./weights/myVGG.h5',
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     monitor='val_loss') ]

    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

    history_dict = history.history

    train_loss = history_dict[ 'loss' ]
    train_accuracy = history_dict[ 'Accuracy' ]
    val_loss = history_dict[ 'val_loss' ]
    val_accuracy = history_dict[ 'val_Accuracy' ]

    # loss plot
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # accuracy plot
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.show()


if __name__ == '__main__':
    main()
