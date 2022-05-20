from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os


def main():
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

    # # show training image examples, plot image with 1 row and 5 columns
    # sample_train_images, sample_train_labels = next(train_data_gen)  # label is one-hot coding
    #
    #
    # def plotImages(images_arr):
    #     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    #     axes = axes.flatten()
    #     for img, ax in zip(images_arr, axes):
    #         ax.imshow(img)
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    #
    #
    # plotImages(sample_train_images[ :5 ])

    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    # model = AlexNet_v2(num_classes=5)
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    model.summary()

    # # using keras high level api for training
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    #               metrics=[ "Accuracy" ])
    #
    # callbacks = [ tf.keras.callbacks.ModelCheckpoint(filepath='./weights/myAlex.h5',
    #                                                  save_best_only=True,
    #                                                  save_weights_only=True,
    #                                                  monitor='val_loss') ]
    #
    # history = model.fit(x=train_data_gen,
    #                     steps_per_epoch=total_train // batch_size,
    #                     epochs=epochs,
    #                     validation_data=val_data_gen,
    #                     validation_steps=total_val // batch_size,
    #                     callbacks=callbacks)
    #
    # history_dict = history.history
    #
    # train_loss = history_dict[ 'loss' ]
    # train_accuracy = history_dict[ 'Accuracy' ]
    # val_loss = history_dict[ 'val_loss' ]
    # val_accuracy = history_dict[ 'val_Accuracy' ]
    #
    # # loss plot
    # plt.figure()
    # plt.plot(range(epochs), train_loss, label='train_loss')
    # plt.plot(range(epochs), val_loss, label='val_loss')
    # plt.legend()
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    #
    # # accuracy plot
    # plt.figure()
    # plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    # plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    # plt.legend()
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    #
    # plt.show()

    # not using keras high level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(_images, _labels):
        with tf.GradientTape() as tape:
            predictions = model(_images, training=True)
            loss = loss_object(_labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(_labels, predictions)

    @tf.function
    def test_step(_images, _labels):
        predictions = model(_images)
        t_loss = loss_object(_labels, predictions)

        val_loss(t_loss)
        val_accuracy(_labels, predictions)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for step in range(total_train // batch_size):
            images, labels = next(train_data_gen)
            train_step(images, labels)

        for step in range(total_val // batch_size):
            test_images, test_labels = next(val_data_gen)
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              val_loss.result(),
                              val_accuracy.result() * 100))

        if val_loss.result() < best_val_loss:
            model.save_weights('./weights/myAlex.ckpt', save_format='tf')


if __name__ == '__main__':
    main()
