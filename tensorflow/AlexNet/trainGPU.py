import glob
import os
import random
import time

import tensorflow as tf

from model import AlexNet_v1

os.environ[ 'CUDA_DEVICE_ORDER' ] = 'PCI_BUS_ID'
os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

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

    data_class = [ cla for cla in os.listdir(train_dir) if os.path.isdir((os.path.join(train_dir, cla))) ]
    class_dict = dict((value, index) for index, value in enumerate(data_class))
    class_num = len(data_class)

    # load train images list
    train_image_list = glob.glob(train_dir + "/*/*.jpg")
    random.shuffle(train_image_list)
    train_num = len(train_image_list)
    assert train_num > 0, "cannot find any .jpg file in {}.".format(train_dir)
    train_label_list = [ class_dict[ path.split(os.path.sep)[ -2 ] ] for path in train_image_list ]

    # load validation image list
    val_image_list = glob.glob(validation_dir + "/*/*.jpg")
    random.shuffle(val_image_list)
    val_num = len(val_image_list)
    assert val_num > 0, "cannot find any .jpg file in {}.".format(validation_dir)
    val_label_list = [ class_dict[ path.split(os.path.sep)[ -2 ] ] for path in val_image_list ]

    print("using {} for training, {} images for validation.".format(train_num, val_num))


    def process_path(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [ im_height, im_width ])
        return image, label


    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=train_num) \
        .map(process_path, num_parallel_calls=AUTOTUNE) \
        .repeat().batch(batch_size).prefetch(AUTOTUNE)

    # load validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_path, num_parallel_calls=AUTOTUNE) \
        .repeat().batch(batch_size)

    # model instantiation
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    model.summary()

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

        t1 = time.perf_counter()
        for index, (images, labels) in enumerate(train_dataset):
            train_step(images, labels)
            if index+1 == train_num // batch_size:
                break
        print(time.perf_counter()-t1)

        for index, (images, labels) in enumerate(val_dataset):
            test_step(images, labels)
            if index+1 == val_num // batch_size:
                break

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              val_loss.result(),
                              val_accuracy.result() * 100))

        if val_loss.result() < best_val_loss:
            model.save_weights('./weights/myAlex_gpu.ckpt', save_format='tf')


if __name__ == "__main__":
    main()
