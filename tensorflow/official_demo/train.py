import tensorflow as tf
from model import MyModel


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # add channels dimension
    x_train = x_train[ ..., tf.newaxis ]
    x_test = x_test[ ..., tf.newaxis ]

    # create data generator
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # create model
    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    # define train function including calculating loss, applying gradient and calculating accuracy
    @tf.function
    def train_step(_images, _labels):
        with tf.GradientTape() as tape:
            predicts = model(_images)
            loss = loss_object(_labels, predicts)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(_labels, predicts)

    @tf.function
    def test_step(_images, _labels):
        predicts = model(_images)
        t_loss = loss_object(_labels, predicts)

        test_loss(t_loss)
        test_accuracy(_labels, predicts)

    epochs = 5

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, loss: {}, acc: {}, test loss: {}, test acc: {}'
        print(
            template.format(epoch + 1, train_loss.result(),
                            train_accuracy.result() * 100, test_loss.result(),
                            test_accuracy.result() * 100))


if __name__ == '__main__':
    main()
