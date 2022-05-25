from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf


conv_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                mode='fan_out',
                                                                distribution='truncated_normal')
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1./3.,
                                                                 mode='fan_out',
                                                                 distribution='uniform')


def VGG(feature, im_height=224, im_width=224, num_classes=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu',
                     kernel_initializer=dense_kernel_initializer)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu',
                     kernel_initializer=dense_kernel_initializer)(x)
    x = layers.Dense(num_classes,
                     kernel_initializer=dense_kernel_initializer)(x)
    output = layers.Softmax()(x)
    model = Model(inputs=input_image, outputs=output)
    return model


def make_features(config):
    feature_layers = [ ]
    for v in config:
        if v == 'M':
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding='SAME', activation='relu',
                                   kernel_initializer=conv_kernel_initializer)
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name='feature')


configs = {
    'vgg11': [ 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ],
    'vgg13': [ 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ],
    'vgg16': [ 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M' ],
    'vgg19': [ 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M' ],
}


def vgg(model_name='vgg16', im_height=224, im_width=224, num_classes=1000):
    assert model_name in configs.keys(), "not support model {}".format(model_name)
    config = configs[model_name]
    model = VGG(make_features(config), im_height=im_height, im_width=im_width, num_classes=num_classes)
    return model
