from keras import layers, Model, Sequential


def make_features(config):
    feature_layers = [ ]
    for v in config:
        if v == 'M':
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding='SAME', activation='relu')
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name='feature')


configs = {
    'vgg11': [ 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ],
    'vgg13': [ 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M' ],
    'vgg16': [ 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M' ],
    'vgg19': [ 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M' ],
}

