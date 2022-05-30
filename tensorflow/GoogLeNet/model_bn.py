from keras import layers, models, Model, Sequential


def GoogLeNet(im_height=224, im_width=224, num_classes=1000, aux_logits=False):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", use_bias=False, name="conv2d_1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/bn")(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)

    x = layers.Conv2D(64, kernel_size=1, use_bias=False, name="conv2d_2")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/bn")(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(192, kernel_size=3, padding="SAME", use_bias=False, name="conv2d_3")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/bn")(x)
    x = layers.ReLU()(x)

    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)

    x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)

    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
    x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)
    if aux_logits:
        aux1 = Auxiliary(num_classes, name="aux_1")(x)

    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
    if aux_logits:
        aux2 = Auxiliary(num_classes, name="aux_2")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
    x = layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)

    x = layers.Flatten(name="output_flatten")(x)
    x = layers.Dropout(rate=0.4, name="output_dropout")(x)
    x = layers.Dense(num_classes, name="output_dense")(x)
    aux3 = layers.Softmax(name="aux_3")(x)

    if aux_logits:
        model = models.Model(inputs=input_image, outputs=[ aux1, aux2, aux3 ])
    else:
        model = models.Model(inputs=input_image, outputs=aux3)
    return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.branch1 = Sequential([
            layers.Conv2D(ch1x1, kernel_size=1, use_bias=False, name="conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="bn"),
            layers.ReLU() ], name="branch1")

        self.branch2 = Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, use_bias=False, name="0/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="0/bn"),
            layers.ReLU(),
            layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", use_bias=False, name="1/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="1/bn"),
            layers.ReLU() ], name="branch2")  # output_size= input_size

        self.branch3 = Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, use_bias=False, name="0/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="0/bn"),
            layers.ReLU(),
            layers.Conv2D(ch5x5, kernel_size=3, padding="SAME", use_bias=False, name="1/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="1/bn"),
            layers.ReLU() ], name="branch3")  # output_size= input_size

        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),  # caution: default strides==pool_size
            layers.Conv2D(pool_proj, kernel_size=1, use_bias=False, name="1/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="1/bn"),
            layers.ReLU() ], name="branch4")  # output_size= input_size

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class Auxiliary(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(Auxiliary, self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)

        self.conv = layers.Conv2D(128, kernel_size=1, use_bias=False, name="conv/conv")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv/bn")
        self.relu1 = layers.ReLU()

        self.fc1 = layers.Dense(1024, activation='relu', name="fc1")
        self.fc2 = layers.Dense(num_classes, name="fc2")
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        x = self.averagePool(inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = self.fc1(x)
        x = layers.Dropout(0.5)(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
