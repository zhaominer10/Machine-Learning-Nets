from tensorflow.keras import layers, models, Models, Sequential


def GoogLeNet(im_height=224, im_width=224, num_classes=1000, aux_logits=False):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, poo_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)

