import os
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import AlexNet_v1, AlexNet_v2


def main():
    im_height = 224
    im_width = 224

    img_path = '../../dataset/predict_image/sunflower.jpg'
    assert os.path.exists(img_path), 'file: {} does not exist.'.format(img_path)
    img = Image.open(img_path).convert('RGB')

    img = img.resize((im_width, im_height))
    plt.imshow(img)
    plt.show()

    img = np.array(img) / 255.

    # add the image to a batch where it's the only member
    img = (np.expand_dims(img, 0))

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: {} does not exist.".format(json_path)
    with open(json_path, 'r') as f:
        class_indices = json.load(f)

    model = AlexNet_v1(num_classes=5)
    weights_path = './weights/myAlex.h5'
    assert os.path.exists(weights_path), "file {} does not exist.".format(weights_path)
    model.load_weights(weights_path)

    result = np.squeeze(model.predict(img))
    print(result)
    predict_class = np.argmax(result)
    print(predict_class)

    print_res = "predicted class is {} which has prob: {:.3}".format(class_indices[ str(predict_class) ],
                                                                     result[ predict_class ])

    print(print_res)
    print()
    for i in range(len(result)):
        print('class: {:10} prob: {:.3}'.format(class_indices[ str(i) ],
                                                result[ i ]))


if __name__ == "__main__":
    main()