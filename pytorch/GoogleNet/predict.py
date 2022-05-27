import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogleNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([ transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

    img_path = '../../dataset/predict_image/sunflower.jpg'
    assert os.path.exists(img_path), "file: {} does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: {} does not exist.".format(json_path)

    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    model = GoogleNet(num_classes=5, aux_logits=False).to(device)

    weights_path = './googlenet.pth'
    assert os.path.exists(weights_path), "file: {} does not exist.".format(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

        res = "predicted class is: {}   prob: {:.3}".format(class_indict[ str(predict_cla) ],
                                                            predict[ predict_cla ].numpy())

        print(res)
        print()

        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[ str(i) ],
                                                      predict[ i ].numpy()))


if __name__ == "__main__":
    main()