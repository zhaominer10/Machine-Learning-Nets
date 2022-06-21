import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import MobileNetV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path = "../../dataset/predict_image/sunflower.jpg"
assert os.path.exists(img_path), "file: {} does not exists.".format(img_path)
img = Image.open(img_path)
plt.imshow(img)
# plt.show()

img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

json_path = './'
