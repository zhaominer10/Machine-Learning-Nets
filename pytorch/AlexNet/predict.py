import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([ transforms.Resize((224, 244)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

# load image
img_path = '../../dataset/predict_image/sunflower.jpg'
assert os.path.exists(img_path), 'file: {} dose not exist.'.format(img_path)
img = Image.open(img_path).convert('RGB')

plt.imshow(img)
img = data_transform(img)

# [N, C, H, W]
img = torch.unsqueeze(img, dim=0)

# read class_indices
json_path = './class_indices.json'
assert os.path.exists(json_path), 'file: {} does not exist.'.format(json_path)

with open(json_path, 'r') as f:
    class_dict = json.load(f)

# create model
model = AlexNet(num_classes=5).to(device)

# load model weights
weights_path = './AlexNet.pth'
assert os.path.exists(weights_path), 'file {} does not exist.'.format(weights_path)
model.load_state_dict(torch.load(weights_path))

model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print(class_dict[str(predict_cla)], predict[predict_cla].item())

plt.show()
