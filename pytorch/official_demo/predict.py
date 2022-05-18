import torch
import torchvision.transforms as transforms


def main():
    from PIL import Image
    from model import LeNet

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    net = LeNet()
    net.load_state_dict(torch.load('./Lenet.pth'))

    image = Image.open('./plane.jpg')
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)

    with torch.no_grad():
        output = net(image)
        predict = torch.max(output, dim=1)[1].data.numpy()

        # # we can use softmax alternatively
        # predict = torch.softmax(output, dim=1)

    print(classes[int(predict)])


if __name__ == '__main__':
    main()
