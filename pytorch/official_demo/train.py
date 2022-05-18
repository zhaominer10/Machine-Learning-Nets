from model import LeNet
from torch.utils.data import DataLoader

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)),  # normalize the image in the range [-1, 1]
    ])

    # 50,000 train images
    train_set = torchvision.datasets.CIFAR10(root="./data",
                                             train=True,
                                             download=False,
                                             transform=transform)

    train_loader = DataLoader(train_set,
                              batch_size=50,
                              shuffle=True,
                              num_workers=0)

    # 10,000 test images
    test_set = torchvision.datasets.CIFAR10(root="./data",
                                            train=False,
                                            download=False,
                                            transform=transform)

    test_loader = DataLoader(test_set,
                             batch_size=10000,
                             shuffle=False,
                             num_workers=0)

    test_data_iter = iter(test_loader)
    test_images, test_labels = test_data_iter.next()

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
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # sets the gradients of all optimized tensors to zero
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()

            # updates parameters
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(test_images)

                    # return the index of the max value of each row
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == test_labels
                                ).sum().item() / test_labels.size(dim=0)

                    print('[%d, %5d] train_loss:%.3f test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
