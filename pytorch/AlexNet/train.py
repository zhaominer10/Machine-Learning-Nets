import json
import os
import time

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "train": transforms.Compose([ transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]),
        "val": transforms.Compose([ transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    }

    # train dataset
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, 'dataset', 'flower_data')
    assert os.path.exists(image_path), "{} path dose not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(os.path.join(image_path, "train"),
                                         transform=data_transform[ "train" ])

    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    print(json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32

    # number of worker
    nw = min([ os.cpu_count(), batch_size if batch_size > 1 else 0, 8 ])
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    # validation dataset
    val_dataset = datasets.ImageFolder(os.path.join(image_path, "val"),
                                       transform=data_transform[ "val" ])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

    # show images
    # change val_loader parameters, batch_size = 4, shuffle = True, num_workers = 0
    # test_data_iter = iter(val_loader)
    # test_image, test_label = test_data_iter.next()
    #
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # denormalize
    #     nping = img.numpy()
    #     plt.imshow(np.transpose(nping, (1, 2, 0)))
    #     plt.show()
    #
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    params = list(net.parameters())

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)

    # train
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end='')
        print()
        print(time.perf_counter() - t1)

        # val
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for data_test in val_loader:
                test_images, test_labels = data_test
                outputs = net(test_images.to(device))
                predicts_y = torch.max(outputs, dim=1)[ 1 ]
                acc += (predicts_y == test_labels.to(device)).sum().item()

        accuracy = acc / val_num
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(net.state_dict(), save_path)

        print('[epoch %d] train_loss: %.3f accuracy: %.3f' % (epoch + 1, running_loss / train_steps, accuracy))

    print('Finished Training')


if __name__ == '__main__':
    main()
