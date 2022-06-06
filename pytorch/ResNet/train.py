import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import resnet34, resnet50, resnet101
import torchvision.models.resnet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 这里的normalize参数是参考pytorch官方关于transform learning的教程
    data_transform = {
        'train': transforms.Compose([ transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]),
        'val': transforms.Compose([ transforms.Resize(256),     # 将最小边缩放到256，因为长宽比不变，因此另一个边自适应
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    }

    data_root = '../..'
    image_path = os.path.join(data_root, 'dataset', 'flower_data')
    assert os.path.exists(image_path), "{} path does not exits.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform[ 'train' ])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                       transform=data_transform[ 'val' ])
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    flower_list = train_dataset.class_to_idx
    print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([ os.cpu_count(), batch_size if batch_size > 1 else 0, 8 ])
    print('Using {} dataloader workers every process.'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)

    print('Using {} images for training, {} images for validation.'.format(train_num, val_num))

    net = resnet34()    # 此时的实例化模型有1000个节点
    model_weight_path = 'resnet34-pre.pth'
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 5)    # 迁移学习，重新赋值全连接层
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = 'resnet34.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step+1)/len(train_loader)
            a = '*' * int(rate * 50)
            b = '.' * int((1 - rate) * 50)

            print( '\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}'.format(int(rate*100), a, b, running_loss), end="")

        # validate
        # net.train 与net.eval能控制是否流走模型中的batch normalization层
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[ 1 ]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
