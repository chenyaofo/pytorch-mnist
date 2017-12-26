import torch.utils.data
import torch.nn.functional
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms

import pickle

from model import LeNet5

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
            download=True,
        ),
        batch_size=100,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
            download=True,
        ),
        batch_size=100,
        shuffle=True,
    )

    net = LeNet5()
    cudable = torch.cuda.is_available()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)

    if cudable:
        net = net.cuda()

    def train(epoch):
        net.train()
        for batch_index, (data, target) in enumerate(train_loader,start=1):
            if cudable:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                print(f"TRAIN PHASE epoch={epoch:03d} iter={batch_index:03d} loss={loss.data[0]:.4f}")


    def test(epoch):
        net.eval()
        correct = 0
        for data, target in test_loader:
            if cudable:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = net(data)
            loss = criterion(output, target)
            predict_target = output.data.max(1)[1]
            correct += predict_target.eq(target.data).cpu().sum()
        print(f"TEST  PHASE epoch={epoch:03d} acc={correct/len(test_loader.dataset)*100:.2f}% loss={loss.data[0]:.4f}\n")


    max_epoch = 3
    for epoch in range(max_epoch):
        train(epoch)
        test(epoch)

    with open("LeNet5.model","wb") as f:
        pickle.dump(net,f)

    print("done.")
