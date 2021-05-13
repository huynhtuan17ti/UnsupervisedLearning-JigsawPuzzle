import torch
import torch.nn as nn
from torchvision import models


class JigsawAlexNet(nn.Module):

    def __init__(self, classes=1000):
        super(JigsawAlexNet, self).__init__()

        self.conv = models.alexnet(pretrained = True).features
        self.conv._modules['0'] = nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2) # modified to stride 2
        self.fc6 = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(9*1024,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.conv(x[i])
            z = self.fc6(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = torch.cat(x_list,1)
        x = self.fc7(x.view(B,-1))
        x = self.classifier(x)

        return x

class AlexNet(nn.Module):
    def __init__(self, classes = 10):
        super(AlexNet, self).__init__()
        self.conv = models.alexnet(pretrained = True).features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # testing
    net = JigsawAlexNet().cuda()
    from torchsummary import summary
    print(summary(net, (9, 3, 75, 75)))
    #print(net)
