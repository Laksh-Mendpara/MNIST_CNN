import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(CNN, self).__init__()
        self.my_cnn = nn.Sequential(  # 28x28

            nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),  # 26x26

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # 24x24

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 12x12

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # 10x10

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # 10x10

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 5x5

            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(5, 5), stride=(1, 1)),  # 1x1
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        self.my_nn = nn.Sequential(
            nn.Linear(10, 10),  # 576
            # nn.BatchNorm1d(120),
            # nn.ReLU(),
            # # nn.Dropout(0.25),
            # nn.Linear(120, num_classes)
        )
        self.initialize_weights()

    def forward(self, x):
        x = self.my_cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.my_nn(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = CNN()
    summary(model, (1, 28, 28))
