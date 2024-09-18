import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchsummary import summary
from option import get_args
opt = get_args()

def CustomMobileNetV3():
    model = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, opt.num_classes)
    return model


if __name__ == '__main__':
    model = CustomMobileNetV3()
    print(model)
    print(summary(model.to(opt.device), (3, opt.loadsize, opt.loadsize), opt.batch_size))

