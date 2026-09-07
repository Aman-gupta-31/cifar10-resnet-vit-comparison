import torch
from basic_block import BasicBlock
from resnet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    # ResNet18 configuration is [2, 2, 2, 2]
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    return model.to(device)
