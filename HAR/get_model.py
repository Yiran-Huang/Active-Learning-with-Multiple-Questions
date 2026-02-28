import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p
        self.custom_eval_mode = False
        self.mask = None

    def inference_mode(self, mode=False):
        self.for_mode = mode
        if not mode:
            self.mask = None

    def forward(self, X):
        if self.training:
            return nn.functional.dropout(X, p=self.p, training=True)
        elif self.for_mode:
            n, *other_dims = X.size()
            if self.mask is None:
                self.mask = (torch.rand(1, *other_dims) > self.p).float().to(X.device)
            mask = self.mask.expand(n, *other_dims)
            return X * mask / (1 - self.p)
        else:
            return X

def set_inference_mode(model, mode=False):
    for module in model.modules():
        if isinstance(module, CustomDropout):
            module.inference_mode(mode)

class Resnet50_transfer(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50_transfer, self).__init__()
        resnet_model = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(resnet_model.children())[:-2])
        trainable = False
        for name, param in self.base_model.named_parameters():
            if name == "7.0.conv1.weight":
                trainable = True
            param.requires_grad = trainable
        self.secondpart= nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            CustomDropout(0.5),
        )
        self.last = nn.Linear(1024, num_classes)

    def forward(self, x,need_embed=False):
        x = self.base_model(x)
        x = self.secondpart(x)
        out = self.last(x)
        if need_embed:
            return out, x
        else:
            return out
    def get_embedding_dim(self):
        return 1024

class Simple_CNN(nn.Module):
    def __init__(self,num_classes):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = CustomDropout(0.25)
        self.dropout2 = CustomDropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.last = nn.Linear(128, num_classes)

    def forward(self, x,need_embed=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        out = self.last(x)
        if need_embed:
            return out, x
        else:
            return out
    def get_embedding_dim(self):
        return 128

class Logistic(nn.Module):
    def __init__(self,num_classes):
        super(Logistic, self).__init__()
        self.dropout1 = CustomDropout(0.25)
        self.last = nn.Linear(64,num_classes)

    def forward(self, x,need_embed=False):
        x = self.dropout1(x)
        out = self.last(x)
        if need_embed:
            return out, x
        else:
            return out
    def get_embedding_dim(self):
        return 64

class ANNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(561, 128)
        self.dropout1 = CustomDropout(0.25)
        self.last = nn.Linear(128, num_classes)

    def forward(self, x,need_embed=False):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        out = self.last(x)
        if need_embed:
            return out, x
        else:
            return out
    def get_embedding_dim(self):
        return 128

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])

        self.secondpart= nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            CustomDropout(0.5),
        )
        self.last = nn.Linear(1024, num_classes)

    def forward(self, x,need_embed=False):
        x = self.base_model(x)
        x = self.secondpart(x)
        out = self.last(x)
        if need_embed:
            return out, x
        else:
            return out
    def get_embedding_dim(self):
        return 1024