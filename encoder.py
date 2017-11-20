import torch
import torchvision.models as models
import torch.nn as nn

class EncoderCNN(nn.Module):
  def __init__(self, embed_size):
    super(EncoderCNN, self).__init__()
    self.vgg16 = models.vgg16(pretrained=True)
    # removed this for most recent trial
    self.vgg16 = models.vgg16_bn(pretrained=True)
    if torch.cuda.is_available():
      self.vgg16.cuda()
    # delete so final layer is fc7
    modules = list(self.vgg16.children())
    pooling_layer = modules[0]
    self.pooling_layer = pooling_layer
    self.classify_layer = nn.Sequential(*(modules[1][i] for i in range(6) if (i + 1) % 3 != 0))
    self.linear = nn.Linear(4096, embed_size)
    # not sure if i should keep this one yet
    self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    if torch.cuda.is_available():
      self.linear.cuda()
      self.bn.cuda()

  def forward(self, images):
    features = self.pooling_layer(images)
    (_, C, H, W) = features.data.size()
    features = features.view( -1 , C * H * W)
    features = self.classify_layer(features)
    #features = self.bn(self.linear(features))
    return features
