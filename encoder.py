import torch
import torchvision.models as models
import torch.nn as nn

class EncoderCNN(nn.Module):
  def __init__(self, isNormalized=False):
    super(EncoderCNN, self).__init__()
    vgg16 = models.vgg16_bn(pretrained=True) if isNormalized else models.vgg16(pretrained=True)
    print("using normalized" if isNormalized else "not normalized")
    if torch.cuda.is_available():
      vgg16.cuda()
    # removing dropout and classification layer
    modules = list(vgg16.children())
    self.pooling_layer = modules[0]
    self.classify_layer = nn.Sequential(*(modules[1][i] for i in range(6) if (i + 1) % 3 != 0))

  def forward(self, images):
    features = self.pooling_layer(images)
    (_, C, H, W) = features.data.size()
    features = features.view( -1 , C * H * W)
    return self.classify_layer(features)
