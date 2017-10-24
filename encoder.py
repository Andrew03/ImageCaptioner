import torch
import torchvision.models as models
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        if torch.cuda.is_available():
            self.vgg16.cuda()
        # delete so final layer is fc7
        # do i delete the entire classify layer?
        # its of size 25088 if i do...
        modules = list(self.vgg16.children())
        #modules = list(vgg16.children())[:-1]
        pooling_layer = modules[0]
        self.pooling_layer = pooling_layer
        self.classify_layer = nn.Sequential(*(modules[1][i] for i in range(6)))
        # should i keep the dropout and relu layer?
        #self.vgg16 = nn.Sequential(pooling_layer, classify_layer)
        #self.vgg16 = nn.Sequential(*modules)
        self.linear = nn.Linear(4096, embed_size)
        #to_embedding = nn.Linear(to_embedding[-1].in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        if torch.cuda.is_available():
            self.linear.cuda()
            self.bn.cuda()
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.2)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        #features = self.vgg16(images)
        features = self.pooling_layer(images)
        (_, C, H, W) = features.data.size()
        features = features.view( -1 , C * H * W)
        features = self.classify_layer(features)
        #features = self.linear(features)
        #features = self.bn(self.linear(features))
        return features
