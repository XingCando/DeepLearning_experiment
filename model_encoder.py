import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN_Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, network):
        super(CNN_Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #2048,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        # 在ResNet的基础上引入了分组卷积（Grouped Convolution）的改进版本，具有更高的参数效率和准确率。
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]

        self.cnn = cnn
        self.cnn_list = nn.ModuleList(modules)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, image):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
        # feat2 = self.cnn(imageB)
        feat = image
        for module in self.cnn_list:
            feat = module(feat)

        return feat

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 3 through 4
        if fine_tune:
            for c in self.cnn_list:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class FeedForward(nn.Module):
    # 定义前馈神经网络FFN
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # 先升维到 hidden_dim ,再降维到原始维度
        return self.net(x)

        
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 输入与输出维度不变
        x = self.norm(x) # 先对输入作层归一化
        qkv = self.to_qkv(x).chunk(3, dim=-1) # 将输入映射至查询、键和值，每个维度为 inner_dim
        # 将查询、键和值拆分为多头，每个头的维度为 dim_head
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # 计算查询和键的注意力分数, 注意力分数矩阵形状为 [batch, heads, seq_len, seq_len]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        # 将注意力分数与值作加权求和
        out = torch.matmul(attn, v)
        # 将各个头计算的值拼接，形状为 [batch, seq_len, inner_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # [batch, seq_len, dim]


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        print("transformer encoder_n_layers=", depth)
        for _ in range(depth):
            # 定义Transformer层
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # 每一层都带残差连接
            x = attn(x) + x 
            x = ff(x) + x
        return self.norm(x)


class CNNTransformerModel(nn.Module):
    def __init__(self, network, emb_dim, num_heads, num_layers, dropout, h_feat, w_feat):
        super(CNNTransformerModel, self).__init__()
        self.cnn_encoder = CNN_Encoder(network)
        self.feature_dim = self.cnn_encoder.cnn.fc.in_features
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.h_embedding = nn.Embedding(h_feat, int(emb_dim/2))
        self.w_embedding = nn.Embedding(w_feat, int(emb_dim/2))
        self.feature_embedding = nn.Conv2d(self.feature_dim, emb_dim,1)
        
        self.transformer_encoder = TransformerEncoder(emb_dim, num_layers, num_heads, 64, dropout)
        

    def forward(self, images):
        features = self.cnn_encoder(images)
        features = self.feature_embedding(features)
        features = self.add_pos_embedding(features)
        output = self.transformer_encoder(features)
        # output的形状为[batch, h*w, embed_dim]
        return output
 
    def add_pos_embedding(self, x):
        if len(x.shape) == 3: # NLD
            b = x.shape[0]
            c = x.shape[-1]
            x = x.transpose(-1, 1).view(b, c, self.h_feat, self.w_feat)
        b, c, h, w = x.shape
        pos_h = torch.arange(h).to(device)
        pos_w = torch.arange(w).to(device)
        embed_h = self.h_embedding(pos_h)
        embed_w = self.w_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1) #(h, w, emb_dim)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1) #(batch, emb_dim, h, w)
        x = x + pos_embedding #(batch, emb_dim, h, w)
        # reshape back to NLD
        x = x.view(b, c, -1).transpose(-1, 1)  # NLD (batch, h*w, emb_dim)
        return x