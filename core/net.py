import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# 定义Patch Embedding模块
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2

        # 将图片划分为patches并嵌入embedding向量
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # 将图片划分为 patches
        x = x.flatten(2)  # 展开成序列
        x = x.transpose(1, 2)  # 调整维度为 [B, num_patches, embed_dim]
        x = x + self.position_embedding  # 加入位置编码
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding size needs to be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, seq_length, embed_dim = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nhqk", Q, K)
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", attention, V).reshape(N, seq_length, self.embed_dim)
        out = self.fc_out(out)
        return out


# 定义Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4., dropout=0.1,hidden_size=3072,):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # MLP模块
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# 定义Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, output_dim=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.output_dim = output_dim

        # Patch Embedding模块
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                          embed_dim=embed_dim)

        # 分类Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
              for _ in range(depth)]
        )

        # 最后的分类Head
        self.norm = nn.LayerNorm(embed_dim)
        # self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        B = x.shape[0]

        # 获得patch embeddings
        x = self.patch_embed(x)

        # 添加分类token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)

        # 位置编码的dropout
        x = self.position_dropout(x)

        # 经过多个Transformer Block
        for block in self.blocks:
            x = block(x)

        # 分类token通过分类head输出类别
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 取出分类token
        x = cls_token_final.transpose(0,1)
        # out = self.head(cls_token_final)

        return x


class ViT1(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.1):
        super(ViT1,self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (N, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape,'|',self.pos_embedding.shape)
        x = x + self.pos_embedding
        x = self.dropout(x)

        return x


class ViT2(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, hidden_size=3072, mlp_ratio=4., dropout=0.1):
        super(ViT2, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_size=hidden_size, dropout=dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT3(nn.Module):
    def __init__(self, embed_dim=768):
        super(ViT3, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm(x)
        # cls_token_final = x[:, 0]  # 取出分类token
        # x = x.t()
        return x


class BasicBlock(nn.Module):  # Block for ResNet

    expansion = 1

    def __init__(self, mastermodel, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.mastermodel = mastermodel

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[2, 2, 2, 2], avg_output=False, output_dim=-1, resprestride=1,
                 res1ststride=1, res2ndstride=1, inchan=3):
        super().__init__()
        img_chan = inchan
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_chan, 64, kernel_size=3, padding=1, bias=False, stride=resprestride),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.in_channels = 64
        self.conv2_x = self._make_layer(block, 64, num_block[0], res1ststride)
        self.conv3_x = self._make_layer(block, 128, num_block[1], res2ndstride)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.conv6_x = nn.Identity() if output_dim <= 0 else self.conv_layer(512, output_dim, 1, 0)
        self.conv6_is_identity = output_dim <= 0
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if output_dim > -1:
            self.output_dim = output_dim
        else:
            self.output_dim = 512 * block.expansion
        self.avg_output = avg_output

    def conv_layer(self, input_channel, output_channel, kernel_size=3, padding=1):
        print("conv layer input", input_channel, "output", output_channel)
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2))
        return res

    def _make_layer(self, block, out_channels, num_blocks, stride):
        print("Making resnet layer with channel", out_channels, "block", num_blocks, "stride", stride)

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(None, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.conv6_x(output)
        if self.avg_output:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
        return output


def build_backbone(img_size, backbone_name, projection_dim, inchan=3, patchsize=8):
    if backbone_name == 'resnet18':
        backbone = ResNet(output_dim=projection_dim, inchan=inchan, resprestride=1, res1ststride=1, res2ndstride=2)
        cam_size = int(img_size / 8)
    elif backbone_name == 'resnet34':
        backbone = ResNet(output_dim=projection_dim, inchan=inchan, num_block=[3, 4, 6, 3], resprestride=1,
                          res1ststride=2, res2ndstride=2)
        cam_size = int(img_size / 32)
    elif backbone_name == 'vit':
        print("====Using VIT!====")
        # backbone = VisionTransformer(output_dim=768, img_size=img_size, patch_size=patchsize, depth=6)
        embeding_dim = 768
        backbone = [ViT1(img_size=img_size, patch_size=patchsize),ViT2(depth=1),ViT3()] # Change Depth For Pretrain and Test
        cam_size = int(img_size / 8)
        return backbone, embeding_dim, cam_size
    else:
        valid_backbone = backbone_name
        raise Exception(f'Backbone \"{valid_backbone}\" is not defined.')

    return backbone, backbone.output_dim, cam_size


class BaselineNet(nn.Module):
    def __init__(self, args):
        super(BaselineNet, self).__init__()
        backbone, feature_dim, _ = build_backbone(img_size=args['img_size'],
                                                  backbone_name=args['backbone'],
                                                  pretrained=args['pretrained'],
                                                  projection_dim=-1,
                                                  inchan=3)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = conv1x1(feature_dim, args['num_known'])

    def forward(self, x, y=None):
        x = self.backbone(x)
        ft = self.classifier(x)
        logits = self.pool(ft)
        logits = logits.view(logits.size(0), -1)
        outputs = {'logits': [logits]}
        return outputs

    def get_params(self, prefix='extractor'):
        extractor_params = list(self.backbone.parameters())
        extractor_params_ids = list(map(id, self.backbone.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim() == 1:
            logit = logit.unsqueeze(0)
        return logit


class MultiBranchNet(nn.Module):
    def __init__(self, args=None):
        super(MultiBranchNet, self).__init__()
        self.img_size = args['img_size']
        self.gate_temp = args['gate_temp']
        self.num_known = args['num_known']
        self.norm = nn.LayerNorm(self.num_known)
        backbone, vit_feature_dim, self.cam_size = build_backbone(img_size=args['img_size'],
                                                              backbone_name=args['backbone'],
                                                              projection_dim=-1,
                                                              inchan=3)
        resnet, conv_feature_dim, __ = build_backbone(img_size=args['img_size'], backbone_name='resnet18',projection_dim=-1,inchan=3)
        # vit for b1 and b2
        self.patchemb_l3 = backbone[0]
        self.branch1_l4 = backbone[1]
        self.branch1_l5 = backbone[2]
        self.branch1_cls = nn.Linear(vit_feature_dim,self.num_known) #Conv1x1d1(feature_dim, self.num_known)
        self.branch2_cls = nn.Linear(vit_feature_dim,self.num_known) #Conv1x1d1(feature_dim, self.num_known)
        self.branch3_cls = nn.Linear(vit_feature_dim,self.num_known) #Conv1x1d1(feature_dim, self.num_known)

        self.img_size = args['img_size']
        self.gate_temp = args['gate_temp']
        self.num_known = args['num_known']
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_poold1 = nn.AdaptiveAvgPool1d(1)

        self.branch2_l4 = copy.deepcopy(self.branch1_l4)
        self.branch2_l5 = copy.deepcopy(self.branch1_l5)

        # resnet for b3 and b4
        self.shared_l3 = nn.Sequential(*list(resnet.children())[:-6])
        self.branch3_l4 = nn.Sequential(*list(resnet.children())[-6:-3])
        self.branch3_l5 = nn.Sequential(*list(resnet.children())[-3])
        self.branch3_cls = conv1x1(conv_feature_dim, self.num_known)

        self.branch4_l4 = copy.deepcopy(self.branch3_l4)
        self.branch4_l5 = copy.deepcopy(self.branch3_l5)
        self.branch4_cls = conv1x1(conv_feature_dim, self.num_known)

        self.gate_l3 = copy.deepcopy(self.shared_l3)
        self.gate_l4 = copy.deepcopy(self.branch3_l4)
        self.gate_l5 = copy.deepcopy(self.branch3_l5)
        self.gate_cls = nn.Sequential(Classifier(conv_feature_dim, int(conv_feature_dim / 4), bias=True),
                                      Classifier(int(conv_feature_dim / 4), 4, bias=True)) # 4 branchs linear to 4

    def forward(self, x, y=None, return_ft=False):

        b = x.size(0)
        conv_ft_till_l3 = self.shared_l3(x)
        vit_ft_till_l3 = self.patchemb_l3(x)
        # b1 is vit
        branch1_l4 = self.branch1_l4(vit_ft_till_l3.clone())
        branch1_l5 = self.branch1_l5(branch1_l4)
        b1_ft_cams = branch1_l5
        b1_logits = self.norm(self.branch1_cls(branch1_l5[:, 0]))
        # b2 is vit
        branch2_l4 = self.branch2_l4(vit_ft_till_l3.clone())
        branch2_l5 = self.branch2_l5(branch2_l4)
        b2_ft_cams = branch2_l5
        b2_logits = self.norm(self.branch2_cls(branch2_l5[:, 0]))
        # b3 is resnet
        branch3_l4 = self.branch3_l4(conv_ft_till_l3.clone())
        branch3_l5 = self.branch3_l5(branch3_l4)
        b3_ft_cams = self.branch3_cls(branch3_l5)
        b3_logits = self.avg_pool(b3_ft_cams).view(b, -1) # 128,6,4,4 -> 128,6,1,1 -> 128,6
        # b4 is resnet
        branch4_l4 = self.branch4_l4(conv_ft_till_l3.clone())
        branch4_l5 = self.branch4_l5(branch4_l4)
        b4_ft_cams = self.branch4_cls(branch4_l5)
        b4_logits = self.avg_pool(b4_ft_cams).view(b, -1) # 128,6,4,4 -> 128,6,1,1 -> 128,6
        if y is not None:
            # 得到正确类下的分类器特征
            y = y.long() # For cifar10, y is in int32 but for gather need int64, coverage here
            b1_ft_cams = b1_ft_cams.permute(0, 2, 1)
            b2_ft_cams = b2_ft_cams.permute(0, 2, 1)
            b, _, h, w = b3_ft_cams.size()
            cams = torch.cat([
                b1_ft_cams.gather(dim=1, index=y[:, None, None].repeat(1, 1, b1_ft_cams.shape[-1]))[:, :, 1:], # b,1,x+1 -> b,1,x
                b2_ft_cams.gather(dim=1, index=y[:, None, None].repeat(1, 1, b2_ft_cams.shape[-1]))[:, :, 1:],
                (b3_ft_cams.gather(dim=1, index=y[:, None, None, None].repeat(1, 1, b3_ft_cams.shape[-2],b3_ft_cams.shape[-1]))).view(b,1,h*w), # b,1,8,8 -> b,1,64
                (b4_ft_cams.gather(dim=1, index=y[:, None, None, None].repeat(1, 1, b4_ft_cams.shape[-2],b4_ft_cams.shape[-1]))).view(b,1,h*w),
            ], dim=1)
        if return_ft:
            # (b,17,768) for vit,(b,clsnum,4,4) for resnet
            # fts = b1_ft_cams.detach().clone() + b2_ft_cams.detach().clone() + b3_ft_cams.detach().clone() + b4_ft_cams.detach().clone()
            fts = b3_ft_cams.detach().clone() + b4_ft_cams.detach().clone()

        gate_l5 = self.gate_l5(self.gate_l4(self.gate_l3(x))) # 128,512,4,4 for resne|128,17,768 for vit

        if gate_l5.dim() > 3:
            gate_pool = self.avg_pool(gate_l5).view(b, -1)
        else:
            gate_pool = gate_l5[:,0]
        gate_pred = F.softmax(self.gate_cls(gate_pool) / self.gate_temp, dim=1) # (batch_size,3)
        gate_logits = torch.stack([b1_logits.detach(), b2_logits.detach(), b3_logits.detach(), b4_logits.detach()], dim=-1)
        gate_logits = gate_logits * gate_pred.view(gate_pred.size(0), 1, gate_pred.size(1))
        gate_logits = gate_logits.sum(-1)

        logits_list = [b1_logits, b2_logits, b3_logits, b4_logits, gate_logits]
        if return_ft and y is None:
            outputs = {'logits': logits_list, 'gate_pred': gate_pred, 'fts': fts}
        else:
            outputs = {'logits': logits_list, 'gate_pred': gate_pred, 'cams': cams}

        return outputs

    def get_params(self, prefix='extractor'):
        extractor_params = list(self.shared_l3.parameters()) + \
                           list(self.branch1_l4.parameters()) + list(self.branch1_l5.parameters()) + \
                           list(self.branch2_l4.parameters()) + list(self.branch2_l5.parameters()) + \
                           list(self.branch3_l4.parameters()) + list(self.branch3_l5.parameters()) + \
                           list(self.gate_l3.parameters()) + list(self.gate_l4.parameters()) + list(
            self.gate_l5.parameters())
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params


if __name__ == "__main__":
    batch_size = 1
    img_size = 16
    img = (torch.randn(batch_size, 3, img_size, img_size))#.to('cuda')
    y = (torch.randint(0,6,(batch_size,)))#.to('cuda'))
    '''
    vit = ViTEncoder(img_size=32, patch_size=8, in_channels=3, output_dim=1024)
    op32x32 = vit(img)
    print(op32x32.shape)  # Should be [1, 768]
    '''
    mln = MultiBranchNet(args={'backbone':'vit','img_size':img_size,'gate_temp':1,'num_known':20,'gate_temp':1})
    # mln.to('cuda')
    # import loadpretrain
    # loadpretrain.load_4b_2v(mln,"C:\mlcodes\datasets\pretrainModel\imagenet21k_ViT-B_8.npz")

    # state_dict = mln.state_dict()
    # for key in state_dict:
    #     print(key, state_dict[key].shape)
    # print("=============================")

    output = mln(img,y=y)
    # output = mln(img,return_ft=True)



    # torch.cuda.memory_summary()
    # total = sum([param.nelement() for param in mln.parameters()])
    # print('Number of parameter: % .4fM' % (total / 1e6)) # 567.493M for 12Blocks #95.081M for 2 Blocks
    # print(output)
    # print(len(output['logits']))
    # print(output['logits'][0].shape)
    #
    from thop import profile
    flops, params = profile(mln, inputs=(img,y))
    print("F&P:",flops,params)

    import torchinfo


    # print(output['cams'].shape) # (8,3,4,4) for resnet
    # print(output['fts'].shape)
    '''
    4
    torch.Size([8, 20])
    torch.Size([8, 3])
    torch.Size([8, 20])
    '''