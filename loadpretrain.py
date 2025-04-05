import torch
import numpy as np


def load_vit(model, weight_path="imagenet21k_ViT-B_8.npz"):
    # 加载预训练的权重
    npz_weights = np.load(weight_path)

    model.patch_embed.weight.data = torch.from_numpy(npz_weights['embedding/kernel'].transpose(3, 2, 0, 1))
    model.patch_embed.bias.data = torch.from_numpy(npz_weights['embedding/bias'])

    # model.pos_embedding.data = torch.from_numpy(npz_weights['Transformer/posembed_input/pos_embedding'])
    model.cls_token.data = torch.from_numpy(npz_weights['cls'])

    for i, block in enumerate(model.transformer_blocks):
        block.norm1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        block.norm1.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])

        block.norm2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
        block.norm2.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])

        block.attention.query.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.query.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].flatten())
        block.attention.key.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].transpose(1, 0,
                                                                                                             2).reshape(
                768, -1))
        block.attention.key.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].flatten())
        block.attention.value.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.value.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].flatten())
        block.attention.fc_out.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].transpose(2, 1,
                                                                                                             0).reshape(
                768, 768))
        block.attention.fc_out.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        block.ff.fc1.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'].T)
        block.ff.fc1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        block.ff.fc2.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'].T)
        block.ff.fc2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

    model.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    model.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    model.pre_logits.weight.data = torch.from_numpy(npz_weights['pre_logits/kernel'])
    model.pre_logits.bias.data = torch.from_numpy(npz_weights['pre_logits/bias'])

    model.head.weight.data = torch.from_numpy(npz_weights['head/kernel'].T)
    model.head.bias.data = torch.from_numpy(npz_weights['head/bias'])


def load_mbvit(model, weight_path="imagenet21k_ViT-B_8.npz"):
    # 加载预训练的权重
    npz_weights = np.load(weight_path)

    model.shared_l3.patch_embed.weight.data = torch.from_numpy(npz_weights['embedding/kernel'].transpose(3, 2, 0, 1))
    model.shared_l3.patch_embed.bias.data = torch.from_numpy(npz_weights['embedding/bias'])
    # model.shared_l3.pos_embedding.data = torch.from_numpy(npz_weights['Transformer/posembed_input/pos_embedding'])
    model.shared_l3.cls_token.data = torch.from_numpy(npz_weights['cls'])

    model.gate_l3.patch_embed.weight.data = torch.from_numpy(npz_weights['embedding/kernel'].transpose(3, 2, 0, 1))
    model.gate_l3.patch_embed.bias.data = torch.from_numpy(npz_weights['embedding/bias'])
    # model.gate_l3.pos_embedding.data = torch.from_numpy(npz_weights['Transformer/posembed_input/pos_embedding'])
    model.gate_l3.cls_token.data = torch.from_numpy(npz_weights['cls'])

    for i, block in enumerate(model.branch1_l4.blocks):
        block.norm1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        block.norm1.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])

        block.norm2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
        block.norm2.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])

        block.attention.query.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.query.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].flatten())
        block.attention.key.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].transpose(1, 0,
                                                                                                             2).reshape(
                768, -1))
        block.attention.key.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].flatten())
        block.attention.value.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.value.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].flatten())
        block.attention.fc_out.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].transpose(2, 1,
                                                                                                             0).reshape(
                768, 768))
        block.attention.fc_out.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        block.ff.fc1.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'].T)
        block.ff.fc1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        block.ff.fc2.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'].T)
        block.ff.fc2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

    for i, block in enumerate(model.branch2_l4.blocks):
        block.norm1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        block.norm1.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])

        block.norm2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
        block.norm2.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])

        block.attention.query.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.query.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].flatten())
        block.attention.key.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].transpose(1, 0,
                                                                                                             2).reshape(
                768, -1))
        block.attention.key.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].flatten())
        block.attention.value.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.value.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].flatten())
        block.attention.fc_out.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].transpose(2, 1,
                                                                                                             0).reshape(
                768, 768))
        block.attention.fc_out.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        block.ff.fc1.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'].T)
        block.ff.fc1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        block.ff.fc2.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'].T)
        block.ff.fc2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

    for i, block in enumerate(model.branch3_l4.blocks):
        block.norm1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        block.norm1.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])

        block.norm2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
        block.norm2.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])

        block.attention.query.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.query.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].flatten())
        block.attention.key.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].transpose(1, 0,
                                                                                                             2).reshape(
                768, -1))
        block.attention.key.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].flatten())
        block.attention.value.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.value.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].flatten())
        block.attention.fc_out.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].transpose(2, 1,
                                                                                                             0).reshape(
                768, 768))
        block.attention.fc_out.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        block.ff.fc1.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'].T)
        block.ff.fc1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        block.ff.fc2.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'].T)
        block.ff.fc2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

    for i, block in enumerate(model.gate_l4.blocks):
        block.norm1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        block.norm1.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])

        block.norm2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
        block.norm2.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])

        block.attention.query.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.query.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].flatten())
        block.attention.key.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].transpose(1, 0,
                                                                                                             2).reshape(
                768, -1))
        block.attention.key.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].flatten())
        block.attention.value.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.value.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].flatten())
        block.attention.fc_out.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].transpose(2, 1,
                                                                                                             0).reshape(
                768, 768))
        block.attention.fc_out.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        block.ff.fc1.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'].T)
        block.ff.fc1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        block.ff.fc2.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'].T)
        block.ff.fc2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

    model.branch1_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    model.branch1_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    model.branch2_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    model.branch2_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    model.branch3_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    model.branch3_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    model.gate_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    model.gate_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    # model.pre_logits.weight.data = torch.from_numpy(npz_weights['pre_logits/kernel'])
    # model.pre_logits.bias.data = torch.from_numpy(npz_weights['pre_logits/bias'])

    # model.head.weight.data = torch.from_numpy(npz_weights['head/kernel'].T)
    # model.head.bias.data = torch.from_numpy(npz_weights['head/bias'])

def load_4b_2v(model, weight_path="imagenet21k_ViT-B_8.npz"):
    # 加载预训练的权重
    npz_weights = np.load(weight_path)

    model.patchemb_l3.patch_embed.weight.data = torch.from_numpy(npz_weights['embedding/kernel'].transpose(3, 2, 0, 1))
    model.patchemb_l3.patch_embed.bias.data = torch.from_numpy(npz_weights['embedding/bias'])
    # model.patchemb_l3.pos_embedding.data = torch.from_numpy(npz_weights['Transformer/posembed_input/pos_embedding'])
    model.patchemb_l3.cls_token.data = torch.from_numpy(npz_weights['cls'])


    for i, block in enumerate(model.branch1_l4.blocks):
        block.norm1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        block.norm1.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])

        block.norm2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
        block.norm2.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])

        block.attention.query.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.query.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].flatten())
        block.attention.key.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].transpose(1, 0,
                                                                                                             2).reshape(
                768, -1))
        block.attention.key.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].flatten())
        block.attention.value.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.value.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].flatten())
        block.attention.fc_out.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].transpose(2, 1,
                                                                                                             0).reshape(
                768, 768))
        block.attention.fc_out.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        block.ff.fc1.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'].T)
        block.ff.fc1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        block.ff.fc2.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'].T)
        block.ff.fc2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

    for i, block in enumerate(model.branch2_l4.blocks):
        block.norm1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
        block.norm1.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])

        block.norm2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
        block.norm2.weight.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])

        block.attention.query.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.query.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].flatten())
        block.attention.key.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].transpose(1, 0,
                                                                                                             2).reshape(
                768, -1))
        block.attention.key.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].flatten())
        block.attention.value.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].transpose(1, 0,
                                                                                                               2).reshape(
                768, -1))
        block.attention.value.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].flatten())
        block.attention.fc_out.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].transpose(2, 1,
                                                                                                             0).reshape(
                768, 768))
        block.attention.fc_out.bias.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'])
        block.ff.fc1.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'].T)
        block.ff.fc1.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
        block.ff.fc2.weight.data = torch.from_numpy(
            npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'].T)
        block.ff.fc2.bias.data = torch.from_numpy(npz_weights[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])

    model.branch1_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    model.branch1_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    model.branch2_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    model.branch2_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    # model.branch3_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    # model.branch3_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    # model.gate_l5.norm.bias.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/bias'])
    # model.gate_l5.norm.weight.data = torch.from_numpy(npz_weights['Transformer/encoder_norm/scale'])

    # model.pre_logits.weight.data = torch.from_numpy(npz_weights['pre_logits/kernel'])
    # model.pre_logits.bias.data = torch.from_numpy(npz_weights['pre_logits/bias'])

    # model.head.weight.data = torch.from_numpy(npz_weights['head/kernel'].T)
    # model.head.bias.data = torch.from_numpy(npz_weights['head/bias'])