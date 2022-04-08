from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


class Multi_self_attn(nn.Module):
    """Multi-head self-attention module of the transformer with residual link"""

    def __init__(self, head_number, encoding_size, drop_out_rate):
        super(Multi_self_attn, self).__init__()
        self.encoding_size = torch.tensor(encoding_size)
        self.combined_weight = nn.Linear(encoding_size, encoding_size * head_number * 3)
        self.soft_max = nn.Softmax(dim=-1)
        self.head_number = head_number
        self.map_to_size = nn.Linear(encoding_size * head_number, encoding_size)
        self.norm = nn.LayerNorm(encoding_size)
        self.merge_attention = nn.Linear(encoding_size * head_number, encoding_size)
        self.dropout = nn.Dropout(p=drop_out_rate)

    def forward(self, x):
        normed_x = self.norm(x)
        combined_qkv = self.combined_weight(normed_x)
        combined_qkv = torch.chunk(combined_qkv, 3, dim=-1)
        query, key, value = combined_qkv
        # using N and D as shwon in the ViT paper
        query = rearrange(
            query, "batch N (heads D) -> batch heads N D", heads=self.head_number
        )
        key = rearrange(
            key, "batch N (heads D) -> batch heads N D", heads=self.head_number
        )
        value = rearrange(
            value, "batch N (heads D) -> batch heads N D", heads=self.head_number
        )
        attention_val = self.soft_max(
            query @ key.transpose(-2, -1) / (torch.sqrt(self.encoding_size))
        )
        output = attention_val @ value
        output = rearrange(output, "batch heads N D -> batch N (heads D)")
        output = self.merge_attention(output)
        output = self.dropout(output)
        x = output + x
        return x


class Feed_forward(nn.Module):
    """Feed forward compoent of the transformer with residual link"""

    def __init__(self, encoding_dimension, mlp_dimension, drop_out_rate):
        super(Feed_forward, self).__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(encoding_dimension, mlp_dimension),
            nn.Dropout(p=drop_out_rate),
            nn.GELU(),
            nn.Linear(mlp_dimension, encoding_dimension),
            nn.Dropout(p=drop_out_rate),
        )
        self.norm = nn.LayerNorm(encoding_dimension)

    def forward(self, x):
        x = self.linear_layer(self.norm(x)) + x
        return x


class Transformer_unit(nn.Module):
    """A single transformer unit"""

    def __init__(self, head_number, encoding_size, mlp_dimension, drop_out_rate):
        super(Transformer_unit, self).__init__()
        self.attention = Multi_self_attn(head_number, encoding_size, drop_out_rate)
        self.ffn = Feed_forward(encoding_size, mlp_dimension, drop_out_rate)

    def forward(self, x):
        x = self.ffn(self.attention(x))
        return x

class Vit_custom(nn.Module):
    """Some Information about Vit_custom"""

    def __init__(
        self,
        config
    ):
        super(Vit_custom, self).__init__()
        transformer_heads = config["model_spec"]["model_args"]["transformer_heads"]
        mlp_dimension = config["model_spec"]["model_args"]["mlp_dimension"]
        class_num = config["model_spec"]["model_args"]["class_num"]
        self.image_size = config["model_spec"]["model_args"]["image_size"]
        self.patch_size = config["model_spec"]["model_args"]["patch_size"]
        self.N = self.image_size * self.image_size // (self.patch_size ** 2)
        self.D = config["model_spec"]["model_args"]["encoding_dim"]
        self.pose_embedding = nn.Parameter(torch.randn((1, self.N + 1, self.D)))
        self.class_token = nn.Parameter(torch.randn((1, self.D)))
        self.embedding_layer = nn.Linear(self.patch_size * self.patch_size * 3, self.D)
        self.drop_out_rate= config["model_spec"]["model_args"]["drop_out_rate"]

        transformers = []
        for i in range(config["model_spec"]["model_args"]["trans_layer_count"]):
            transformers.append(
                Transformer_unit(
                    transformer_heads, self.D, mlp_dimension, self.drop_out_rate
                )
            )
        self.transformers = nn.Sequential(*transformers)
        self.MLP_head = nn.Linear(self.D, class_num)
        self.embed_dropout = nn.Dropout(p=self.drop_out_rate)
        self.flatten_to_patches = Rearrange(
            "batch chan (p_per_w patch1) (p_per_h patch2) -> batch (p_per_w p_per_h) (patch1 patch2 chan)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )

    def forward(self, x):
        patches = self.flatten_to_patches(x)
        embedding = self.embedding_layer(patches)
        class_token_for_batch = repeat(
            self.class_token, "one D -> batch one D", batch=x.shape[0]
        )
        embedding = torch.cat([class_token_for_batch, embedding], dim=1)
        embedding = embedding + self.pose_embedding
        embedding = self.embed_dropout(embedding)
        class_output = self.transformers(embedding)[:, :1, :]
        class_output = self.MLP_head(class_output)
        return class_output.squeeze(1)
