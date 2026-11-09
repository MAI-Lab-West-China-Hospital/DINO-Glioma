import torch
import numpy as np
import torch.nn as nn

class DINOv3Encoder(nn.Module):
    def __init__(self, backbone, block_idxs_list, mod_keys=None):
        super().__init__()
        self.feature_model = backbone
        self.block_idxs_list = block_idxs_list
        self.embed_dim = self.feature_model.num_features
        default_keys = ("flair_3ch", "t1_3ch", "t1ce_3ch", "t2_3ch")
        if mod_keys is None:
            mod_keys = default_keys
        if not isinstance(mod_keys, (list, tuple)):
            raise ValueError("mod_keys must be an ordered list or tuple")
        mod_keys = tuple(mod_keys)
        self.mod_keys = mod_keys

    @staticmethod
    def reshape_features(F_1mod, B, D):
        reshaped_F = []

        for p, c in F_1mod:
            # p: (B*D, num_patches, dim)
            p = p.view(B, D, p.shape[1], p.shape[2]).contiguous()

            # c: (B*D, dim)
            c = c.view(B, D, c.shape[1]).contiguous()

            reshaped_F.append((p, c))

        return tuple(reshaped_F)

    def extract_features_per_mod(self, x_2d):
        return self.feature_model.get_intermediate_layers(
            x_2d, n=self.block_idxs_list, return_class_token=True
        )

    def forward(self, x):
        # mod_keys = ["flair_3ch", "t1_3ch", "t1ce_3ch", "t2_3ch"] 
        encoder_features = {}

        for key in self.mod_keys:
            x_mod=x[key] # B,C[3],D,H,W
            B, C, D, H, W = x_mod.shape
            x_2d = x_mod.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

            features = self.extract_features_per_mod(x_2d) 
            reshape_mod_features = self.reshape_features(features, B, D)
            encoder_features[key] = reshape_mod_features
                    
        return encoder_features

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes, 
            out_planes, 
            kernel_size=(1, 2, 2),  # (H, W, D)
            stride=(1, 2, 2),       # Stride matches kernel size
            padding=(0, 0, 0)       # No padding
        )

    def forward(self, x):
        return self.block(x)
    
class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        return self.block(x)

class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.GroupNorm(num_groups = 16, num_channels = out_planes),
            #nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.GroupNorm(num_groups = 16, num_channels = out_planes),
            #nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

def reshape_transformer_output_3d(z, embed_dim, h, w, d, patch_size):
    batch, D, num, dim = z.shape  
    num_patches = (h // patch_size) * (w // patch_size)
    if num == num_patches + 1: 
        z = z[:, :, 1:, :]  
    h_patches = h // patch_size 
    w_patches = w // patch_size 
    return z.permute(0, 3, 1, 2).reshape(batch, embed_dim, D, h_patches, w_patches) #num => h * w

class UNETRDecoder3D(nn.Module):
    def __init__(self, embed_dim, patch_size=16, input_dim=3, output_dim=3, ds=True):
        super().__init__()
        self.ds = ds
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim * input_dim, 512),
            Deconv3DBlock(512, 256),
            Deconv3DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim * input_dim, 512),
            Deconv3DBlock(512, 256)
        )

        self.decoder9 = Deconv3DBlock(embed_dim * input_dim, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim * input_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512),
            Conv3DBlock(512, 512),
            SingleDeconv3DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256),
            Conv3DBlock(256, 256),
            SingleDeconv3DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 128),
            SingleDeconv3DBlock(128, 64),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        ## z0:torch.Size([2, 128, 4, 224, 224]),z3/6/9/12:torch.Size([2, 128, 196, 3072])
        B, D, C, H, W = x.shape
        z0 = z0.permute(0,2,1,3,4) # torch.Size([2, 4, 128, 224, 224])
        z3 = reshape_transformer_output_3d(z3, self.embed_dim*C, H, W, D, self.patch_size)
        z6 = reshape_transformer_output_3d(z6, self.embed_dim*C, H, W, D, self.patch_size)
        z9 = reshape_transformer_output_3d(z9, self.embed_dim*C, H, W, D, self.patch_size)
        z12 = reshape_transformer_output_3d(z12, self.embed_dim*C, H, W, D, self.patch_size)
        # print(f"z0 : {z0.shape}, z3 : {z3.shape}, z6 : {z6.shape}, z9 : {z9.shape}, z12 : {z12.shape}")
        
        # Decoder operations
        z12 = self.decoder12_upsampler(z12)  # 512
        z9 = self.decoder9(z9)  # 512
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1)) 
        z6 = self.decoder6(z6)  # 256
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1)) 
        z3 = self.decoder3(z3)  # 128
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1)) 
        z0 = self.decoder0(z0)  # 64
        output = self.decoder0_header(torch.cat([z0, z3], dim=1)) 
        return output

