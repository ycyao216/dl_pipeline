import torch
import torch.nn as nn
from zmq import device


class Nerf_block(nn.Module):
    """Nerf_block"""

    def __init__(self, in_dim, pose_enc_len=10, dir_enc_len=4, ppe_dim=60, pde_dim=24):
        super(Nerf_block, self).__init__()
        self.mlp_rou1 = nn.Sequential(
            nn.Linear(ppe_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.mlp_rou2 = nn.Sequential(
            nn.Linear(ppe_dim + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Linear(256, 257),
        )
        self.mlp_rgb = nn.Sequential(
            nn.Linear(257 + pde_dim, 128), nn.ReLU(), nn.Linear(128, 3), nn.Sigmoid()
        )
        self.pose_enc_len = pose_enc_len
        self.dir_enc_len = dir_enc_len

    def forward(self, x):
        """Forward

        Parameters
        ----------
        x : torch.tensor
            The input tensor of encoded position and direction of rays

        Returns
        -------
        torch.tensor
            RGB, rou
        """
        x_e, d_e = self.posenc(x)
        x = nn.ReLU()(x_e)
        x = torch.cat([self.mlp_rou1(x), x_e], dim=1)
        x = nn.ReLU(inplace=True)(x)
        x = torch.cat([self.mlp_rou2(x), d_e], dim=1)
        rou = x[:,-1:]
        x = nn.ReLU(inplace=True)(x)
        x = self.mlp_rgb(x)
        return torch.cat([x,rou],dim=-1)

    def positional_encoding(self, x):
        """Position encoding using frequency function sin and cos

        Parameters
        ----------
        x : torch.tensor
            The inpu batch [batch num, 6(:3 position, 3: direction)]

        Returns
        -------
        tuple   
            position encoding [ batch num, 3], direction encoding [batch num, 3]
        """
        x_position = x[:, :3]
        x_dir = x[:, 3:]
        indices_x_pose = torch.arange(0,self.pose_enc_len,dtype=torch.int, device=device)
        indices_x_dir = torch.arange(0,self.dir_enc_len,dtype=torch.int, device=device)
        coeffs_pose = torch.pi*torch.pow((torch.ones(self.pose_enc_len,device=device)*2), indices_x_pose)
        coeffs_dir = torch.pi*torch.pow((torch.ones(self.pose_enc_len,device=device)*2), indices_x_dir)
        sin_pos = torch.sin(coeffs_pose*x_position)
        cos_pos = torch.cos(coeffs_pose*x_position)
        pose_enc = torch.stack((sin_pos, cos_pos), dim=0).view(x.shape[0], 2*self.pose_enc_len).t().contiguous().view(x.shape[0], 2*self.pose_enc_len)
        sin_dir = torch.sin(coeffs_dir*x_dir)
        cos_dir = torch.cos(coeffs_dir*x_dir)
        dir_enc = torch.stack((sin_dir, cos_dir), dim=0).view(x.shape[0], 2*self.dir_enc_len).t().contiguous().view(x.shape[0], 2*self.dir_enc_len)
        return pose_enc, dir_enc


    def posenc(self,x):
        pos_x = x[:,:3]
        dir_x = x[:,3:]
        pose_enc, dir_enc = [],[]
        for i in range(self.pose_enc_len):
            for fn in [torch.sin, torch.cos]:
                pose_enc.append(fn(2.**i * torch.pi* pos_x))
                if i < self.dir_enc_len:
                    dir_enc.append(fn(2.**i * torch.pi* dir_x))
        return torch.cat(pose_enc,-1), torch.cat(dir_enc,-1)

class Leveled_nerf:
    """Some Information about Leveled_nerf"""
    def __init__(self,configs):
        in_dim = configs["model_spec"]["model_args"]["in_dim"]
        pose_enc_len=10
        dir_enc_len=4
        ppe_dim=60
        pde_dim=24
        super(Leveled_nerf, self).__init__()
        self.model1 = Nerf_block(in_dim, pose_enc_len, dir_enc_len, ppe_dim, pde_dim)
        self.model2 = Nerf_block(in_dim, pose_enc_len, dir_enc_len, ppe_dim, pde_dim)

    def get_model(self):
        return list([self.model1, self.model2])