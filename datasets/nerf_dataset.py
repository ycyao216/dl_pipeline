from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, Dataset
import os 
import torchvision.transforms as T
import torch
import PIL.Image as Image 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Nerf_dataset(Dataset):
    """Some Information about Nerf_dataset"""
    def __init__(self, config, train=0):
        super(Nerf_dataset, self).__init__()
        dirs = [config["meta"]["training_dir_name"],
               config["meta"]["validating_dir_name"],
               config["meta"]["testing_dir_name"]]
        self.dirs = dirs
        self.pose_dir = os.path.join(dirs[train],"pose")
        self.rgb_dir = os.path.join(dirs[train],"rgb")
        self.intrinsic_file = os.path.join(dirs[train], "intrinsics.txt")
        self.key_words = ["train","val","test"]
        self.images = self.read_images(train)
        self.intrinsic = self.read_intrinsics()
        self.extrinsics = self.read_extrinsics(train)
        self.focal = self.intrinsic[0,0]
        #@NOTE: lazy option
        self.size = 800
        
    def __getitem__(self, index):
        """
        This implementation should only be used with a batch size of 1.
        """
        rays_o, rays_d = self.get_rays(self.size, self.size, self.focal, self.extrinsics[index])
        rays_o = rays_o.reshape((-1,3))
        rays_d = rays_d.reshape((-1,3))
        image_pixels = self.images[index].to(device)
        image_pixels = image_pixels.permute((1,2,0))[...,:3].reshape(-1,3)
        output = torch.cat([rays_o, rays_d, image_pixels],dim=-1)
        randid = torch.randperm(output.shape[0])
        output = output[randid]
        rays = output[...,:6]
        pixel = output[...,6:]
        return rays.to(device)[:6400,...], pixel.to(device)[:6400,...]

    def __len__(self):
        return len(self.images)

    def get_rays(self, H, W, focal, c2w):
        """Get ray origins, directions from a pinhole camera."""
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                        torch.arange(H, dtype=torch.float32), indexing='xy')
        i = i.to(device)
        j = j.to(device)
        dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim =-1)
        rays_o = torch.broadcast_to(c2w[:3, -1].to(device), rays_d.shape)
        return rays_o, rays_d

    def read_images(self,train):
        images = []
        if train == 2:
            train = 1
        for i in os.listdir(self.rgb_dir):
            if self.key_words[train] == i.split("_")[1]:
                name = os.path.join(self.rgb_dir,i)
                images.append(T.ToTensor()(Image.open(name)))
        return images 

    def read_intrinsics(self):
        with open(self.intrinsic_file,"r") as in_file:
            intrinsics = []
            lines = in_file.readlines()
            for i in lines:
                for j in i.split(" "):
                    intrinsics.append(float(j))
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device).reshape(3,3)
        return intrinsics

    def extrinsic_handler(self, file_name):
        extrinsic = []
        with open(file_name,"r") as in_file:
            lines = in_file.readlines()
            for i in lines:
                for j in i.split(" "):
                    extrinsic.append(float(j))
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32, device=device).reshape(4,4)
        return extrinsic

    def read_extrinsics(self,train):
        extrinsics = []
        for i in os.listdir(self.pose_dir):
            if self.key_words[train] == i.split("_")[1]:
                name = os.path.join(self.pose_dir,i)
                extrinsics.append(self.extrinsic_handler(name))
        return extrinsics 

