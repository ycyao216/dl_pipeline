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
        self.pose_dir = os.path.join(dirs[train],"pose")
        self.rgb_dir = os.path.join(dirs[train],"rgb")
        self.intrinsic_file = os.path.join(dir[train], "intrinsics.txt")
        self.key_words = ["train","val","test"]
        self.images = self.read_images(train)
        self.intrinsic = self.read_intrinsics()
        self.extrinsics = self.read_extrinsics(train)
        self.focal = self.intrinsic[0,0]

    def __getitem__(self, index):
        #@TODO: Make flexible, temporarly set to 800 for the given dataset
        return 800,800,self.focal, self.extrinsics[index], self.intrinsic, self.images[index]

    def __len__(self):
        return len(self.images)

    def read_images(self,train):
        images = []
        for i in os.listdir(self.rgb_dir):
            if self.key_words[train] == i.split("_")[1]:
                name = os.path.join(self.rgb_dir,i)
                images.append(T.ToTensor()(Image.open(i)))
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

    def read_extrinsics(self,train):
        extrinsics = []
        for i in os.listdir(self.pose_dir):
            if self.key_words[train] == i.split("_")[1]:
                name = os.path.join(self.rgb_dir,i)
                extrinsic = extrinsic.append(self.extrinsic_handler(name))
        return extrinsics 

    def extrinsic_handler(self, file_name):
        extrinsic = []
        with open(file_name,"r") as in_file:
            lines = in_file.readlines()
            for i in lines:
                for j in i.split(" "):
                    extrinsic.append(float(j))
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32, device=device).reshape(4,4)
        return extrinsic