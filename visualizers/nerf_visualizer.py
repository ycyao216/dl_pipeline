import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from run_moduls.nerf_execute import render, render_path
import PIL.Image as Image 
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Nerf_visualizer:
    def __init__(self, model, test_loader):
        self.model = model
        self.model.eval()
        self.test_loader = test_loader

    def visualize(self,config):
        images = []
        img_save_dir = config["visualization"]["save_dir"]
        if not os.path.exists(img_save_dir):
            os.mkdir(img_save_dir)
        with torch.no_grad():
            for rays, _ in self.test_loader:
                rays_o, rays_d = rays 
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)
                rgb_output = render(rays_d, rays_o, model = self.model)
                images.append(rgb_output.cpu().detach().numpy())

        for index, img in enumerate(images): 
            image_pil = Image.fromarray(np.uint8(img))
            model_name = config["model_spec"]["name"]
            output_dir = os.path.join(img_save_dir,model_name)
            output_dir += ("_visual_"+str(index) +".PNG")
            image_result = image_pil.save(output_dir)
