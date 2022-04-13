from tkinter import N
import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class segmentation_visualizer:
    def __init__(self, model, test_loader):
        self.model = model
        self.model.eval()
        self.test_loader = test_loader
        self.colormap = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        )

    def visualize(self):
        softmax = nn.Softmax()
        with torch.no_grad():
            for batch, _ in self.test_loader:
                for i in batch:
                    i = i.to(device)
                    output = self.model(i).squeeze(0)
                    output = softmax(output)
                    output = torch.argmax(output, dim=1).reshape((-1, 1)).numpy()
                    points = i.cpu().detach().numpy()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    colors = self.colormap[output]
                    pcd.colors = colors
                    o3d.visualization.draw_geometries([pcd])
