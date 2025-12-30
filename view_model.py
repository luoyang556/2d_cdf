import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# from mlp import MLPRegression

# import time 

# import robot_plot2D
# import os 
PI = math.pi



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
obj = torch.tensor([-1.5, 1.5]).to(device)

obj_r = 0.3


model_dir1 = "model.pth"
# model_dir1 = "./model.pth"

model = torch.load(model_dir1, weights_only=False)
model.eval()



t = np.linspace(-math.pi,math.pi, 50)
q0,q1 = np.meshgrid(t,t)
q0_torch = torch.from_numpy(q0).float()
q1_torch = torch.from_numpy(q1).float()
Q_sets = torch.cat([q0_torch.unsqueeze(-1),q1_torch.unsqueeze(-1)],dim=-1).view(-1,2).to(device)

p  = torch.ones_like(Q_sets)

p  = p*obj
print(p[:2, :])


print(p.shape)

# r = torch.full((p.shape[0], 1), obj_r, dtype=torch.float).to(device)
# r = torch.ones((p.shape[0], 1))*0.3.to(device)

inputs = torch.cat([p, Q_sets], dim = -1).to(device)


outputs = model(inputs)

proj = outputs[:, :2]

threshold = 0.5
mask = outputs[:, 2]
mask = mask>threshold

print(mask[:5])
d = torch.norm(Q_sets - proj, dim=-1)

d[mask] = -d[mask]
d_outputs = d.detach().cpu().numpy()

fig1, ax = plt.subplots(figsize=(10,8)) 
ax.set_aspect('equal', 'box')  # Make sure the pixels are square
ax.set_title('cdf', size=30)  # Add a title to your plot
ax.set_xlabel('q1', size=20)
ax.set_ylabel('q2', size=20)
axis_limits = (-PI, PI)  # Set the limits for both axes to be the same
ax.set_xlim(axis_limits)
ax.set_ylim(axis_limits)
ax.tick_params(axis='both', labelsize=20)
ax.contour(q0, q1, d_outputs.reshape(50, 50), levels=[0], linewidths=6, colors='black', alpha=1.0)
ct = ax.contourf(q0, q1, d_outputs.reshape(50, 50), levels=[-10, 0, 0.3, 0.6, 0.9, 1.5, 5.0], linewidths=1, cmap='coolwarm')
ax.clabel(ct, inline=False, fontsize=15, colors='black', fmt='%.1f')


plt.show()
    








