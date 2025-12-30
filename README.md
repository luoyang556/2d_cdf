# 2d_cdf
2d_cdf

关于2d模型：
model.pth
输入：[batch_size, 4], 4 <- 2维障碍物坐标 + 2维机械臂角
输出：[batch, 3],      3 <- 2维贴住障碍物的角度 + 1维分类是障碍物内部还是外部即cdf正负（大于0.5为负）


求cdf值即：
torch.norm(outputs - inputs, dim=-1) 
然后给出正负
