2d_cdf
快速运行
运行以下代码即可启动：
bash
运行

python view_model.py

关于 2d 模型
模型文件：model.pth
输入输出说明

    输入维度：[batch_size, 4]
    4 个维度含义：2 维障碍物坐标 + 2 维机械臂角度
    输出维度：[batch, 3]
    3 个维度含义：2 维贴住障碍物的角度 + 1 维分类（表示障碍物内部 / 外部，即 CDF 正负，大于 0.5 为负）

CDF 值计算方式
求 CDF 值的公式：
python
运行

torch.norm(outputs - inputs, dim=-1)

再根据计算结果判断正负（大于 0.5 为负）