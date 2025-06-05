X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded  # 未解码的边界框参数索引：X坐标, Y坐标, Z坐标, 宽度, 长度, 高度, yaw角的正弦, yaw角的余弦, X方向速度, Y方向速度, Z方向速度
CNS, YNS = 0, 1  # centerness and yawness indices in quality  # 质量评估中的中心度和偏航度索引
YAW = 6  # decoded  # 解码后的边界框参数中的yaw角索引 (通常 SIN_YAW 和 COS_YAW 会被解码成 YAW)
