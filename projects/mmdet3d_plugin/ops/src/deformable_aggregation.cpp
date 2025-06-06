#include <torch/extension.h> // 包含PyTorch C++扩展所需的头文件
#include <c10/cuda/CUDAGuard.h> // 包含CUDA相关的头文件，如CUDAGuard，用于管理CUDA设备上下文

// 声明CUDA/C++实现的deformable_aggregation核心函数 (前向传播)
// 这个函数通常在 .cu 文件中定义，并由CUDA编译器(nvcc)编译
void deformable_aggregation(
  float* output, // 输出张量指针
  const float* mc_ms_feat, // 输入的多相机、多尺度特征图谱指针 (已格式化为柱状)
  const int* spatial_shape, // 每个特征图的空间形状 (H, W) 指针
  const int* scale_start_index, // 每个尺度特征在扁平化特征图中的起始索引指针
  const float* sample_location, // 采样点位置指针 (归一化坐标)
  const float* weights, // 采样点权重指针
  int batch_size, // 批量大小
  int num_cams, // 相机数量
  int num_feat, // 扁平化后特征的总点数 (所有H*W*C的总和，或者对于柱状格式是H*W的总和)
  int num_embeds, // 特征嵌入维度 (C)
  int num_scale, // 特征层级的数量
  int num_anchors, // 锚点/查询的数量
  int num_pts, // 每个锚点/查询的采样点数量
  int num_groups // 注意力头的数量或分组数量
);
  

/*
注释：描述输入输出张量的期望形状 (从Python端视角)
feat: (bs, num_feat_total_pixels, C) - 扁平化的多相机多尺度特征 (mc_ms_feat)
_spatial_shape: (num_cams, num_levels, 2) - 每个(相机,层级)特征图的(H,W)
_scale_start_index: (num_cams, num_levels) - 每个(相机,层级)特征图在扁平化特征中的起始像素索引 (相对于该相机的扁平化特征块)
_sampling_location: (bs, num_anchors, num_pts, num_cams, num_levels, num_groups, 2) 或类似结构，最终传递给CUDA核时可能是 (bs, num_anchors, num_pts, num_cams, 2) - 采样点在对应特征图上的归一化2D坐标
_weights: (bs, num_anchors, num_pts, num_cams, num_levels, num_groups) - 对应采样点的权重
output: (bs, num_anchors, C) - 聚合后的输出特征
kernel: (未使用在此上下文中)
*/


// PyTorch C++扩展的前向接口函数
at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat, // 输入的多相机、多尺度特征图谱 (来自Python)
  const at::Tensor &_spatial_shape, // 空间形状信息
  const at::Tensor &_scale_start_index, // 尺度起始索引
  const at::Tensor &_sampling_location, // 采样点位置
  const at::Tensor &_weights // 采样点权重
) {
  // at::DeviceGuard guard(_mc_ms_feat.device()); // 确保后续操作在正确的设备上执行 (例如，如果输入在GPU上，则输出也在GPU上)
  // const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat)); // 可选的CUDA设备保护

  // 从输入张量中获取维度信息
  int batch_size = _mc_ms_feat.size(0); // 批量大小
  int num_feat = _mc_ms_feat.size(1);   // 扁平化后特征的总点数 (所有H*W的总和)
  int num_embeds = _mc_ms_feat.size(2); // 特征嵌入维度 (C)
  int num_cams = _spatial_shape.size(0);   // 相机数量 (或相机组/FPN实例数量)
  int num_scale = _spatial_shape.size(1);  // 每个相机/FPN实例的特征层级数量
  int num_anchors = _sampling_location.size(1); // 锚点/查询的数量
  int num_pts = _sampling_location.size(2);     // 每个锚点的采样点数量
  int num_groups = _weights.size(5); // 注意力头的数量或分组数量

  // 获取指向张量数据的原始指针
  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  // 创建输出张量，形状为 (batch_size, num_anchors, num_embeds)，设备和数据类型与输入特征相同
  auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());

  // 调用核心的deformable_aggregation函数 (通常是CUDA实现)
  deformable_aggregation(
    output.data_ptr<float>(), // 输出张量的数据指针
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, // 输入数据指针
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups // 维度信息
  );
  return output; // 返回计算得到的输出张量
}


// 声明CUDA/C++实现的deformable_aggregation核心函数 (反向传播)
void deformable_aggregation_grad(
  const float* mc_ms_feat, // 前向传播时的输入特征图谱
  const int* spatial_shape, // 空间形状信息
  const int* scale_start_index, // 尺度起始索引
  const float* sample_location, // 采样点位置
  const float* weights, // 采样点权重
  const float* grad_output, // 输出张量的梯度 (来自上一层)
  float* grad_mc_ms_feat, // 计算得到的对输入特征图谱的梯度
  float* grad_sampling_location, // 计算得到的对采样点位置的梯度
  float* grad_weights, // 计算得到的对采样点权重的梯度
  int batch_size, // 批量大小
  int num_cams, // 相机数量
  int num_feat, // 扁平化特征总点数
  int num_embeds, // 特征嵌入维度
  int num_scale, // 特征层级数量
  int num_anchors, // 锚点/查询数量
  int num_pts, // 每个锚点的采样点数量
  int num_groups // 分组数量
);


// PyTorch C++扩展的反向接口函数
void deformable_aggregation_backward(
  const at::Tensor &_mc_ms_feat, // 前向传播时保存的输入特征图谱
  const at::Tensor &_spatial_shape, // 前向传播时保存的空间形状
  const at::Tensor &_scale_start_index, // 前向传播时保存的尺度起始索引
  const at::Tensor &_sampling_location, // 前向传播时保存的采样点位置
  const at::Tensor &_weights, // 前向传播时保存的采样点权重
  const at::Tensor &_grad_output, // 输出张量的梯度
  at::Tensor &_grad_mc_ms_feat, // 用于存储对输入特征图谱的梯度的张量 (预分配)
  at::Tensor &_grad_sampling_location, // 用于存储对采样点位置的梯度的张量 (预分配)
  at::Tensor &_grad_weights // 用于存储对采样点权重的梯度的张量 (预分配)
) {
  // at::DeviceGuard guard(_mc_ms_feat.device()); // 设备保护
  // const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));

  // 获取维度信息 (与前向传播中类似)
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  // 获取指向张量数据的原始指针
  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();
  const float* grad_output = _grad_output.data_ptr<float>(); // 输出的梯度

  // 获取指向预分配的梯度张量的数据指针
  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights = _grad_weights.data_ptr<float>();

  // 调用核心的deformable_aggregation_grad函数 (通常是CUDA实现) 来计算梯度
  deformable_aggregation_grad(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { // 使用Pybind11定义Python模块接口
  // TORCH_EXTENSION_NAME 是一个宏，通常在setup.py中定义，表示编译后的模块名称
  // m 是模块对象
  m.def( // 定义一个名为 "deformable_aggregation_forward" 的函数，它在Python中对应C++的 deformable_aggregation_forward 函数
    "deformable_aggregation_forward", // Python中调用的函数名
    &deformable_aggregation_forward, // C++中对应的函数指针
    "deformable_aggregation_forward" // 函数的文档字符串 (可选)
  );
  m.def( // 定义一个名为 "deformable_aggregation_backward" 的函数
    "deformable_aggregation_backward", // Python中调用的函数名
    &deformable_aggregation_backward, // C++中对应的函数指针
    "deformable_aggregation_backward" // 函数的文档字符串 (可选)
  );
}
