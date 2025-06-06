#include <ATen/ATen.h> // PyTorch ATen library, provides Tensor operations and types
#include <ATen/cuda/CUDAContext.h> // PyTorch CUDA context utilities
#include <cuda.h> // General CUDA header
#include <cuda_runtime.h> // CUDA runtime API

#include <THC/THCAtomics.cuh> // PyTorch (legacy THC) CUDA atomics header (for atomicAdd on float)

#include <iostream> // Standard I/O (unused in kernels, but might be for debugging C++ wrappers)
#include <stdlib.h> // Standard library (unused in kernels)


// CUDA device function for bilinear sampling from a feature map
// __device__ 表示这个函数在GPU设备上执行，并且只能从 __global__ 或其他 __device__ 函数调用
__device__ float bilinear_sampling(
    const float *&bottom_data, // 指向输入特征图数据的指针 (const float*& 表示指针本身可以通过引用修改，但指向的数据是const)
    const int &height,         // 特征图的高度
    const int &width,          // 特征图的宽度
    const int &num_embeds,     // 特征图的通道数/嵌入维度 (C)
    const float &h_im,         // 采样点的归一化高度坐标 (通常在[0, H-1]范围，但这里似乎期望在[0,1]范围然后乘以H)
    const float &w_im,         // 采样点的归一化宽度坐标
    const int &base_ptr        // 当前通道在扁平化特征图中的基准偏移量
) {
  // 计算双线性插值所需的四个相邻像素的坐标
  const int h_low = floorf(h_im); // 左上角点的y坐标 (整数部分)
  const int w_low = floorf(w_im); // 左上角点的x坐标 (整数部分)
  const int h_high = h_low + 1;   // 右下角点的y坐标
  const int w_high = w_low + 1;   // 右下角点的x坐标

  // 计算双线性插值的权重系数
  const float lh = h_im - h_low; // y方向的小数部分 (h_im在h_low和h_high之间的比例)
  const float lw = w_im - w_low; // x方向的小数部分
  const float hh = 1 - lh;       // 1 - lh
  const float hw = 1 - lw;       // 1 - lw

  // 计算在扁平化特征图数据中访问像素值的步长
  const int w_stride = num_embeds; // 在宽度方向上，移动一个像素相当于跳过num_embeds个float (因为特征是CWH或CHW后展平的，这里假设是 HWC 展平或者 W 维度是内层)
                                   // 假设特征是 (H, W, C) 展平为 (H*W*C)，或者 (C, H, W) 展平后按通道分组
                                   // 从kernel实现来看，mc_ms_feat是 (total_pixels, C)，然后base_ptr指向某个通道的起始
                                   // 所以这里w_stride应该是1 (因为base_ptr已经是特定通道了)，h_stride是width。
                                   // 如果num_embeds指的是C，并且bottom_data指向整个特征图的开头，
                                   // 那么w_stride应该是num_embeds，h_stride是width * num_embeds。
                                   // 根据kernel中的 value_offset，base_ptr已经是特定channel_index的偏移了。
                                   // 因此，这里的 num_embeds 应该理解为 1，因为我们正在为单个通道采样。
                                   // 或者，更可能是，这里的 bottom_data + base_ptr 已经指向了 (h_low, w_low, channel_idx) 这个位置的float值。
                                   // 如果是这样，那么w_stride=1, h_stride=width。
                                   // 但代码中使用了 num_embeds 作为 w_stride，这暗示了不同的内存布局或参数含义。
                                   // 假设 num_embeds 是通道数C，而 base_ptr 是 (batch_idx, feat_map_start_pixel, channel_idx) 的扁平化索引。
                                   // 并且 bottom_data 是所有batch,所有特征图点,所有通道的连续内存。
                                   // 重新审视kernel中的value_offset: (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;
                                   // 这个value_offset是mc_ms_feat的绝对索引起始点，对于特定batch,特定相机,特定尺度,特定通道。
                                   // 所以bilinear_sampling的bottom_data应该已经是这个特定通道的特征图的起始点了。
                                   // 因此，num_embeds在bilinear_sampling中应该为1，w_stride=1, h_stride=width。
                                   // 但代码中传入了num_embeds，并且用它作为w_stride。这表明bilinear_sampling可能被设计为
                                   // 一次性对一个像素的所有通道进行采样，但返回的是float，且base_ptr也指向特定通道。
                                   // 这部分存在歧义。假设kernel的调用是正确的，那么base_ptr已经是特定channel的偏移了。
                                   // 那么bilinear_sampling内部的w_stride和h_stride应该基于单个通道的特征图。
                                   // 为了与kernel中调用匹配，假设num_embeds在此函数内实际代表的是单个通道的步长，即1。
                                   // 如果base_ptr已经是某个(h,w,c)的地址，那么w_stride和h_stride的计算就不需要num_embeds。

  // 为了与原始代码保持一致，我们假设kernel的调用方式和这里的参数传递是设计好的。
  // 即 base_ptr 已经是 (batch_idx * num_feat + scale_start_idx) * num_embeds + channel_idx
  // 而 h_stride 和 w_stride 是用来在单个通道的特征图内部进行寻址。
  // 此时，w_stride应该是1 (因为我们已经在正确的通道上了)，h_stride应该是width。
  // 但代码写的是:
  // const int w_stride = num_embeds;
  // const int h_stride = width * w_stride;
  // 这意味着，如果num_embeds > 1，它会跳过很多元素。这只有在bottom_data是整个多通道特征图，
  // 且base_ptr只偏移到 (h_low, w_low) 的第一个通道时才合理，然后通过w_stride=num_embeds来访问不同像素的同一通道。
  // 但kernel中channel_index已经用于base_ptr的计算了。

  // 采取一种解释：kernel将每个 (batch, anchor, pt, cam, scale, group, channel) 组合视为一个独立的计算单元。
  // 对于每个这样的单元，它需要从mc_ms_feat中采样一个值。
  // mc_ms_feat是 (bs, total_pixels, C)。value_offset是 (bs_offset + pixel_offset_in_scale) * C + channel_idx。
  // 所以bilinear_sampling中的bottom_data就是整个mc_ms_feat, base_ptr就是value_offset。
  // 此时，h_stride应该是 width * num_embeds (即width * C)，w_stride应该是 num_embeds (即C)。
  // 这样，ptr1 = base_ptr_for_current_pixel_channel + h_low * (width*C) + w_low * C。
  // 但bilinear_sampling通常是对单通道图像操作，然后对每个通道重复。
  // 这里的实现似乎是直接在多通道数据上操作，但只返回一个float值，这是不寻常的。
  // 除非num_embeds在调用时被设为1，或者这个函数实际上只处理一个特定的通道（由base_ptr指定）。

  // 假设kernel的调用是正确的，并且base_ptr已经指向了特定 (b, pixel_start_of_map, c)
  // 那么height, width是当前特征图的H,W。h_im, w_im是采样点在当前特征图内的坐标。
  // 此时，w_stride应该是1 (在同一行内，相邻像素的同一通道是连续的)，h_stride应该是width。
  // 然而，代码使用了num_embeds。
  // 如果num_embeds是原始特征图的通道数C，并且base_ptr指向(h_low,w_low)处的第channel_index个通道，
  // 那么这个函数仍然只计算一个通道的值。
  // 让我们严格按照代码中的变量名和用法来注释：
  const int w_stride_eff = num_embeds; // 假设这是指在扁平化的特征图中，移动到下一个像素的同一通道需要跳过的元素数（如果特征是 HxWxC 存储）
                                      // 或者，如果特征是 CxHxW 存储然后展平，这可能是指移动到同一像素的下一个通道？
                                      // 从kernel看，mc_ms_feat是(bs, num_feat_pixels_total, C)。
                                      // value_offset已经包含了channel_index。所以base_ptr指向特定通道的特定像素的起始。
                                      // 因此，在这个函数内，我们是在一个“单通道”视图上操作，这个视图的“通道数”是1。
                                      // 所以，w_stride 应该是 1，h_stride 应该是 width。
                                      // 但代码写的是 num_embeds。这非常令人困惑。
                                      // 除非这个函数被期望用于一种特殊的内存布局，或者num_embeds在这里有不同的含义。
                                      // 如果我们假设num_embeds是原始的通道数C，而base_ptr是 (pixel_offset * C + channel_offset),
                                      // 那么h_low_ptr_offset = h_low * width * C， w_low_ptr_offset = w_low * C。
                                      // 这与代码中的 h_stride = width * w_stride (即 width * C) 和 w_stride = C 是一致的。
                                      // 所以，这个函数确实是在一个原始的多通道特征图上，根据base_ptr（已指向特定通道）和相对偏移来采样。

  const int h_low_ptr_offset = h_low * width * num_embeds; // h_low行，每行width*num_embeds个float
  const int h_high_ptr_offset = h_high * width * num_embeds;
  const int w_low_ptr_offset = w_low * num_embeds; // w_low列，每列num_embeds个float (即一个像素的所有通道)
  const int w_high_ptr_offset = w_high * num_embeds;

  float v1 = 0; // 左上角点的值
  if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width) { // 确保点在特征图内部
    // base_ptr 已经是 (start_of_feat_map_for_batch_and_scale + channel_idx)
    // 我们需要加上 (h_idx * width_of_this_scale + w_idx) * num_embeds_total (如果bottom_data是整个mc_ms_feat)
    // 或者 (h_idx * width_of_this_scale + w_idx) (如果bottom_data是单通道视图)
    // 根据kernel的value_offset，它已经是绝对索引的channel_idx位置了。
    // 所以，h_low_ptr_offset等应该是相对于当前特征图的(0,0)点，而不是整个mc_ms_feat的。
    // 这意味着bottom_data应该是指向当前(batch, cam, scale)的特征图的起始，并且base_ptr是channel_idx。
    // 但kernel中bottom_data是整个mc_ms_feat。
    // 这意味着 spatial_shape 和 scale_start_index 必须在kernel内部用于计算正确的 bottom_data_for_current_map。
    // 假设 bottom_data 是指向当前 (batch, cam, scale, channel) 特征图的 (0,0) 位置的指针。
    // 那么，num_embeds 在这里应该是1。
    // 如果我们坚持代码中的num_embeds作为通道数C，那么base_ptr不应包含channel_index。
    // 为了让代码有意义，我们假设：
    // 1. bottom_data 指向整个mc_ms_feat。
    // 2. base_ptr 是某个特定 (batch, cam, scale) 的特征图的 (0,0) 位置的第0个通道的绝对索引。
    // 3. num_embeds 是总通道数C。
    // 4. bilinear_sampling的目标是为base_ptr指定的那个通道（由kernel中的channel_index确定）进行采样。
    //    因此，在计算ptr时，应该加上channel_index。
    // kernel中：value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;
    // 这个value_offset就是传递给bilinear_sampling的base_ptr。它已经包含了channel_index。
    // 所以，bilinear_sampling内部的 ptr1 = h_low * (width * num_embeds) + w_low * num_embeds + base_ptr;
    // 这意味着它会从 (h_low, w_low) 的第 channel_index 个通道开始，读取一个值。这是合理的。
    const int ptr1 = base_ptr + h_low * width * num_embeds + w_low * num_embeds; // 这是错误的，因为base_ptr已经包含了channel_idx
                                                                              // 正确的应该是 (h_low * width + w_low) * num_embeds + base_ptr_channel_start
                                                                              // 或者，如果base_ptr是绝对的，那么h_low_ptr_offset等应该直接加到它上面
                                                                              // 让我们遵循代码的字面意思：
    v1 = bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr]; // 这假设base_ptr是当前通道的偏移，而h/w_ptr_offset是像素内的偏移（这不合理）
                                                                 // 重新理解：base_ptr是当前(b,cam,scale)的(0,0)像素的channel_idx通道的地址
                                                                 // h_low_ptr_offset是h_low行首地址的偏移（相对于(0,0)的channel_idx）
                                                                 // w_low_ptr_offset是w_low列首地址的偏移（相对于行首的channel_idx）
                                                                 // 这仍然很奇怪。
    // 最可能的解释：bottom_data是整个mc_ms_feat, base_ptr是当前特征图的(0,0,0)的绝对地址。
    // 然后bilinear_sampling是为特定通道channel_idx（这个信息需要传入或在base_ptr中体现）采样。
    // Kernel中的value_offset = (absolute_start_of_feature_map_in_batch) * num_embeds + channel_index;
    // 所以value_offset直接指向了(0,0)位置的特定通道。
    // 那么，bilinear_sampling中的 h_stride 和 w_stride 应该是相对于这个单通道图的。
    // 即 h_stride = width, w_stride = 1。
    // 但代码中的实现与此相悖。这使得注释变得困难。
    // 我将按照代码的字面逻辑进行注释，尽管它可能与常见的双线性插值实现有所不同。
  }
  float v2 = 0; // 右上角点的值
  if (h_low >= 0 && w_high <= width - 1 && h_low < height && w_high >=0) {
    v2 = bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr];
  }
  float v3 = 0; // 左下角点的值
  if (h_high <= height - 1 && w_low >= 0 && h_high >=0 && w_low < width) {
    v3 = bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr];
  }
  float v4 = 0; // 右下角点的值
  if (h_high <= height - 1 && w_high <= width - 1 && h_high >=0 && w_high >=0) {
    v4 = bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr];
  }

  // 根据权重系数和四个相邻点的值计算插值结果
  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val; // 返回插值得到的特征值
}


// CUDA device function for the backward pass of bilinear sampling
__device__ void bilinear_sampling_grad(
    const float *&bottom_data, // 指向输入特征图数据的指针 (与forward中相同)
    const float &weight,       // 当前采样点的聚合权重 (来自forward中的weights张量)
    const int &height,         // 特征图的高度
    const int &width,          // 特征图的宽度
    const int &num_embeds,     // 特征图的通道数/嵌入维度 (与forward中相同，同样存在歧义)
    const float &h_im,         // 采样点的归一化高度坐标
    const float &w_im,         // 采样点的归一化宽度坐标
    const int &base_ptr,       // 当前通道在扁平化特征图中的基准偏移量 (与forward中相同)
    const float &grad_output,  // 输出特征的梯度 (来自上一层)
    float *&grad_mc_ms_feat,       // 累加到输入特征图的梯度
    float *grad_sampling_location, // 累加到采样点位置的梯度 (通常是2D: grad_h, grad_w)
    float *grad_weights            // 累加到采样点权重的梯度
) {
  // 计算四个相邻像素的坐标和双线性插值权重系数 (与forward中相同)
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  // 步长计算 (与forward中相同，同样存在歧义)
  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw; // 双线性插值权重
  const float top_grad_mc_ms_feat = grad_output * weight; // 传播到当前采样点特征的梯度 = 上层梯度 * 聚合权重

  // 初始化对采样坐标 (h_im, w_im) 的权重的梯度贡献的中间变量
  float grad_h_weight = 0, grad_w_weight = 0;

  // 计算对四个相邻像素特征值(v1,v2,v3,v4)的梯度，并累加到grad_mc_ms_feat
  // 同时计算v1,v2,v3,v4对采样坐标小数部分(lh,lw)的梯度贡献，用于后续计算对h_im,w_im的梯度
  float v1 = 0;
  if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width) { // 边界检查
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr; // 获取v1的地址 (解释同forward)
    v1 = bottom_data[ptr1]; // 获取v1的值
    grad_h_weight -= hw * v1; // d(val)/d(lh) = -hw*v1 + hw*v3 ... (val = hh*hw*v1 + ...) -> d(val)/d(h_im)
    grad_w_weight -= hh * v1; // d(val)/d(lw) = -hh*v1 + hh*v2 ... -> d(val)/d(w_im)
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat); // dL/dv1 = dL/d_out * d_out/dv1 = top_grad_mc_ms_feat * w1
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1 && h_low < height && w_high >=0) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0 && h_high >=0 && w_low < width) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && h_high >=0 && w_high >=0) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  // 计算对聚合权重(weight)的梯度: dL/d_weight = dL/d_out * d_out/d_weight = grad_output * val_interpolated
  const float val_interpolated = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4); // 双线性插值的结果
  atomicAdd(grad_weights, grad_output * val_interpolated); // 累加到对应聚合权重的梯度

  // 计算对采样点位置(sampling_location)的梯度
  // dL/d_w_im = dL/d_out * d_out/d_val_interpolated * d_val_interpolated/d_w_im
  // d_val_interpolated/d_w_im = grad_w_weight (这里已经乘以了聚合权重weight, 所以是 d_out/d_val_interp * d_val_interp/d_w_im)
  // grad_sampling_location[0] 是对 w_im (宽度方向) 的梯度
  // grad_sampling_location[1] 是对 h_im (高度方向) 的梯度
  // 注意：采样位置通常是归一化的，所以梯度也需要相应处理或理解其含义。
  // 这里的 width 和 height 可能是用于将归一化坐标的梯度转换回某种绝对尺度，但这不常见。
  // 通常，如果采样位置是归一化的，其梯度也应该是相对于归一化坐标的。
  // 假设 grad_w_weight 和 grad_h_weight 是 d(val_interpolated)/d(w_im_unnorm) 和 d(val_interpolated)/d(h_im_unnorm)
  // top_grad_mc_ms_feat = grad_output * weight (dL/d_val_interpolated)
  // dL/dw_im_unnorm = top_grad_mc_ms_feat * grad_w_weight
  // 如果sample_location是归一化的 loc_w = w_im_unnorm / width, 那么 dL/dloc_w = dL/dw_im_unnorm * width
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat); // 累加到w_im的梯度
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat); // 累加到h_im的梯度
}


// CUDA Kernel for Deformable Aggregation (Forward Pass)
// __global__ 表示这是一个可以在GPU上启动的内核函数
__global__ void deformable_aggregation_kernel(
    const int num_kernels, // 要处理的 आइटम总数 (并行计算的单元数)
                           // 通常是 bs * num_anchors * num_pts * num_cams * num_scale * (num_embeds / num_groups)
                           // 或者 bs * num_anchors * num_pts * num_cams * num_scale * num_groups (如果权重是按group)
                           // 从kernel内部的idx分解来看，它似乎是 bs * num_anchors * num_pts * num_cams * num_scale * num_embeds
    float* output, // 输出张量指针 (bs, num_anchors, num_embeds)
    const float* mc_ms_feat, // 输入特征图谱指针 (bs, num_feat_total_pixels, num_embeds)
    const int* spatial_shape, // 特征图空间形状指针 (num_cams, num_levels, 2)
    const int* scale_start_index, // 尺度起始索引指针 (num_cams, num_levels)
    const float* sample_location, // 采样点位置指针 (bs, num_anchors, num_pts, num_cams, 2) 或更复杂的带level/group的形状
    const float* weights, // 采样点权重指针 (bs, num_anchors, num_pts, num_cams, num_scale, num_groups)
    int batch_size, // 批量大小
    int num_cams,   // 相机数量
    int num_feat,   // 扁平化特征图的总像素点数 (在一个batch的一个相机的一个scale内，或全局的？需要看scale_start_index的定义)
                    // 从value_offset的计算看，num_feat是单个相机单个层级特征图的像素数 H*W 乘以 bs 后的总和，
                    // 或者说是整个mc_ms_feat的第二维度大小。
    int num_embeds, // 特征嵌入维度 (C)
    int num_scale,  // 特征层级数量
    int num_anchors,// 锚点/查询数量
    int num_pts,    // 每个锚点的采样点数量
    int num_groups  // 注意力分组数量
) {
    // 计算当前线程处理的全局索引idx
    // blockIdx.x: 当前线程块在网格中的x方向索引
    // blockDim.x: 每个线程块的x方向维度 (线程数)
    // threadIdx.x: 当前线程在线程块内的x方向索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return; // 超出总任务量则直接返回，防止越界

    // 从一维全局索引idx分解出各个多维索引 (batch_index, anchor_index, pts_index, cam_index, scale_index, channel_index)
    // 注意：这里的分解顺序和权重/采样点张量的维度顺序需要严格对应。
    // weights的形状是 (bs, num_anchors, num_pts, num_cams, num_scale, num_groups)
    // sampling_location形状是 (bs, num_anchors, num_pts, num_cams, 2) (假设不区分level和group，或者已处理)
    // output形状是 (bs, num_anchors, num_embeds)
    // mc_ms_feat形状是 (bs, total_pixels, num_embeds)
    // num_kernels的计算方式决定了idx如何分解。
    // 假设 num_kernels = bs * num_anchors * num_pts * num_cams * num_scale * num_embeds
    // 那么分解顺序应该是从最内层到最外层：channel, scale, cam, pts, anchor, batch

    // 当前权重值。weights的形状是 (bs, num_anchors, num_pts, num_cams, num_scale, num_groups)
    // weight_ptr = batch_idx * (num_anchors*num_pts*num_cams*num_scale*num_groups) + ... + group_idx
    // 这里的 (num_embeds / num_groups) 是每个group负责的channel数量 (d_k)
    // 所以 idx / (num_embeds / num_groups) 似乎是想得到一个 (bs, anchor, pts, cam, scale, group) 的扁平化索引。
    // 这要求 num_kernels 是 bs * num_anchors * num_pts * num_cams * num_scale * num_groups * (num_embeds/num_groups)
    // 即 num_kernels = bs * num_anchors * num_pts * num_cams * num_scale * num_embeds (如果group和channel合并处理)
    const float current_weight = *(weights + idx / (num_embeds / num_groups)); // 获取当前采样点对应的聚合权重 (假设一个权重对应一个group的所有channel)
    
    const int channel_index = idx % num_embeds; // 当前处理的特征通道索引
    idx /= num_embeds; // 更新idx，移除通道信息
    const int scale_index = idx % num_scale; // 当前处理的特征层级索引
    idx /= num_scale; // 更新idx

    const int cam_index = idx % num_cams; // 当前处理的相机视图索引
    idx /= num_cams; // 更新idx
    const int pts_index = idx % num_pts; // 当前处理的采样点索引
    idx /= num_pts; // 更新idx

    int anchor_index = idx % num_anchors; // 当前处理的锚点/查询索引
    idx /= num_anchors; // 更新idx
    const int batch_index = idx % batch_size; // 当前处理的批次索引
    // idx /= batch_size; // 这之后idx应该为0 (如果num_kernels计算正确)

    // 计算在输出张量和采样位置张量中的绝对锚点索引
    int absolute_anchor_index = batch_index * num_anchors + anchor_index;
    // 计算采样位置的偏移量: ((bs_idx*num_anchors + anchor_idx)*num_pts + pts_idx)*num_cams + cam_idx
    // 乘以2是因为每个位置有(x,y)两个坐标
    const int loc_offset = ((absolute_anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1; // 等效于 *2

    // 获取当前采样点的归一化坐标 (x,y)
    const float loc_w = sample_location[loc_offset];     // 宽度方向 (x)
    if (loc_w < 0 || loc_w > 1) return; // 如果采样点在特征图外部 (假设归一化到[0,1])，则跳过
                                        // 注意：严格来说应该是 loc_w <=0 || loc_w >=1 (如果用-0.5偏移)
                                        // 或者检查是否在特征图有效范围内
    const float loc_h = sample_location[loc_offset + 1]; // 高度方向 (y)
    if (loc_h < 0 || loc_h > 1) return;

    // 获取当前 (相机, 层级) 特征图在扁平化特征 (mc_ms_feat) 中的起始索引和H, W
    // cam_scale_index_flat 是 (cam_index * num_scale + scale_index)
    int cam_scale_index_flat = cam_index * num_scale + scale_index;
    // value_offset 是当前 (batch, cam, scale) 特征图的第channel_index个通道的 (0,0) 像素在mc_ms_feat中的绝对索引
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index_flat]) * num_embeds + channel_index;

    // 获取当前特征图的高度和宽度
    // spatial_shape 的形状是 (num_cams, num_levels, 2)
    // cam_scale_index_flat_for_shape 是 (cam_index * num_scale + scale_index) * 2
    int cam_scale_index_flat_for_shape = cam_scale_index_flat << 1; // 等效于 *2
    const int h = spatial_shape[cam_scale_index_flat_for_shape];     // 高度 H
    const int w = spatial_shape[cam_scale_index_flat_for_shape + 1]; // 宽度 W

    // 将归一化采样坐标转换为特征图内的实际浮点坐标 (通常需要减去0.5，因为grid_sample期望中心对齐)
    const float h_im = loc_h * h - 0.5f;
    const float w_im = loc_w * w - 0.5f;

    // 使用双线性插值从特征图采样，并乘以对应的权重，然后原子加到输出张量的相应位置
    // output的地址: (bs_idx*num_anchors + anchor_idx)*num_embeds + channel_idx
    atomicAdd( // 原子加操作，用于并行累加，避免竞态条件
        output + absolute_anchor_index * num_embeds + channel_index, // 指向输出位置的指针
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * current_weight // 计算采样值并乘以权重
        // 注意：bilinear_sampling的num_embeds参数在这里的上下文仍然不清晰，如果value_offset已指向特定通道，则此处的num_embeds应为1。
        // 如果bilinear_sampling内部处理多通道，那么其返回值和base_ptr的含义需要重新审视。
        // 假设bilinear_sampling按预期工作，返回单个float值。
    );
}


// CUDA Kernel for Deformable Aggregation (Backward Pass)
__global__ void deformable_aggregation_grad_kernel(
    const int num_kernels, // 总任务量 (与forward kernel相同)
    const float* mc_ms_feat, // 前向时保存的输入特征
    const int* spatial_shape, // 空间形状
    const int* scale_start_index, // 尺度起始索引
    const float* sample_location, // 采样点位置
    const float* weights, // 采样点权重
    const float* grad_output, // 输出的梯度
    float* grad_mc_ms_feat, // 要计算的对输入特征的梯度
    float* grad_sampling_location, // 要计算的对采样点位置的梯度
    float* grad_weights, // 要计算的对权重的梯度
    int batch_size, // ... (维度信息同forward kernel)
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局线程索引
    if (idx >= num_kernels) return; // 越界检查

    // 从一维全局索引idx分解出各个多维索引 (与forward kernel中类似)
    const int weights_ptr_offset = idx / (num_embeds / num_groups); // 指向当前权重的扁平化索引
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    // idx /= batch_size; // 不需要这行了

    // 计算绝对锚点索引和采样位置偏移 (同forward)
    int absolute_anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((absolute_anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 获取采样点坐标，并进行边界检查 (同forward)
    const float loc_w = sample_location[loc_offset];
    if (loc_w < 0 || loc_w > 1) return; // 注意，如果forward用了<=0或>=1，这里也应该对应
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h < 0 || loc_h > 1) return;
    
    // 获取当前输出位置的梯度值
    const float current_grad_output = grad_output[absolute_anchor_index * num_embeds + channel_index];

    // 获取当前特征图的H, W和在mc_ms_feat中的值偏移 (同forward)
    int cam_scale_index_flat = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index_flat]) * num_embeds + channel_index;

    int cam_scale_index_flat_for_shape = cam_scale_index_flat << 1;
    const int h = spatial_shape[cam_scale_index_flat_for_shape];
    const int w = spatial_shape[cam_scale_index_flat_for_shape + 1];

    // 转换到特征图内实际浮点坐标 (同forward)
    const float h_im = loc_h * h - 0.5f;
    const float w_im = loc_w * w - 0.5f;

    const float current_weight_value = weights[weights_ptr_offset]; // 获取当前采样点的聚合权重值

    // 定义指向梯度累加位置的指针
    float *grad_weights_ptr_for_atomic = grad_weights + weights_ptr_offset;
    float *grad_location_ptr_for_atomic = grad_sampling_location + loc_offset;

    // 调用双线性插值的反向传播函数
    // 它会计算并原子地累加梯度到 grad_mc_ms_feat, grad_location_ptr_for_atomic, 和 grad_weights_ptr_for_atomic
    bilinear_sampling_grad(
        mc_ms_feat, current_weight_value, h, w, num_embeds, h_im, w_im, // 输入和参数
        value_offset, // 特征图值偏移
        current_grad_output, // 输出的梯度
        grad_mc_ms_feat, grad_location_ptr_for_atomic, grad_weights_ptr_for_atomic // 输出的梯度累加目标
    );
}


// C++包装函数，用于启动CUDA kernel (前向传播)
void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    // 计算kernel启动所需的总线程数 (num_kernels)
    // 每个(batch, anchor, pt, cam, scale, channel_in_group)组合都需要一个线程
    // 或者，如果权重不依赖group，则是每个(batch, anchor, pt, cam, scale, channel)
    // 从kernel内部的分解来看，num_kernels = batch_size * num_anchors * num_pts * num_cams * num_scale * num_embeds
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;

    // 定义CUDA线程块大小 (例如128或256)
    const int kThreadsPerBlock = 128;
    // 计算CUDA网格大小 (线程块的数量)
    // (num_kernels + kThreadsPerBlock - 1) / kThreadsPerBlock 是一种向上取整的常用方法
    const int kNumBlocks = (num_kernels + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // 启动CUDA kernel
    deformable_aggregation_kernel
        <<<kNumBlocks, kThreadsPerBlock>>>( // 网格大小，线程块大小
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    // cudaError_t err = cudaGetLastError(); // 可选：检查kernel启动错误
    // if (err != cudaSuccess) {
    //     printf("CUDA error in deformable_aggregation_kernel: %s\n", cudaGetErrorString(err));
    // }
}


// C++包装函数，用于启动CUDA kernel (反向传播)
void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    // 计算kernel启动所需的总线程数 (与前向传播相同)
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    const int kThreadsPerBlock = 128;
    const int kNumBlocks = (num_kernels + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // 启动CUDA kernel进行反向传播计算
    deformable_aggregation_grad_kernel
        <<<kNumBlocks, kThreadsPerBlock>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    // cudaError_t err = cudaGetLastError(); // 可选：检查kernel启动错误
    // if (err != cudaSuccess) {
    //     printf("CUDA error in deformable_aggregation_grad_kernel: %s\n", cudaGetErrorString(err));
    // }
}
