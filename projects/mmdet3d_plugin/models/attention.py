import warnings  # 导入 warnings 模块，用于发出警告信息
import math  # 导入 math 模块，用于数学运算

import torch  # 导入 torch 模块，PyTorch 深度学习框架
import torch.nn as nn  # 导入 torch.nn 模块，神经网络模块
from torch.nn.functional import linear  # 从 torch.nn.functional 模块导入 linear 函数，用于线性变换
from torch.nn.init import xavier_uniform_, constant_  # 从 torch.nn.init 模块导入参数初始化方法

from mmcv.utils import deprecated_api_warning  # 从 mmcv.utils 模块导入废弃API警告装饰器
from mmcv.runner import auto_fp16  # 从 mmcv.runner 模块导入 auto_fp16 装饰器，用于自动混合精度训练
from mmcv.runner.base_module import BaseModule  # 从 mmcv.runner.base_module 模块导入 BaseModule 基类
from mmcv.cnn.bricks.drop import build_dropout  # 从 mmcv.cnn.bricks.drop 模块导入 build_dropout 函数，用于构建dropout层
from mmcv.cnn.bricks.registry import ATTENTION  # 从 mmcv.cnn.bricks.registry 模块导入 ATTENTION 注册表
import torch.utils.checkpoint as cp  # 导入 torch.utils.checkpoint 模块，用于梯度检查点 (节省显存)


from einops import rearrange  # 导入 einops 库中的 rearrange 函数，用于张量操作
try:
    # 尝试导入 flash_attn V1 接口
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_unpadded_kvpacked_func') # 打印使用的FlashAttention接口版本
except:
    # 如果V1导入失败，尝试导入 flash_attn V2 接口 (名称有所变化)
    from flash_attn.flash_attn_interface import  flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func
    print('Use flash_attn_varlen_kvpacked_func') # 打印使用的FlashAttention接口版本
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis # 从flash_attn导入处理padding的工具函数


def _in_projection_packed(q, k, v, w, b = None): # 定义一个内部辅助函数，用于将q, k, v分别进行线性投影
    """
    对输入的q, k, v张量使用同一个权重矩阵w（和偏置b，如果提供）的不同块进行线性投影。
    这常用于MultiHeadAttention中一次性计算Q, K, V的投影。

    Args:
        q (Tensor): 查询张量。
        k (Tensor): 键张量。
        v (Tensor): 值张量。
        w (Tensor): 包含q, k, v投影权重的组合权重矩阵，通常形状为 (3 * embed_dim, embed_dim)。
        b (Tensor, optional): 包含q, k, v投影偏置的组合偏置向量，形状为 (3 * embed_dim)。默认为None。

    Returns:
        tuple[Tensor, Tensor, Tensor]: 投影后的q_proj, k_proj, v_proj。
    """
    w_q, w_k, w_v = w.chunk(3)  # 将权重w沿第0维切分为3块，分别对应q, k, v的权重
    if b is None:  # 如果未提供偏置
        b_q = b_k = b_v = None  # q, k, v的偏置都为None
    else:
        b_q, b_k, b_v = b.chunk(3)  # 同样将偏置b切分为3块
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)  # 分别进行线性变换并返回


class FlashAttention(nn.Module):  # 定义FlashAttention模块，这是一个高效的注意力实现
    """Implement the scaled dot product attention with softmax. # 实现带有softmax的缩放点积注意力机制。
    Arguments # 参数说明
    ---------
        softmax_scale: The temperature to use for the softmax attention. # softmax的缩放因子。
                      (default: 1/sqrt(d_keys) where d_keys is computed at # （默认：1/sqrt(d_keys)，其中d_keys在运行时计算）
                      runtime)
        attention_dropout: The dropout rate to apply to the attention # 应用于注意力分数的dropout率。
                           (default: 0.1) # （默认：0.1）
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None): # 初始化函数
        super().__init__()
        self.softmax_scale = softmax_scale  # softmax缩放因子
        self.dropout_p = attention_dropout  # 注意力dropout概率
        self.fp16_enabled = True  # 标记是否启用fp16 (mmcv的auto_fp16会使用)

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)  # mmcv的装饰器，自动处理q和kv的fp16转换，输出转为fp32
    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        """Implements the multihead softmax attention. # 实现多头softmax注意力。
        Arguments # 参数说明
        ---------
            q: The tensor containing the query. (B, T, H, D) # 查询张量，形状为 (批量大小, 查询序列长度, 头数, 头维度)
            kv: The tensor containing the key, and value. (B, S, 2, H, D) # 键和值打包的张量，形状为 (批量大小, 键/值序列长度, 2, 头数, 头维度)，2表示k和v
            key_padding_mask: a bool tensor of shape (B, S) # 键的填充掩码，形状为 (批量大小, 键/值序列长度)，True表示该位置是padding
        """
        # 断言输入数据类型和设备满足FlashAttention的要求
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        # 断言查询和键/值的批量大小、头数、头维度一致
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]  # 批量大小
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]  # 查询序列长度和键/值序列长度

        if key_padding_mask is None:  # 如果没有提供key_padding_mask，表示所有序列都是有效长度
            # 将q和kv的批量和序列长度维度合并，以适应flash_attn_unpadded_kvpacked_func的输入格式
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k  # 最大查询序列长度和最大键序列长度
            # 构建cu_seqlens，用于指示每个序列在合并后张量中的起始位置
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=kv.device)                    
            output = flash_attn_unpadded_kvpacked_func(  # 调用FlashAttention核心函数
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,  # 训练时使用dropout
                softmax_scale=self.softmax_scale, causal=causal  # 是否使用因果掩码 (例如decoder)
            )
            # 将输出张量恢复为 (批量大小, 序列长度, ...) 的形状
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:  # 如果提供了key_padding_mask，表示序列中可能存在padding
            nheads = kv.shape[-2]  # 获取头数
            q = rearrange(q, 'b s ... -> (b s) ...') # 合并q的批量和序列维度
            max_sq = seqlen_q # 最大查询序列长度
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device) # q的cu_seqlens

            # 处理kv的padding
            x = rearrange(kv, 'b s two h d -> b s (two h d)') # 预处理kv的形状以适应unpad_input
            # unpad_input会移除padding部分，并返回处理后的数据、原始索引、cu_seqlens和最大有效序列长度
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            # 将unpad后的数据恢复为 (有效总token数, 2, 头数, 头维度) 的形状
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)

            output_unpad = flash_attn_unpadded_kvpacked_func( # 对unpad后的数据调用FlashAttention
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            # 将输出张量恢复为 (批量大小, 序列长度, ...) 的形状
            # 注意：FlashAttention的输出是针对unpadded输入的，如果需要pad回原始形状，这里可能需要额外处理，
            # 但当前实现直接将 (b s) ... -> b s ...，意味着输出的序列长度是seqlen_q，padding部分可能需要被忽略或后续处理。
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None  # 返回注意力输出和None (通常多头注意力会返回注意力权重，但FlashAttention不直接返回)


class FlashMHA(nn.Module): # 定义Flash Multi-Head Attention模块

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first  # 强制要求batch_first为True，因为FlashAttention的输入格式通常是 (batch, seqlen, ...)
        factory_kwargs = {'device': device, 'dtype': dtype} # 构造参数，用于创建内部模块
        super().__init__()
        self.embed_dim = embed_dim  # 输入特征维度
        self.causal = causal  # 是否使用因果掩码
        self.bias = bias  # 线性投影时是否使用偏置

        self.num_heads = num_heads  # 注意力头的数量
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads" # 输入维度必须能被头数整除
        self.head_dim = self.embed_dim // num_heads  # 每个头的维度
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8" # FlashAttention对头维度的要求

        # 输入投影权重，用于将Q, K, V 投影到多头空间，(3 * embed_dim, embed_dim) 表示将三个投影矩阵合并存储
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias: # 如果使用偏置
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) # 输入投影偏置
        else:
            self.register_parameter('in_proj_bias', None) # 否则注册为None

        # 内部注意力模块，使用前面定义的FlashAttention
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters() # 初始化参数

    def _reset_parameters(self) -> None: # 参数初始化方法
        xavier_uniform_(self.in_proj_weight) # 使用xavier均匀分布初始化输入投影权重
        if self.in_proj_bias is not None: # 如果存在偏置
            constant_(self.in_proj_bias, 0.) # 将输入投影偏置初始化为0
            constant_(self.out_proj.bias, 0.) # 将输出投影偏置初始化为0
        
    def forward(self, q, k, v, key_padding_mask=None): # 前向传播函数
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) # 输入q,k,v的形状
        key_padding_mask: bool tensor of shape (batch, seqlen) # 键的填充掩码
        """
        # 对q, k, v进行打包投影
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        # 将投影后的q, k, v调整形状以匹配多头注意力的输入格式 (batch, seqlen, num_heads, head_dim)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        # 将k和v打包在一起，因为FlashAttention期望kv是打包的
        kv = torch.stack([k, v], dim=2) # (batch, seqlen, 2, num_heads, head_dim)
        
        # 调用内部的FlashAttention模块计算注意力上下文
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        # 将上下文张量调整回 (batch, seqlen, embed_dim) 并通过输出投影层
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights # 返回最终输出和注意力权重(此处为None)


@ATTENTION.register_module()  # 将MultiheadFlashAttention注册到mmcv的ATTENTION注册表中
class MultiheadFlashAttention(BaseModule):  # 定义一个基于FlashMHA的MultiheadAttention包装器
    """A wrapper for ``torch.nn.MultiheadAttention``. # torch.nn.MultiheadAttention 的包装器。
    This module implements MultiheadAttention with identity connection, # 此模块实现了带有恒等连接的多头注意力，
    and positional encoding  is also passed as input. # 并且位置编码也作为输入传递。
    Args: # 参数说明
        embed_dims (int): The embedding dimension. # 嵌入维度。
        num_heads (int): Parallel attention heads. # 并行注意力头的数量。
        attn_drop (float): A Dropout layer on attn_output_weights. # 注意力输出权重上的Dropout层。
            Default: 0.0. # 默认为0.0。
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`. # `nn.MultiheadAttention` 之后的Dropout层。
            Default: 0.0. # 默认为0.0。
        dropout_layer (agent:`ConfigDict`): The dropout_layer used # 添加快捷连接时使用的dropout_layer。
            when adding the shortcut.
        init_cfg (agent:`mmcv.ConfigDict`): The Config for initialization. # 初始化配置。
            Default: None. # 默认为None。
        batch_first (bool): When it is True,  Key, Query and Value are shape of # 如果为True，则Key, Query和Value的形状为
            (batch, n, embed_dim), otherwise (n, batch, embed_dim). # (batch, n, embed_dim)，否则为 (n, batch, embed_dim)。
             Default to False. # 默认为False。(在此实现中实际强制为True)
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.), # dropout层配置
                 init_cfg=None, # 初始化配置
                 batch_first=True, # 是否batch在第一维
                 **kwargs):
        super(MultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs: # 处理旧的dropout参数以保持兼容性
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True # FlashMHA强制batch_first为True
        self.attn = FlashMHA( # 实例化FlashMHA作为核心注意力模块
            embed_dim=embed_dims, 
            num_heads=num_heads, 
            attention_dropout=attn_drop, 
            dtype=torch.float16, # 通常FlashAttention使用float16或bfloat16以获得最佳性能
            device='cuda', # FlashAttention通常在CUDA上运行
            **kwargs
        )

        self.proj_drop = nn.Dropout(proj_drop) # 输出投影后的dropout
        self.dropout_layer = build_dropout( # 残差连接前的dropout层
            dropout_layer) if dropout_layer else nn.Identity() # 如果未配置则为恒等映射

    @deprecated_api_warning({'residual': 'identity'}, # 标记residual参数已被identity替代
                            cls_name='MultiheadAttention')
    def forward(self,
                query, # 查询张量
                key=None, # 键张量
                value=None, # 值张量
                identity=None, # 用于残差连接的恒等张量
                query_pos=None, # 查询的位置编码
                key_pos=None, # 键的位置编码
                attn_mask=None, # 注意力掩码 (当前实现中不支持)
                key_padding_mask=None, # 键的填充掩码
                **kwargs):
        """Forward function for `MultiheadAttention`. # `MultiheadAttention`的前向函数。
        **kwargs allow passing a more general data flow when combining # **kwargs允许在与`transformerlayer`中的其他操作结合时传递更通用的数据流。
        with other operations in `transformerlayer`.
        Args: # 参数说明
            query (Tensor): The input query with shape [num_queries, bs, # 输入查询张量，如果self.batch_first为False，形状为[num_queries, bs, embed_dims]，否则为[bs, num_queries, embed_dims]。
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs, # 键张量，如果self.batch_first为False，形状为[num_keys, bs, embed_dims]，否则为[bs, num_keys, embed_dims]。
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None. # 如果为None，将使用`query`。默认为None。
            value (Tensor): The value tensor with same shape as `key`. # 值张量，形状与`key`相同。
                Same in `nn.MultiheadAttention.forward`. Defaults to None. # 与`nn.MultiheadAttention.forward`中的相同。默认为None。
                If None, the `key` will be used. # 如果为None，将使用`key`。
            identity (Tensor): This tensor, with the same shape as x, # 此张量形状与x相同，将用于恒等连接。
                will be used for the identity link.
                If None, `x` will be used. Defaults to None. # 如果为None，将使用`x`。默认为None。
            query_pos (Tensor): The positional encoding for query, with # 查询的位置编码，形状与`x`相同。如果不为None，则在前向函数之前会加到`x`上。默认为None。
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the # `key`的位置编码，形状与`key`相同。默认为None。如果不为None，则在前向函数之前会加到`key`上。如果为None，且`query_pos`与`key`形状相同，则`query_pos`将用于`key_pos`。默认为None。
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries, # 形状为[num_queries, num_keys]的ByteTensor掩码。
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None. # 与`nn.MultiheadAttention.forward`中的相同。默认为None。
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys]. # 形状为[bs, num_keys]的ByteTensor。
                Defaults to None. # 默认为None。
        Returns: # 返回值
            Tensor: forwarded results with shape # 前向传播结果张量，形状为
            [num_queries, bs, embed_dims] # [num_queries, bs, embed_dims]（如果self.batch_first为False），
            if self.batch_first is False, else # 否则为
            [bs, num_queries embed_dims]. # [bs, num_queries, embed_dims]。
        """
        assert attn_mask is None, 'attn mask not supported now.' # 当前实现不支持attn_mask
        if key is None: # 如果key未提供，则key等于query (自注意力)
            key = query
        if value is None: # 如果value未提供，则value等于key
            value = key
        if identity is None: # 如果identity未提供，则identity等于query (用于残差连接)
            identity = query
        if key_pos is None: # 如果key的位置编码未提供
            if query_pos is not None: # 且query的位置编码已提供
                # use query_pos if key_pos is not available # 如果key_pos不可用，则使用query_pos
                if query_pos.shape == key.shape: # 如果query_pos和key的形状相同
                    key_pos = query_pos # 则将key_pos设为query_pos
                else: # 否则发出警告
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None: # 如果提供了query的位置编码，则加到query上
            query = query + query_pos
        if key_pos is not None: # 如果提供了key的位置编码，则加到key上
            key = key + key_pos

        # The dataflow('key', 'query', 'value') of ``FlashAttention`` is (batch, num_query, embed_dims).
        # FlashAttention期望的输入格式是 (batch, seqlen, embed_dims)
        if not self.batch_first: # 如果当前输入不是batch_first (即 (seqlen, batch, embed_dims))
            # 转置为 (batch, seqlen, embed_dims)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        out = self.attn( # 调用FlashMHA进行注意力计算
            q=query,
            k=key,
            v=value,
            key_padding_mask=key_padding_mask)[0] # FlashMHA返回(context, attn_weights)，这里取context

        if not self.batch_first: # 如果原始输入不是batch_first，则将输出转置回去
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out)) # 应用输出dropout和残差连接前的dropout，然后进行残差连接


def gen_sineembed_for_position(pos_tensor, hidden_dim=256): # 生成基于正弦函数的位置编码
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/ # 大部分代码复制粘贴自DAB-DETR的实现
    """
    half_hidden_dim = hidden_dim // 2 # 隐藏维度的一半
    scale = 2 * math.pi # 缩放因子
    # 生成一个维度序列，用于计算不同频率的正弦和余弦函数
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim) # 计算10000^(2i/d)项

    # 提取x和y坐标并进行缩放
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    # 计算x和y坐标在不同维度上的正弦/余弦值
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t

    # 将正弦和余弦值交错排列
    # pos_x[..., 0::2].sin() 取偶数索引计算sin，pos_x[..., 1::2].cos() 取奇数索引计算cos
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    # 将y和x的位置编码连接起来 (通常Transformer中x和y的位置编码是分开或交错的)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

