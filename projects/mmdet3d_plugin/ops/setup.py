import os # 导入os模块，用于与操作系统交互，例如获取环境变量

import torch # 导入PyTorch库
from setuptools import setup # 从setuptools导入setup函数，用于打包和分发Python模块
from torch.utils.cpp_extension import ( # 从PyTorch的cpp_extension模块导入相关类
    BuildExtension,  # 构建扩展的基类
    CppExtension,  # C++扩展类
    CUDAExtension,  # CUDA扩展类
)


def make_cuda_ext( # 定义一个辅助函数，用于创建CUDA或C++扩展模块
    name, # 扩展模块的名称 (例如 "deformable_aggregation_ext")
    module, # 包含源文件的模块路径 (例如 ".")
    sources, # C++源文件列表
    sources_cuda=[], # CUDA源文件列表 (可选)
    extra_args=[], # 额外的编译参数 (可选)
    extra_include_path=[], # 额外的包含路径 (可选)
):

    define_macros = [] # 初始化宏定义列表
    extra_compile_args = {"cxx": [] + extra_args} # 初始化额外的编译参数字典，cxx部分包含通用额外参数

    # 检查CUDA是否可用，或者是否通过环境变量强制使用CUDA
    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)] # 如果使用CUDA，添加WITH_CUDA宏定义
        extension = CUDAExtension # 使用CUDAExtension类来构建
        # 为nvcc编译器添加额外的编译参数
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",      # 禁用CUDA的半精度操作符
            "-D__CUDA_NO_HALF_CONVERSIONS__",    # 禁用CUDA的半精度转换
            "-D__CUDA_NO_HALF2_OPERATORS__",     # 禁用CUDA的half2操作符
        ]
        sources += sources_cuda # 将CUDA源文件添加到总的源文件列表中
    else: # 如果不使用CUDA
        print("Compiling {} without CUDA".format(name)) # 打印提示信息
        extension = CppExtension # 使用CppExtension类来构建 (仅编译C++部分)
        # sources列表保持不变，不包含CUDA源文件

    # 返回构建好的扩展对象
    return extension(
        name="{}.{}".format(module, name), # 扩展的完整名称，格式为 "module.name"
        sources=[os.path.join(*module.split("."), p) for p in sources], # 将源文件路径转换为相对于模块的完整路径
        include_dirs=extra_include_path, # 指定额外的包含目录
        define_macros=define_macros, # 传递宏定义
        extra_compile_args=extra_compile_args, # 传递额外的编译参数
    )


if __name__ == "__main__": # 如果该脚本作为主程序运行
    setup( # 调用setuptools的setup函数来配置和构建扩展
        name="deformable_aggregation_ext", # 扩展包的名称
        ext_modules=[ # 扩展模块列表
            make_cuda_ext( # 调用辅助函数创建具体的扩展模块
                name="deformable_aggregation_ext", # 扩展模块的名称
                module=".", # 模块路径 (当前目录)
                sources=[ # 源文件列表
                    f"src/deformable_aggregation.cpp", # C++源文件
                    f"src/deformable_aggregation_cuda.cu", # CUDA源文件 (如果CUDA可用)
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension}, # 指定构建扩展时使用的命令类
    )
