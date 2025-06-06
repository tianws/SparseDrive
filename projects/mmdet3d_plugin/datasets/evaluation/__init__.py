# 该 __init__.py 文件用于将 'evaluation' 目录标记为一个Python子包。
# 与评估指标、评估钩子或评估流程相关的自定义模块、类或函数可以在此包中定义，
# 并可通过此文件导出（如果需要），以便在项目其他地方方便地导入。
#
# 例如，如果有一个自定义的评估指标实现：
# from .custom_metrics import CustomAveragePrecision
#
# __all__ = ['CustomAveragePrecision']
#
# 这样，其他代码就可以通过 from projects.mmdet3d_plugin.datasets.evaluation import CustomAveragePrecision 来使用它。
# 当前此文件为空，表示该包可能通过其他方式导入其模块，或者目前没有需要导出的公共接口。
