"""
模型工厂 - 统一管理所有分割模型
"""
import torch
import torch.nn as nn
from typing import Dict, Any
import importlib


class ModelFactory:
    """模型工厂类"""

    # 支持的模型列表
    MODEL_REGISTRY = {
        # U-Net系列
        'unet': 'models.unet.UNet',
        'attention_unet': 'models.unet.AttentionUNet',
        'nested_unet': 'models.unet.NestedUNet',

        # 其他经典模型
        'deeplabv3plus': 'models.deeplabv3plus.DeepLabV3Plus',
        'pspnet': 'models.pspnet.PSPNet',
        'segnet': 'models.segnet.SegNet',

        # 高分辨率网络
        'hrnet': 'models.hrnet.HRNet',

        # Transformer模型
        'segformer': 'models.segformer.SegFormer',
        'swin_unet': 'models.swin_unet.SwinUperNet',

        # 注意力网络
        'danet': 'models.danet.DANet',
        'ccnet': 'models.ccnet.CCNet',
    }

    @classmethod
    def list_models(cls):
        """列出所有支持的模型"""
        print("支持的模型列表:")
        for name, path in cls.MODEL_REGISTRY.items():
            print(f"  - {name}: {path}")
        return list(cls.MODEL_REGISTRY.keys())

    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> nn.Module:
        """
        创建模型实例

        Args:
            model_name: 模型名称
            **kwargs: 模型参数
                - in_channels: 输入通道数
                - num_classes: 类别数
                - 其他模型特定参数

        Returns:
            模型实例
        """
        model_name = model_name.lower()

        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(f"不支持的模型: {model_name}. 可用模型: {list(cls.MODEL_REGISTRY.keys())}")

        # 动态导入模型类
        module_path, class_name = cls.MODEL_REGISTRY[model_name].rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # 创建模型实例
        model = model_class(**kwargs)

        return model

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        model_name = model_name.lower()

        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(f"不支持的模型: {model_name}")

        # 模型描述
        descriptions = {
            'unet': 'U-Net: 经典的编码器-解码器结构，适合医学图像分割',
            'attention_unet': 'Attention U-Net: 在跳跃连接中加入注意力门控',
            'nested_unet': 'U-Net++: 嵌套U-Net，使用密集跳跃连接',
            'deeplabv3plus': 'DeepLabV3+: 带ASPP的编码器-解码器结构',
            'pspnet': 'PSPNet: 金字塔场景解析网络，使用多尺度池化',
            'segnet': 'SegNet: 使用VGG编码器-解码器结构',
            'hrnet': 'HRNet: 高分辨率网络，全程保持高分辨率表示',
            'segformer': 'SegFormer: 基于Transformer的高效分割网络',
            'swin_unet': 'Swin UperNet: 使用Swin Transformer的层次化网络',
            'danet': 'DANet: 双重注意力网络，结合位置和通道注意力',
            'ccnet': 'CCNet: Criss-Cross注意力网络，捕获全局上下文',
        }

        # 默认参数
        default_params = {
            'unet': {'base_channels': 64, 'bilinear': True},
            'attention_unet': {'base_channels': 64, 'bilinear': True},
            'nested_unet': {'base_channels': 32, 'deep_supervision': False},
            'deeplabv3plus': {'backbone': 'resnet50', 'output_stride': 16},
            'pspnet': {'backbone': 'resnet50'},
            'segnet': {},
            'hrnet': {'base_channels': 32},
            'segformer': {'img_size': 512},
            'swin_unet': {'img_size': 512, 'embed_dim': 96},
            'danet': {'backbone': 'resnet50'},
            'ccnet': {'backbone': 'resnet50', 'recurrence': 2},
        }

        return {
            'name': model_name,
            'class': cls.MODEL_REGISTRY[model_name],
            'description': descriptions.get(model_name, '暂无描述'),
            'default_params': default_params.get(model_name, {})
        }

    @classmethod
    def count_parameters(cls, model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters())

    @classmethod
    def count_trainable_parameters(cls, model: nn.Module) -> int:
        """计算可训练参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置构建模型

    Args:
        config: 配置字典，包含:
            - model_name: 模型名称
            - in_channels: 输入通道数
            - num_classes: 类别数
            - model_params: 其他模型参数

    Returns:
        模型实例
    """
    model_name = config['model_name']
    in_channels = config.get('in_channels', 3)
    num_classes = config.get('num_classes', 2)

    # 获取模型默认参数
    model_info = ModelFactory.get_model_info(model_name)
    model_params = model_info['default_params'].copy()

    # 覆盖默认参数
    if 'model_params' in config:
        model_params.update(config['model_params'])

    # 创建模型
    model = ModelFactory.create_model(
        model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        **model_params
    )

    return model


if __name__ == "__main__":
    # 测试模型工厂
    ModelFactory.list_models()

    print("\n测试创建模型:")
    for model_name in ['unet', 'deeplabv3plus', 'segformer']:
        try:
            model = ModelFactory.create_model(model_name, in_channels=3, num_classes=2)
            params = ModelFactory.count_parameters(model)
            print(f"{model_name}: {params / 1e6:.2f}M 参数")
        except Exception as e:
            print(f"{model_name}: 创建失败 - {e}")

    # 测试build_model
    config = {
        'model_name': 'unet',
        'in_channels': 3,
        'num_classes': 2,
        'model_params': {'base_channels': 32}
    }
    model = build_model(config)
    print(f"\n从配置构建模型: {model.__class__.__name__}")
