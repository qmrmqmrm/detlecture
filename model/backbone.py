from utils.util_class import MyExceptionToCatch
import model.model_util as mu


def backbone_factory(backbone, conv_kwargs):
    if backbone == "Darknet53":
        return Darknet53(conv_kwargs)
    elif backbone == "Resnet50":
        return ResNet50(conv_kwargs)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {backbone}")


class BackboneBase:
    COUNT = 1
    def __init__(self, conv_kwargs):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **conv_kwargs)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **conv_kwargs)
        self.conv2d_k1_s2 = mu.CustomConv2D(kernel_size=1, strides=2, **conv_kwargs)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **conv_kwargs)

    def residual(self, x, filters):
        short_cut = x
        conv = self.conv2d_k1(x, filters // 2)
        conv = self.conv2d(conv, filters)
        return short_cut + conv

    def residual_50(self, x, filters):
        short_cut = x
        if BackboneBase.COUNT == 1:
            short_cut = self.conv2d_k1_s2(short_cut, filters)
            conv = self.conv2d_k1_s2(x, filters // 4)
        else:
            conv = self.conv2d_k1(x, filters // 4)
        conv = self.conv2d(conv, filters // 4)
        conv = self.conv2d_k1(conv, filters)
        BackboneBase.COUNT += 1
        return short_cut + conv


class Darknet53(BackboneBase):
    def __init__(self, conv_kwargs):
        super().__init__(conv_kwargs)

    def __call__(self, input_tensor):
        """
        conv'n' represents a feature map of which resolution is (input resolution / 2^n)
        e.g. input_tensor.shape[:2] == conv0.shape[:2], conv0.shape[:2]/8 == conv3.shape[:2]
        """
        features = dict()
        print(f"input {input_tensor.shape}")
        conv0 = self.conv2d(input_tensor, 32)
        print(f"input {conv0.shape}")
        conv1 = self.conv2d_s2(conv0, 64)
        print(f"input {conv1.shape}")
        conv1 = self.residual(conv1, 64)
        print(f"input {conv1.shape}")
        conv2 = self.conv2d_s2(conv1, 128)
        print(f"input {conv2.shape}")
        for i in range(2):
            conv2 = self.residual(conv2, 128)
        print(f"input {conv2.shape}")
        conv3 = self.conv2d_s2(conv2, 256)
        print(f"input {conv3.shape}")
        for i in range(8):
            conv3 = self.residual(conv3, 256)
        features["backbone_s"] = conv3
        print(f"input {conv3.shape}")

        conv4 = self.conv2d_s2(conv3, 512)
        print(f"input {conv4.shape}")
        for i in range(8):
            conv4 = self.residual(conv4, 512)
        features["backbone_m"] = conv4
        print(f"input {conv4.shape}")

        conv5 = self.conv2d_s2(conv4, 1024)
        print(f"input {conv5.shape}")
        for i in range(4):
            conv5 = self.residual(conv5, 1024)
        features["backbone_l"] = conv5
        print(f"input {conv5.shape}")

        return features


class ResNet50(BackboneBase):
    def __init__(self, conv_kwargs):
        super().__init__(conv_kwargs)

    def __call__(self, input_tensor):
        """
        conv'n' represents a feature map of which resolution is (input resolution / 2^n)
        e.g. input_tensor.shape[:2] == conv0.shape[:2], conv0.shape[:2]/8 == conv3.shape[:2]
        """
        features = dict()
        conv0 = self.conv2d_s2(input_tensor,32)
        conv1 = conv0
        for i in range(3):
            conv1 = self.residual_50(conv1, 256)
        BackboneBase.COUNT = 1
        conv2 = conv1
        for i in range(4):
            conv2 = self.residual_50(conv2, 512)
        BackboneBase.COUNT = 1
        features["backbone_s"] = conv3 = conv2
        for i in range(6):
            conv3 = self.residual_50(conv3, 1024)
        BackboneBase.COUNT = 1
        features["backbone_m"] = conv4 = conv3
        for i in range(3):
            conv4 = self.residual_50(conv4, 2048)
        features["backbone_l"] =conv4
        BackboneBase.COUNT = 1
        return features
