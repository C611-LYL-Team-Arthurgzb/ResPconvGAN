import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.modules.partialconv2d import PartialConv2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = PartialConv2d(inplanes, planes, kernel_size=1, bias=False, multi_channel=True, return_mask=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PartialConv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, multi_channel=True, return_mask=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PartialConv2d(planes, planes * self.expansion, kernel_size=1, bias=False, multi_channel=True,
                                   return_mask=True)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn4 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.downsample_nn = nn.BatchNorm2d(planes * 4)
        self.stride = stride

    def forward(self, x, mask):
        residual = x
        residual_mask = mask
        # print("Bottleneck初始", x.shape, mask.shape,self.inplanes,self.planes)
        out, out_mask = self.conv1(x, mask)
        #out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        #out = self.bn2(out)
        out = self.relu(out)

        out, out_mask = self.conv3(out, out_mask)
        #out = self.bn3(out)
        # print("对齐数据：",self.downsample)
        if self.downsample is not None:
            # residual_mask= self.downsample(x)
            residual, residual_mask = self.downsample(x, residual_mask)
            # residual = self.downsample_nn

        # print("Bottleneck叠加", out.shape, residual.shape)
        out = out + residual
        out_mask = out_mask + residual_mask
        out = self.bn4(out)
        out = self.relu(out)
        # print("Bottleneck输出", out.shape, out_mask.shape)
        return out, out_mask


class Block_make_layer2(nn.Module):
    def __init__(self, inplanes, block, planes, blocks, stride=1):
        super(Block_make_layer2, self).__init__()

        self.planes = planes
        self.block = block
        self.blocks = blocks
        self.stride = stride
        self.inplanes = inplanes

        self.inplanes = planes * 4
        self.layers_2 = self.block(self.inplanes, self.planes)

    def forward(self, x, mask):
        for i in range(1, self.blocks):
            # print("layers_2", x.shape, mask.shape, self.inplanes)
            # print("KKKKKKKKKKKKKKKKKKK",self.blocks)
            x, mask = self.layers_2(x, mask)
        return x, mask


class Block_make_layer(nn.Module):
    def __init__(self, inplanes, block, planes, blocks, stride=1):
        super(Block_make_layer, self).__init__()
        self.downsample = None
        self.planes = planes
        self.block = block
        self.blocks = blocks
        self.stride = stride
        self.inplanes = inplanes

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     print("rrrrrrrrrrrrrrrrrrr")
        self.downsample = PartialConv2d(self.inplanes, self.planes * 4, kernel_size=1, stride=stride, bias=False,
                                        multi_channel=True, return_mask=True)

        self.layers_1 = self.block(self.inplanes, self.planes, self.stride, self.downsample)
        self.layers_11 = self.block(self.inplanes, self.planes, self.stride)

        # self.inplanes = planes * 4
        # self.layers_2 = self.block(self.inplanes, self.planes)
        self.layers_2 = Block_make_layer2(self.inplanes, block, self.planes, self.blocks, self.stride)

    def forward(self, x, mask):
        if self.stride != 1 or self.inplanes != self.planes * 4:
            # print("layers_1")
            x, mask = self.layers_1(x, mask)
        else:
            # print("layers_11")
            x, mask = self.layers_11(x, mask)

        # self.inplanes = self.planes * 4
        # for i in range(1, self.blocks):
        #     print("layers_2",x.shape, mask.shape,self.inplanes)
        #     x, mask = self.layers_2(x, mask)
        x, mask = self.layers_2(x, mask)
        return x, mask


class PDResNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000):
        self.inplanes = 64
        super(PDResNet, self).__init__()
        self._make_layer = Block_make_layer
        self.conv1 = PartialConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, multi_channel=True, return_mask=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.inplanes, block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(self.inplanes * 4, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.inplanes * 4 * 2, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.inplanes * 4 * 4, block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     self.downsample = None
    #
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         print("rrrrrrrrrrrrrrrrrrr")
    #         self.downsample = nn.Sequential(
    #             PartialConv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False, multi_channel=True, return_mask=True),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, self.downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))
    #
    #     return nn.Sequential(*layers)

    def forward(self, x, mask):

        # print("进入PDResNet", x.shape, mask.shape)
        x0, mask0 = self.conv1(x, mask)

        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        # x = self.maxpool(x)

        # print("\nkkkkkkkk")
        # print(self.layer2)
        # print("一次", x0.shape, mask0.shape)
        x1, mask1 = self.layer1(x0, mask0)
        # print("二次",x1.shape, mask1.shape)
        x2, mask2 = self.layer2(x1, mask1)
        # print("三次", x1.shape, mask1.shape)
        x3, mask3 = self.layer3(x2, mask2)
        # print("四次", x1.shape, mask1.shape)
        x4, mask4 = self.layer4(x3, mask3)
        # print("结束")

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x0, mask0, x1, mask1, x2, mask2, x3, mask3, x4, mask4


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) �C C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        self.inc = in_ch
        self.ouc = out_ch
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activ == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, input, input_mask):
        # print("tongdao:",self.inc,self.ouc)
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PConvUNet(nn.Module):
    def __init__(self, layer_size=5, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        # self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        # self.enc_2 = PCBActiv(64, 128, sample='down-5')
        # self.enc_3 = PCBActiv(128, 256, sample='down-5')
        # self.enc_4 = PCBActiv(256, 512, sample='down-3')
        # self.enc_5 = PCBActiv(512, 512, sample='down-3')
        # PDResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        # self.enc_1 = PDResNet.conv1
        # self.enc_1_BN = PDResNet.bn1
        # self.enc_1_RELU = PDResNet.relu
        # self.enc_2 = PDResNet.layer1
        # self.enc_3 = PDResNet.layer2
        # self.enc_4 = PDResNet.layer3
        # self.enc_5 = PDResNet.layer4
        self.enc_0 = PDResNet(Bottleneck, [3, 4, 6, 3])

        self.dec_5 = PCBActiv(2048 + 1024, 1024, activ='leaky')
        self.dec_4 = PCBActiv(1024 + 512, 512, activ='leaky')
        self.dec_3 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_2 = PCBActiv(256 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ='tanh', conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        h_dict['h_1'], h_mask_dict['h_1'], h_dict['h_2'], h_mask_dict['h_2'], h_dict['h_3'], h_mask_dict['h_3'], h_dict[
            'h_4'], h_mask_dict['h_4'], h_dict['h_5'], h_mask_dict['h_5'] = self.enc_0(h_dict['h_0'],
                                                                                       h_mask_dict['h_0'])

        # h_key_prev = 'h_0'
        # for i in range(1, self.layer_size + 1):
        #     l_key = 'enc_{:d}'.format(i)
        #     h_key = 'h_{:d}'.format(i)
        #     h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
        #         h_dict[h_key_prev], h_mask_dict[h_key_prev])
        #     h_key_prev = h_key
        #
        # h_key = 'h_{:d}'.format(self.layer_size)
        # h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8
        h, h_mask = h_dict['h_5'], h_mask_dict['h_5']
        # print("shape:",h_dict['h_0'].shape)
        # print("shape:", h_dict['h_1'].shape)
        # print("shape:", h_dict['h_2'].shape)
        # print("shape:", h_dict['h_3'].shape)
        # print("shape:", h_dict['h_4'].shape)
        # print("shape:", h_dict['h_5'].shape)
        
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            # print("shape:",h.shape,h_dict[enc_h_key].shape,enc_h_key)
            if (i != 2):
                h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
                h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')
            # print("shape2:", h.shape, h_dict[enc_h_key].shape, enc_h_key)

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            # print("shape2:", h.shape)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


class PConvUNet1(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


if __name__ == '__main__':
    # size = (1, 3, 5, 5)
    # input = torch.ones(size)
    # input_mask = torch.ones(size)
    # input_mask[:, :, 2:, :][:, :, :, 2:] = 0
    #
    # conv = PartialConv(3, 3, 3, 1, 1)
    # l1 = nn.L1Loss()
    # input.requires_grad = True
    #
    # output, output_mask = conv(input, input_mask)
    # loss = l1(output, torch.randn(1, 3, 5, 5))
    # loss.backward()

    # size = (1, 3, 5, 5)
    # input = torch.ones(size)
    # input_mask = torch.ones(size)
    # input_mask[:, :, 2:, :][:, :, :, 2:] = 0
    # Pconv = PDResNet(Bottleneck, [3, 4, 6, 3])
    # output, output_mask= Pconv._make_layer(Bottleneck,64,1)(input,input_mask)
    # print(output, output_mask)

    assert (torch.sum(input.grad != input.grad).item() == 0)
    # assert (torch.sum(torch.isnan(conv.input_conv.weight.grad)).item() == 0)
    # assert (torch.sum(torch.isnan(conv.input_conv.bias.grad)).item() == 0)

    # model = PConvUNet()
    # output, output_mask = model(input, input_mask)
    PConvUNet().module
