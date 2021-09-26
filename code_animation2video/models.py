from torch import nn
from torch import nn
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

class ResBlock2d(nn.Module):
    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += x
        return out

class UpBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class HourglassEncoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(HourglassEncoder, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

class HourglassDecoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(HourglassDecoder, self).__init__()
        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features
    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out

class Hourglass(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = HourglassEncoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = HourglassDecoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters
    def forward(self, x):
        return self.decoder(self.encoder(x))

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out

class Encoder(nn.Module):
    def __init__(self, num_channels, num_down_blocks=3, block_expansion=64, max_features=512,
                 ):
        super(Encoder, self).__init__()
        self.in_conv = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.Sequential(*down_blocks)
    def forward(self, image):
        out = self.in_conv(image)
        out = self.down_blocks(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, num_bottleneck_blocks,num_down_blocks=3, block_expansion=64, max_features=512):
        super(Bottleneck, self).__init__()
        bottleneck = []
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            bottleneck.append(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        self.bottleneck = nn.Sequential(*bottleneck)
    def forward(self, feature_map):
        out = self.bottleneck(feature_map)
        return out

class Decoder(nn.Module):
    def __init__(self,num_channels, num_down_blocks=3, block_expansion=64, max_features=512):
        super(Decoder, self).__init__()
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_conv = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.sigmoid = nn.Sigmoid()
    def forward(self, feature_map):
        out = self.up_blocks(feature_map)
        out = self.out_conv(out)
        out = self.sigmoid(out)
        return out

def warp_image(image, motion_flow):
    _, h_old, w_old, _ = motion_flow.shape
    _, _, h, w = image.shape
    if h_old != h or w_old != w:
        motion_flow = motion_flow.permute(0, 3, 1, 2)
        motion_flow = F.interpolate(motion_flow, size=(h, w), mode='bilinear')
        motion_flow = motion_flow.permute(0, 2, 3, 1)
    return F.grid_sample(image, motion_flow)

def make_coordinate_grid(spatial_size, type):
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

class ForegroundMatting(nn.Module):
    def __init__(self, num_channels,scale_factor,matting_channel,num_blocks,block_expansion, max_features):
        super(ForegroundMatting, self).__init__()
        self.down_sample_image = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.down_sample_flow = AntiAliasInterpolation2d(2, scale_factor)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features= num_channels * 2 + 2,
                                   max_features=max_features, num_blocks=num_blocks)
        self.foreground_mask = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        self.matting_mask = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        self.matting = nn.Conv2d(self.hourglass.out_filters, matting_channel, kernel_size=(7, 7), padding=(3, 3))
        self.scale_factor = scale_factor
        self.sigmoid = nn.Sigmoid()
    def forward(self, reference_image, dense_flow):
        '''
        source_image : b x c x h x w
        dense_tensor: b x h x w x 2
        '''
        res_out = {}
        if self.scale_factor != 1: #down sample the image
            reference_image = self.down_sample_image(reference_image)
            dense_flow = self.down_sample_flow(dense_flow.permute(0,3,1,2)).permute(0,2,3,1)
        batch, _, h, w = reference_image.shape
        warped_image = warp_image(reference_image, dense_flow)#warp the image with dense flow
        res_out['warped_image'] = warped_image
        hourglass_input = torch.cat([reference_image,dense_flow.permute(0,3,1,2),warped_image], dim=1)
        hourglass_out = self.hourglass(hourglass_input)
        foreground_mask = self.foreground_mask(hourglass_out) # compute foreground mask
        foreground_mask = self.sigmoid(foreground_mask).permute(0,2,3,1)
        res_out['foreground_mask'] = foreground_mask
        grid_flow = make_coordinate_grid((h, w), dense_flow.type())
        dense_flow_foreground = dense_flow * foreground_mask + (1-foreground_mask) * grid_flow.unsqueeze(0) ## revise the dense flow
        res_out['dense_flow_foreground'] = dense_flow_foreground
        res_out['dense_flow_foreground_vis'] = dense_flow * foreground_mask
        matting_mask = self.matting_mask(hourglass_out) # compute matting mask
        matting_mask = self.sigmoid(matting_mask)
        res_out['matting_mask'] = matting_mask
        matting_image = self.matting(hourglass_out) # computing matting image
        res_out['matting_image'] = matting_image
        return res_out



class VideoGenerator(nn.Module):
    def __init__(self, num_channels, encoder_num_down_blocks=3,encoder_block_expansion=64,
                 encoder_max_features=512, houglass_num_blocks=5,
                 houglass_block_expansion = 64,houglass_max_features = 1024, num_bottleneck_blocks=6):
        super(VideoGenerator, self).__init__()
        self.encoder = Encoder(num_channels,encoder_num_down_blocks,
                                        encoder_block_expansion,encoder_max_features)
        matting_channel = int(min(encoder_max_features, encoder_block_expansion * (2 ** encoder_num_down_blocks)))
        self.foreground_matting = ForegroundMatting(num_channels,scale_factor=1/(2**encoder_num_down_blocks),matting_channel = matting_channel,
                                               num_blocks = houglass_num_blocks,block_expansion =houglass_block_expansion,
                                               max_features = houglass_max_features)
        self.bottleneck = Bottleneck(num_bottleneck_blocks,encoder_num_down_blocks,
                                              encoder_block_expansion,encoder_max_features)
        self.decoder = Decoder(num_channels,encoder_num_down_blocks, encoder_block_expansion,
                                        encoder_max_features)
    def forward(self, reference_image,dense_flow):
        '''
        source_image: b x c x h x w
        dense_flow: b x h x w x 2
        '''
        feature_map = self.encoder(reference_image) ## compute feature map
        res_out = self.foreground_matting(reference_image, dense_flow) ## compute matting & revise dense flow
        assert feature_map.shape[2] == res_out['matting_mask'].shape[2] and feature_map.shape[3] == res_out['matting_mask'].shape[3]
        warped_feature_map = warp_image(feature_map, res_out['dense_flow_foreground']) * res_out['matting_mask']  + (1-res_out['matting_mask']) * res_out['matting_image']
        warped_feature_map = self.bottleneck(warped_feature_map) # decode feature map
        synthetic_image = self.decoder(warped_feature_map) # decode feature map
        res_out['synthetic_image'] = synthetic_image
        return res_out