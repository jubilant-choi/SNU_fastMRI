import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        alone = None
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.alone = alone

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        
    def norm(self, x):
        b, h, w = x.shape
        try:
            x = x.view(b, h * w)
            mean = x.mean(dim=1).view(b, 1, 1)
            std = x.std(dim=1).view(b, 1, 1)
            x = x.view(b, h, w)
        except:
            x = x.reshape(b, h * w)
            mean = x.mean(dim=1).reshape(b, 1, 1)
            std = x.std(dim=1).reshape(b, 1, 1)
            x = x.reshape(b, h, w)
        return (x - mean) / std, mean, std
    
    def unnorm(self, x, mean, std):
        return x * std + mean
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if self.alone:
            image, mean, std = self.norm(image)
            image = image.unsqueeze(1)
        
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        
        if self.alone:
            output = output.squeeze(1)
            output = self.unnorm(output, mean, std)
        
        return output

    
class UnetPP(Unet):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 5,
        drop_prob: float = 0.0,
        deep_supervision: bool = True,
        alone = True
    ):
        super().__init__(in_chans,out_chans)

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.deep_supervision = deep_supervision
        self.alone = alone
        
        depth_chans = [self.chans * (2**i) for i in range(self.num_pool_layers)]
        
        self.deep_supervision = deep_supervision

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = ConvBlock(self.in_chans, depth_chans[0])
        self.conv1_0 = ConvBlock(depth_chans[0], depth_chans[1])
        self.conv2_0 = ConvBlock(depth_chans[1], depth_chans[2])
        self.conv3_0 = ConvBlock(depth_chans[2], depth_chans[3])
        self.conv4_0 = ConvBlock(depth_chans[3], depth_chans[4])        
        
        self.up0_1 = TransposeConvBlock(depth_chans[1], depth_chans[0])
        self.up1_1 = TransposeConvBlock(depth_chans[2], depth_chans[1])        
        self.up2_1 = TransposeConvBlock(depth_chans[3], depth_chans[2])        
        self.up3_1 = TransposeConvBlock(depth_chans[4], depth_chans[3])
        
        self.conv0_1 = ConvBlock(depth_chans[0]*2, depth_chans[0])
        self.conv1_1 = ConvBlock(depth_chans[1]*2, depth_chans[1])
        self.conv2_1 = ConvBlock(depth_chans[2]*2, depth_chans[2])
        self.conv3_1 = ConvBlock(depth_chans[3]*2, depth_chans[3]) 

        self.up0_2 = TransposeConvBlock(depth_chans[1], depth_chans[0])
        self.up1_2 = TransposeConvBlock(depth_chans[2], depth_chans[1])
        self.up2_2 = TransposeConvBlock(depth_chans[3], depth_chans[2])        
        
        self.conv0_2 = ConvBlock(depth_chans[0]*3, depth_chans[0])
        self.conv1_2 = ConvBlock(depth_chans[1]*3, depth_chans[1])
        self.conv2_2 = ConvBlock(depth_chans[2]*3, depth_chans[2])   

        self.up0_3 = TransposeConvBlock(depth_chans[1], depth_chans[0])
        self.up1_3 = TransposeConvBlock(depth_chans[2], depth_chans[1])        
        
        self.conv0_3 = ConvBlock(depth_chans[0]*4, depth_chans[0])
        self.conv1_3 = ConvBlock(depth_chans[1]*4, depth_chans[1])

        self.up0_4 = TransposeConvBlock(depth_chans[1], depth_chans[0])        
        
        self.conv0_4 = ConvBlock(depth_chans[0]*5, depth_chans[0])        

        if self.deep_supervision:
            self.final1 = nn.Conv2d(depth_chans[0], self.out_chans, kernel_size=1, stride=1)
            self.final2 = nn.Conv2d(depth_chans[0], self.out_chans, kernel_size=1, stride=1)
            self.final3 = nn.Conv2d(depth_chans[0], self.out_chans, kernel_size=1, stride=1)
            self.final4 = nn.Conv2d(depth_chans[0], self.out_chans, kernel_size=1, stride=1)
        else:
            self.final = nn.Conv2d(depth_chans[0], self.out_chans, kernel_size=1, stride=1)
    
    def pool(self, output):
        return F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if self.alone:
            image, mean, std = self.norm(image)
            image = image.unsqueeze(1)
        
        output = image

        x0_0 = self.conv0_0(output)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up0_1(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up0_2(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1_2(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up0_3(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1_3(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up0_4(x1_3)], 1))        

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output = output1+output2+output3+output4

        else:
            output = self.final(x0_4)
        
        if self.alone:
            output = output.squeeze(1)
            output = self.unnorm(output, mean, std)
        
        return output
    

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float=0):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)