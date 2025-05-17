import torch
from torchsummary import summary
import time
import torch.nn.functional as F
from torch import nn as nn

class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels, middle_channels,out_channels):
        super().__init__()
        
        self.step = torch.nn.Sequential(
            # Conv
            torch.nn.Conv3d(in_channels=in_channels,out_channels=middle_channels,kernel_size=3,padding=1,stride=1),
            # Batch Norm
            torch.nn.BatchNorm3d(middle_channels),
            # ReLU
            torch.nn.ReLU(),
            # Conv
            torch.nn.Conv3d(in_channels=middle_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1),
            # Batch Norm
            torch.nn.BatchNorm3d(out_channels),
            # ReLU
            torch.nn.ReLU()
        )
    
    def forward(self,x):
        
        return self.step(x)

class UNet3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Analysis path
        self.layer1 = ConvBlock(1,32,64)
        self.layer2 = ConvBlock(64,64,128)
        self.layer3 = ConvBlock(128,128,256)
        self.layer4 = ConvBlock(256,256,512)
        
        # Synthesis path
        self.layer5 = ConvBlock(256+512,256,256)
        self.layer6 = ConvBlock(128+256,128,128)
        self.layer7 = ConvBlock(64+128,64,64)
        
        # Last conv
        self.layer8  = torch.nn.Conv3d(in_channels=64,out_channels=2,kernel_size=1,padding=0,stride=1)
        
        # Maxpooling
        self.maxpool = torch.nn.MaxPool3d(kernel_size=2, stride = 2)

        # Upconv
        self.upsample1 = torch.nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.upsample2 = torch.nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.upsample3 = torch.nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        # Sigmoid
        self.sigmoid = torch.nn.Sigmoid()
        # Softmax
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self,x):
        # Define the foward pass

        # input: 32 x H x W X D, output: 64 x H x W X D
        x1 = self.layer1(x)

        # input:64 x H x W x D, output: 64 x H/2 x W/2 x D/2
        x1_p = self.maxpool(x1)
        
        # input:  64 x H/2 x W/2 x D/2 , output: 128 x H/2 x W/2 x D/2
        x2 = self.layer2(x1_p)

        # input:128 x H/2 x W/2 x D/2 , output: 128 x H/4 x W/4 x D/4
        x2_p = self.maxpool(x2)
        
        # input: 128 x H/4 x W/4 x D/4, output: 256 x H/4 x W/4 x D/4
        x3 = self.layer3(x2_p)

        #input:256 x H/4 x W/4 x D/4, output: 256 x H/8 x W/8 x D/8
        x3_p = self.maxpool(x3)
        
        #input: 256 x H/8 x W/8 x D/8, output: 512 x H/8 x W/8 x D/8
        x4 = self.layer4(x3_p)
        
        # input: 512 x H/8 x W/8 x D/8，output: 512 x H/4 x W/4 x D/4
        x5 = self.upsample1(x4)

        # cat,output: (512+256) x H/4 x W/4 x D/4
        x5 = torch.cat([x5,x3],dim=1)

        # input: (512+256) x H/4 x W/4 x D/4, output: 256 x H/4 x W/4 x D/4
        x5 = self.layer5(x5)
        
        # input: 256 x H/4 x W/4 x D/4,output: 256 x H/2 x W/2 x D/2
        x6  = self.upsample2(x5)
        
        # cat,output: (256+128) x H/2 x W/2 x D/2
        x6 = torch.cat([x6,x2],dim=1)

        # input: (256+128) x H/2 x W/2 x D/2, output: 128 x H/2 x W/2 x D/2
        x6 = self.layer6(x6)
        
        # input:128 x H/2 x W/2 x D/2, output: 128 x H x W x D
        x7 = self.upsample3(x6)

        # cat, output: (128+64) x H x W x D
        x7 = torch.cat([x7,x1],dim=1)

        # input: (128+64) x H x W x D, output: 64 x H x W x D
        x7 = self.layer7(x7)
        
        # last conv,input: 64 x H x W x D, output: classes x H x W x D
        x8 = self.layer8(x7)
        
        # Softmax
        # x9= self.softmax(x8)
        # Sigmoid
        # x9= self.sigmoid(x8)
        
        return x8
        # return x9
    
## Generalized Dice Loss & dice loss
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='softmax'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)
    
class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='softmax', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())
    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    # Apply softmax to the input
    input = F.softmax(input, dim=1)

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input + target).sum(-1)
    # denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    Input is expected to be probabilities instead of logits.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class dice_loss_logit(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
        dice_per_class = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice_per_class[1:].mean()

class dice_coefficient_logit(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
        dice_per_class = (2 * intersection + 1e-6) / (union + 1e-6)
        return dice_per_class[1:].mean()





if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = UNet3D()
    start_time = time.time()
    summary(model=model, input_size=(1, 384, 384, 64), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))

