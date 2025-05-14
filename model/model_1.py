from torch import nn
from torchsummary import summary
import torch
import time
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        



class UNet3D_test(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D_test, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    
    def forward(self, input):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        #Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out



if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = UNet3D_test(in_channels=1, num_classes=2)
    start_time = time.time()
    summary(model=model, input_size=(1, 64, 48, 48), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))


# class dice_loss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, pred, target):
#         pred = pred[:, 1]
#         target = target[:,1]
#         intersection = torch.sum(pred * target, dim=(1, 2, 3))
#         union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
#         dice = (2 * intersection ) / (union + 1e-6)
#         dice_loss = 1 - dice.mean()
#         return dice_loss

# class dice_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, logits, target, smooth=1e-6):
#         probs = F.softmax(logits, dim=1)
#         probs = probs[:, 1]
#         target = target[:,1]
#         intersection = (probs * target).sum(dim=(1,2,3))
#         union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
#         dice = (2 * intersection + smooth) / (union + smooth)
#         return 1 - dice.mean()
# #
# # class dice_coefficient(torch.nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #     def forward(self, pred, target):
# #         intersection = torch.sum(pred * target, dim=(2, 3, 4))
# #         union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target, dim=(2, 3, 4))
# #         dice = (2 * intersection ) / (union + 1e-6)
# #         return dice.mean()    

# class dice_coefficient(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, logits, target):
#         probs = F.softmax(logits, dim=1)
#         pred  = probs.argmax(dim=1, keepdim=True).long()
#         label = target.argmax(dim=1, keepdim=True).long()
#         intersection = torch.sum(pred * label, dim=(2, 3, 4))
#         union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(label, dim=(2, 3, 4))
#         dice = (2 * intersection ) / (union + 1e-6)
#         return dice.mean()    



# class ComboLoss(nn.Module):
#     def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=50):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
#         self.dice = DiceLoss()
#         self.bce_weight = bce_weight
#         self.dice_weight = dice_weight

#     def forward(self, logits, target):
#         loss_bce = self.bce(logits, target.float())
#         loss_dice = self.dice(logits, target)
#         return self.bce_weight * loss_bce + self.dice_weight * loss_dice

class dice_loss_test(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, target, smooth=1e-6):
        probs = F.softmax(logits, dim=1)
        probs = probs[:, 1]
        target = target[:,1]
        intersection = (probs * target).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
class dice_coefficient_test(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        pred  = probs.argmax(dim=1, keepdim=True).long()
        label = target.argmax(dim=1, keepdim=True).long()
        intersection = torch.sum(pred * label, dim=(2, 3, 4))
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(label, dim=(2, 3, 4))
        dice = (2 * intersection ) / (union + 1e-6)
        return dice.mean()
    
def train_model_test(net, trainloader, epoch, writer, optimizer, scheduler, DEVICE):
    scaler = GradScaler()
    loss_fn = dice_loss_test() 
    net.train()
    dice_loss_sum = 0
    for batch in tqdm(trainloader, "Training"):
        images = batch['mri'][tio.DATA].to(DEVICE)
        labels = batch['segmentation'][tio.DATA].to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            prediction = net(images)
            loss_batch = loss_fn(prediction, labels)
        scaler.scale(loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()
        dice_loss_sum += loss_batch
    scheduler.step()
    # current_lr = optimizer.param_groups[0]['lr']
    # writer.add_scalar(f'learning rate', current_lr, epoch)
    average_dice_loss = dice_loss_sum / len(trainloader)
    writer.add_scalar(f'dice_loss', average_dice_loss, epoch)
    print(f"Epoch {epoch+1}, average dice loss: {average_dice_loss}")