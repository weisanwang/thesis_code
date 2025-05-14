import torch
from torchsummary import summary
import time
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels, middle_channels,out_channels):
        super().__init__()
        
        self.step = torch.nn.Sequential(
            # Cpnv
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
    
# class dice_loss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, pred, target):
#         intersection = torch.sum(pred * target, dim=(2, 3, 4))
#         union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target, dim=(2, 3, 4))
#         dice = (2 * intersection ) / (union + 1e-6) 
#         # print('dice loss',dice.shape)
#         dice_loss = 1 - dice[:, 1].mean()
#         # return dice_loss.mean()
#         return dice_loss


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

# loss_fn = dice_loss()   

# Define Dice score function for evaluation
# class dice_coefficient(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, pred, target):
#         intersection = torch.sum(pred * target, dim=(2, 3, 4))
#         union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target, dim=(2, 3, 4))
#         dice = (2 * intersection ) / (union + 1e-6)
#         return dice.mean()    
# # dice_fn = dice_coefficient()


# def train_model(net, trainloader, epoch, writer, optimizer, scheduler, DEVICE):
#     scaler = GradScaler()
#     loss_fn = dice_loss() 
#     net.train()
#     dice_loss_sum = 0
#     for batch in tqdm(trainloader, "Training"):
#         images = batch['mri'][tio.DATA].to(DEVICE)
#         labels = batch['segmentation'][tio.DATA].to(DEVICE)
#         optimizer.zero_grad()
#         with autocast():
#             prediction = net(images)
#             loss_batch = loss_fn(prediction, labels)
#         scaler.scale(loss_batch).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         dice_loss_sum += loss_batch
#     scheduler.step()
#     # current_lr = optimizer.param_groups[0]['lr']
#     # writer.add_scalar(f'learning rate', current_lr, epoch)
#     average_dice_loss = dice_loss_sum / len(trainloader)
#     writer.add_scalar(f'dice_loss', average_dice_loss, epoch)
#     print(f"Epoch {epoch+1}, average dice loss: {average_dice_loss}")



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

