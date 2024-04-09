import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn.functional import relu
    
    
class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        
        #filters = [64,128,256,512,1024]
        filters = [32,64,128,256,512]
        #filters = [16,32,64,128,256]
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        self.e11 = nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        self.e21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)

        self.e31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)

        self.e41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1) 

        self.e51 = nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1) 
        self.e52 = nn.Conv2d(filters[4], filters[4], kernel_size=3, padding=1) 
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(filters[0], out_channel, kernel_size=1)
        
        
    def forward(self, x):
       # Encoder
       xe11 = relu(self.e11(x))
       xe12 = relu(self.e12(xe11))
       xp1 = self.pool(xe12)

       xe21 = relu(self.e21(xp1))
       xe22 = relu(self.e22(xe21))
       xp2 = self.pool(xe22)

       xe31 = relu(self.e31(xp2))
       xe32 = relu(self.e32(xe31))
       xp3 = self.pool(xe32)

       xe41 = relu(self.e41(xp3))
       xe42 = relu(self.e42(xe41))
       xp4 = self.pool(xe42)

       xe51 = relu(self.e51(xp4))
       
       # bridge
       xe52 = relu(self.e52(xe51))
       
       # Decoder
       xu1 = self.upconv1(xe52)
       xu11 = torch.cat([xu1, xe42], dim=1)
       xd11 = relu(self.d11(xu11))
       xd12 = relu(self.d12(xd11))

       xu2 = self.upconv2(xd12)
       xu22 = torch.cat([xu2, xe32], dim=1)
       xd21 = relu(self.d21(xu22))
       xd22 = relu(self.d22(xd21))

       xu3 = self.upconv3(xd22)
       xu33 = torch.cat([xu3, xe22], dim=1)
       xd31 = relu(self.d31(xu33))
       xd32 = relu(self.d32(xd31))

       xu4 = self.upconv4(xd32)
       xu44 = torch.cat([xu4, xe12], dim=1)
       xd41 = relu(self.d41(xu44))
       xd42 = relu(self.d42(xd41))

       # Output layer
       out = self.outconv(xd42)

       return out

        

def main():
    model = UNet(3, 13)
    summary(model, input_size=(3, 288, 384))
    
if __name__ == "__main__":
    main()
