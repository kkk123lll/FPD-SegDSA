from pickle import NONE
import torch
import torch.nn as nn
import torch.nn.functional as F
from fpd_seg.architectures.utils import InitWeights
from mamba_ssm import Mamba

class Res_conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, is_BN = True):
        super(Res_conv, self).__init__()
        
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1) if in_c != out_c else None  # 1x1卷积改变通道数
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp))
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.conv11 is not None:
            x = self.conv11(x)  # 改变通道数
        res = x
        x = self.conv(x)
        out = x + res  # 残差连接
        out = self.relu(out)
        return out

class Conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0, is_BN = True):
        super(Conv, self).__init__()
        
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1) if in_c != out_c else None
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))       

    def forward(self, x):
        if self.conv11 is not None:
            x = self.conv11(x)
        out = self.conv(x)      
        return out

    
class Seq_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dp = 0,is_BN = True):
        super().__init__()
        self.SeqConv = Res_conv(in_channels, out_channels,dp = dp,is_BN = is_BN)
    def forward(self, x):
        sequence = []
        for i in range(x.shape[2]):  # channel dimension
            image = self.SeqConv(x[:,:,i,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences  # [64,8,32,64,64]


class Down(nn.Module):
    def __init__(self, in_c, out_c,dp,is_BN = True):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class Sequence_down(nn.Module):
    def __init__(self, in_c, out_c,dp,is_BN = True):
        super(Sequence_down, self).__init__()
        self.down = Down(in_c, out_c,dp,is_BN =is_BN)

    def forward(self, x):
        sequence = []
        for i in range(x.shape[1]):
            image = self.down(x[:,i,:,:,:]) #'b,t,c,h,w' -> 'b,c,h,w'
            sequence.append(image)
        sequences = torch.stack(sequence, dim=1)

        return sequences




class BiConvGRUCell(nn.Module):
    """
        ICLR2016: Delving Deeper into Convolutional Networks for Learning Video Representations
        url: https://arxiv.org/abs/1511.06432
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, cuda_flag=True):
        super(BiConvGRUCell, self).__init__()
        self.input_channels  = input_channels
        self.cuda_flag  = cuda_flag
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        padding = self.kernel_size // 2
        self.reset_gate  = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.output_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        # init
        # for m in self.state_dict():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)
        self.mambalayer = MambaLayer(input_channels + hidden_channels, d_state=16, d_conv=4, expand=2)

    def forward(self, x, hidden):
        if hidden is None:
           size_h    = [x.data.size()[0], self.hidden_channels] + list(x.data.size()[2:])
           if self.cuda_flag:
              hidden = torch.zeros(size_h).cuda()
           else:
              hidden = torch.zeros(size_h)

        inputs       = torch.cat((x, hidden), dim=1)
        # inputs = self.mambalayer(inputs)
        reset_gate   = torch.sigmoid(self.reset_gate(inputs))
        update_gate  = torch.sigmoid(self.update_gate(inputs))

        reset_hidden = reset_gate * hidden        
        reset_inputs = torch.tanh(self.output_gate(torch.cat((x, reset_hidden), dim=1)))        
        new_hidden   = (1 - update_gate)*reset_inputs + update_gate*hidden

        return new_hidden

class BiConvGRU(nn.Module):
    def __init__(self, input_channels,output_channels, is_down=True,dp=0.2,is_BN = True) :
        super().__init__()
        
        self.input_channels = input_channels
        self.down = Sequence_down(input_channels*2,output_channels,dp,is_BN = is_BN) if is_down else None
        self.convgru_forward = BiConvGRUCell(output_channels, output_channels, 3)
        self.convgru_backward = BiConvGRUCell(output_channels, output_channels, 3)
        self.bidirection_conv = nn.Conv2d(2*output_channels, output_channels, 3, 1, 1)
        # self.fuse_conv = nn.Conv2d(2*output_channels, output_channels, 1)

    def forward(self, x, s = None):
        if self.down is not None:
            B, S, _, H, W = x.shape
            s = F.interpolate(s, size=(self.input_channels, H, W),
                        mode='trilinear', align_corners=False)   # 对S(输入序列)进行插值操作，实现输入x和s在空间和时间维度上的对齐
            
            x = self.down(torch.cat([s, x], dim=2))
        
        sequence_forward = []
        image = x[:,0,:,:,:]   # 获取输入x序列中的第一个
        # forward
        for i in range(x.shape[1]):  # 遍历序列中的每个图像
            image = image.detach()        # 断开计算图
            image = self.convgru_forward(x[:,i,:,:,:], image) #'b,t,c,h,w' -> 'b,c,h,w'  # 序列维度上的操作
            sequence_forward.append(image)
       

        # backward
        sequence_backward = []
        image = x[:,-1,:,:,:]  
        for i in range(x.shape[1]):
            image = image.detach()        # 断开计算图
            image = self.convgru_backward(x[:,x.shape[1]-1-i,:,:,:], image)
            sequence_backward.append(image)

        # image = sequence_forward[-1] 

        # sequence_backward = []
        # for i in range(x.shape[1]):
        #     image = self.convgru_backward(sequence_forward[x.shape[1]-1-i], image)
        #     sequence_backward.append(image)

        #连接
        sequence_backward = sequence_backward[::-1]  # 倒序
        sequence = []
        for i in range(x.shape[1]):
            image = torch.tanh(self.bidirection_conv(torch.cat((sequence_forward[i], sequence_backward[i]), dim=1)))  
            # image = self.bidirection_conv(torch.cat((sequence_forward[i], sequence_backward[i]), dim=1))
            sequence.append(image)


        # sequences = torch.stack(sequence_backward, dim=1)#'b,c,h,w'->'b,t,c,h,w'
        # sequences = torch.stack(sequence_forward, dim=1)#'b,c,h,w'->'b,t,c,h,w'
        sequences = torch.stack(sequence, dim=1)#'b,c,h,w'->'b,t,c,h,w'4  # 
        # last_sequence = self.fuse_conv(torch.cat([sequence_forward[-1], sequence_backward[-1]], dim=1))
        last_sequence = torch.max(sequences, dim=1)[0]  # 压缩成2D图像

        return sequences, last_sequence


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c,is_BN = True):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        # self.norm = nn.BatchNorm2d(out_c)
        self.norm =nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c)
        # self.norm = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1+x2+x3)
        return out


class Up(nn.Module):
    def __init__(self, in_c, out_c, dp=0,is_BN = True):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            # nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            nn.BatchNorm2d(out_c) if is_BN else nn.InstanceNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x



class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False,is_BN = True):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c,is_BN = is_BN)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = Res_conv(out_c, out_c, dp,is_BN= is_BN )
        if self.is_up == True:
            self.up = Up(out_c, out_c//2, dp,is_BN= is_BN)
        if self.is_down == True:
            self.down = Down(out_c, out_c*2,dp,is_BN= is_BN)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


#################################################################

    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba_type="v3",
                # nslices=num_slices,
        )
    
    def forward(self, x):
        
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        # out = x_out + x_skip
                
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class TemporalAttentionPooling(nn.Module):
    """
    Temporal Attention Pooling (TAP)
    输入:  x [B, T, C, H, W]
    输出:  y [B, C, H, W]
    """
    def __init__(self, C, reduction=8):
        super().__init__()

        self.score = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # [B, C, T, 1, 1]
            nn.Conv3d(C, C // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(C // reduction, 1, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        x_ = x.permute(0, 2, 1, 3, 4)   # [B, C, T, H, W]
        s = self.score(x_)              # [B, 1, T, 1, 1]
        w = torch.softmax(s, dim=2)     # 时间权重

        y = (x_ * w).sum(dim=2)         # 时间加权求和
        return y                        # [B, C, H, W]


class MambaEncoder_SpatialDownsample(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 num_slices=64):
        super().__init__()

        # 下采样层：仅在 H, W 上下采样，不改变 D
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dims[0]),
            nn.GELU(),
        )
        self.temporal_pools = nn.ModuleList([
        TemporalAttentionPooling(dims[0]),
        TemporalAttentionPooling(dims[1]),
        TemporalAttentionPooling(dims[2]),
        TemporalAttentionPooling(dims[3]),])

        self.downsample_layers.append(stem)

        for i in range(3): 
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1],
                          kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 仅下采样H、W
                nn.GELU(),
            )
            self.downsample_layers.append(downsample_layer)

               
        self.stages = nn.ModuleList()
        for i in range(4):
            stage_layers = []
            for _ in range(depths[i]):
                stage_layers.append(MambaLayer(dim=dims[i], num_slices=num_slices))
                stage_layers.append(nn.InstanceNorm3d(dims[i], affine=False))  # 更合适的3D按通道归一
            self.stages.append(nn.Sequential(*stage_layers))

        # 输出层
        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            self.add_module(f'norm{i_layer}', layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        # x: [B, D, C, H, W] -> [B, C, D, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        # s: [B, C, D, H, W]，不需要permute

        outs = []
        for i in range(4):
            # ① 对 s 进行插值，使其与 x 的空间维匹配
            # s = s.float()
            # # mip_image = mip_image.float()
            # _, _, D, H, W = x.shape
            # s_resized = F.interpolate(s, size=(D, H, W), mode='trilinear', align_corners=False)
            # # mip_resized = F.interpolate(mip_image, size=(D, H, W), mode='trilinear', align_corners=False)

            # # ② 拼接 s 和 x（在通道维拼接）
            # x = torch.cat([x, s_resized], dim=1)

            # ③ 依次通过下采样、GSC 和 Mamba stage
            x = self.downsample_layers[i](x)
            # x = self.gscs[i](x)
            x = self.stages[i](x)

            # ④ 输出层处理
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                x_out = x_out.permute(0, 2, 1, 3, 4)  # [B, D, C, H, W]
                # x_out = torch.max(x_out, dim=1)[0]    # 对 D 取 max
                # ✅ 新写法（TAP）
                x_out = self.temporal_pools[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)

class MultiScaleFuseNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- 64x64 输出：1 -> 32 ----
        self.conv64 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # ---- 32x32 输出：32 -> 64 下采样 ----
        self.conv32 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # ↓2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ---- 16x16 输出：64 -> 128 下采样 ----
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), # ↓2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ---- 8x8 输出：128 -> 256 下采样 ----
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), # ↓2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x64 = self.conv64(x)    # [B,32,64,64]
        x32 = self.conv32(x64)  # [B,64,32,32]
        x16 = self.conv16(x32)  # [B,128,16,16]
        x8  = self.conv8(x16)   # [B,256,8,8]
        
        return x64, x32, x16, x8

    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TwoFeatureFusion(nn.Module):
    """
    简化版特征融合模块
    输入: t1, t2 (shape = [B, C, H, W])
    输出: [B, C, H, W]
    """
    def __init__(self, C):
        super().__init__()

        # 对齐层（保证融合前分布一致）
        self.align = nn.Conv2d(C, C, kernel_size=1, bias=False)

        # 轻量通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, C, 1, bias=False),
            nn.Sigmoid()
        )

        # 融合卷积
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(C * 2, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        # 输出 BN
        self.out_bn = nn.BatchNorm2d(C)

    def forward(self, t1, t2):

        # 1. 对齐
        t1a = self.align(t1)
        t2a = self.align(t2)

        # 2. 简单通道注意力（只用在 t2）
        t2_ca = t2a * self.channel_att(t2a)

        # 3. 融合
        x = torch.cat([t1a, t2_ca], dim=1)
        x = self.fuse_conv(x)

        # 4. 残差
        out = x + t1 + t2
        out = self.out_bn(out)

        return out
    
class ThreeFeatureFusion(nn.Module):
    """
    简化版三特征融合模块
    输入: t1, t2, t3 (shape = [B, C, H, W])
    输出: [B, C, H, W]
    """
    def __init__(self, C):
        super().__init__()

        # 对齐层（共享）
        self.align = nn.Conv2d(C, C, kernel_size=1, bias=False)

        # 轻量通道注意力（共享）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, C, 1, bias=False),
            nn.Sigmoid()
        )

        # 融合卷积（3C → C）
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(C * 3, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        # 输出 BN
        self.out_bn = nn.BatchNorm2d(C)

    def forward(self, t1, t2, t3):

        # 1. 对齐
        t1a = self.align(t1)
        t2a = self.align(t2)
        t3a = self.align(t3)

        # 2. 通道注意力（只作用在辅助特征）
        # t2_ca = t2a * self.channel_att(t2a)
        t3_ca = t3a * self.channel_att(t3a)

        # 3. 融合
        x = torch.cat([t1a, t2a, t3_ca], dim=1)
        x = self.fuse_conv(x)

        # 4. 残差
        out = x + t2 + t3
        out = self.out_bn(out)

        return out

##############################################################
class New_Mamba_Net(nn.Module):
    def __init__(self, input_reduce=None, num_classes=2, num_channels=1, feature_scale=2, mamba_channels=32,  dropout=0.2, fuse=True, out_ave=True):
        super(New_Mamba_Net, self).__init__()

        self.input_reduce = input_reduce
        # if input_reduce == "mean" or input_reduce == "min":
        #     self.num_channels = 1
        # elif isinstance(input_reduce, list):
        #     self.num_channels = len(input_reduce)
        # else:
        #     self.num_channels = num_channels
        self.num_channels = num_channels

        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]  # 32,64,128,256,512
        
        self.inc = Seq_conv(num_channels, filters[0],dp=dropout,is_BN=False)
        self.mamba_encoder = MambaEncoder_SpatialDownsample(
            in_chans=mamba_channels,
            depths=[2, 2, 2, 2],
            dims=[filters[0], filters[1], filters[2], filters[3]],
            drop_path_rate=0.,
            layer_scale_init_value=1e-6,
            out_indices=[0, 1, 2, 3],
        )
        self.mip_conv = MultiScaleFuseNet()
        self.mip_conv2 = MultiScaleFuseNet()
        self.fusion1 = TwoFeatureFusion(filters[0])
        self.fusion2 = TwoFeatureFusion(filters[1])
        self.fusion3 = TwoFeatureFusion(filters[2])
        self.fusion4 = TwoFeatureFusion(filters[3])
        # self.fusion1 = ThreeFeatureFusion(filters[0])
        # self.fusion2 = ThreeFeatureFusion(filters[1])
        # self.fusion3 = ThreeFeatureFusion(filters[2])
        # self.fusion4 = ThreeFeatureFusion(filters[3])

        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0]*3, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1]*4, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2]*4, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block40 = block(filters[3]*2, filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        
        # self.decoderUnet = DecoderUNet_CA(num_classes=num_classes)

    def forward(self, x):
        if self.input_reduce == "mean":
            x = torch.mean(x, dim=2, keepdim=True)
        elif self.input_reduce == "min":
            x, _ = torch.min(x, dim=2, keepdim=True)
        elif isinstance(self.input_reduce, list):
            s = torch.split(x, 1, dim=2)
            seq = []
            for i in self.input_reduce:
                seq.append(s[i])
            x = torch.cat(seq, dim=2)
        
        s = x.permute(0, 2, 1, 3, 4)   # X = 'b,1,c,h,w' ->  S = 'b,c,1,h,w'  x = [64,1,8,64,64], s  = [64,8,1,64,64]
        # mip_image = torch.min(s, dim=1, keepdim=True)[0].to(torch.uint8)       
        # 后半段 - 前半段（时间对称）
        diff = s[:, 4:] - s[:, :4]     # [B, 4, C, H, W]
        diff = torch.abs(diff)

        # MIP
        mip_image = torch.max(diff, dim=1)[0]
        orimip_image = torch.min(s, dim=1)[0]   # .to(torch.uint8)  
        # s = torch.cat([s, mip_image], dim=1)
        mip1, mip2, mip3, mip4 = self.mip_conv(mip_image)
        mip5, mip6, mip7, mip8 = self.mip_conv2(orimip_image)

        x = self.inc(x)                              # 获取输入特征（编码）  修改通道数，由1->32  x = [64,8,32,64,64]
        x_mamba = self.mamba_encoder(x)                    # 获取多尺度特征
        # x_mamba = self.mamba_encoder(x) 
        # # sc1, sc2, sc3, sc4 = x[0], x[1], x[2], x[3]  # 每个sc的形状：[B, C, D, H, W]
        sc1, sc2, sc3, sc4 = x_mamba[0], x_mamba[1], x_mamba[2], x_mamba[3]

        sc1 = self.fusion1(mip5, sc1)   # 融合minip和mamba特征
        sc2 = self.fusion2(mip6, sc2)
        sc3 = self.fusion3(mip7, sc3)
        sc4 = self.fusion4(mip8, sc4)
        # sc1 = self.fusion1(mip1, mip5, sc1)
        # sc2 = self.fusion2(mip2, mip6, sc2)
        # sc3 = self.fusion3(mip3, mip7, sc3)
        # sc4 = self.fusion4(mip4, mip8, sc4)

        x1_2, x_down1_2 = self.block1_2(sc1)
        x2_2, x_up2_2 = self.block2_2(sc2)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1 = self.block3_1(sc3)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        # x10 = self.cbam10(x10)   # ✅ CBAM1
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(torch.cat([sc4, mip4], dim=1))
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        # x11 = self.cbam11(x11)   # ✅ CBAM2
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40, mip3], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        # x12 = self.cbam12(x12)   # ✅ CBAM3
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31, mip2], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22, mip1], dim=1))
        # x13 = self.cbam13(x13)   # ✅ CBAM4
        if self.out_ave == True:
            output = (self.final1(x1_1)+self.final2(x10) +
                      self.final3(x11)+self.final4(x12)+self.final5(x13))/5
        else:
            output = self.final5(x13)

        return output  # output 
