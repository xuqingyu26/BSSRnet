import torch
import torch.nn as nn
import numpy as np
from skimage import morphology
import torch.nn.functional as F



def space_to_depth(img, r):
    B, C, H, W = img.shape
    h = int(H/r)
    w = int(W/r)
    img = img.view(B, C, h, r, w, r).permute(0, 1, 3, 5, 2, 4).contiguous()
    img = img.view(B, C*r**2, h, w)
    return img

class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output =self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)          #B*C*H*W

#residual dense block
class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def forward(self, x):
        out = self.body(x)
        return out + x


def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)    #先是去除size<20的区域，后进行补洞处理，将洞小于10的填满
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx, 0, :, :] = buffer[3:-3, 3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)

#N, 81, 8, H, W
class Slice(nn.Module):
    def __init__(self):
        super(Slice,self).__init__()
    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()
        N,_, H, W =guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0,H), torch.arange(0,W)])
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1          #Norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, padding_mode= 'border')
        return coeff.squeeze(2)  #N,81,H,W


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        B, C, H, W = full_res_input.shape
        img = F.unfold(full_res_input, 3, padding=1)
        img = img.permute(0, 2, 1).contiguous().view(B*H*W, 1, 27)       # B*H*W, 1, 27
        kernel = coeff.view(B, 3, 27, H, W)
        kernel = kernel.permute(0, 3, 4, 2, 1).contiguous().view(B*H*W, 27, 3)                  # B*H*W, 27, 3
        out = torch.bmm(img, kernel)
        out = out.view(B, H, W, 3).permute(0, 3, 1, 2)
        return out


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(int(inc),int(outc),kernel_size,padding=padding,stride=stride,bias=use_bias)
        self.activation = activation() if activation else None

    def forward(self,x):
        x=self.conv(x)
        if self.activation:
            x=self.activation(x)

        return x

#generate guidemap
class GuideNN(nn.Module):
    def __init__(self):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(3, 16, 1, 0)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh) ##输出的guidemap值在 [-1，1] 间

    def forward(self,x):
        return self.conv2(self.conv1(x))

#求BG
class Coeffs(nn.Module):
    def __init__(self):
        super(Coeffs, self).__init__()
        self.relu = nn.ReLU()
        self.rdb1 = nn.Sequential(
            RDG(G0=64, C=6, G=16, n_RDB=1)
        )
        self.rdb2 = nn.Sequential(
            ConvBlock(128, 64, 1, padding=0, stride=1),
            RDG(G0=64, C=6, G=16, n_RDB=1)
        )
        self.reconstruct1 = nn.Sequential(
            ResB(64),
            nn.Conv2d(64, 256, 1, 1, 0, bias=True),
            nn.PixelShuffle(2),
        )

        self.reconstruct2 = nn.Sequential(
            ConvBlock(192, 64, 1, padding=0, stride=1),
            ResB(64),
            nn.Conv2d(64, 64*4, 1, 1, 0, bias=True),
            nn.PixelShuffle(2),
        )

        self.splat_features1 = nn.Sequential(
            ConvBlock(64,  64, 3, padding=1, stride=1, use_bias=False),
            ConvBlock(64, 128, 3, stride=2)
        )

        self.splat_features2 = nn.Sequential(
            ConvBlock(128, 64, 1, padding=0, stride=1, use_bias=False),
            ConvBlock(64, 128, 3, stride=2)
        )

        local_features = []
        local_features.append(ConvBlock(128, 64, 1, padding=0, stride=1))
        local_features.append(ConvBlock(64, 64, 3, padding=1, use_bias=False))
        local_features.append(ConvBlock(64, 64, 3, padding=1, activation=None, use_bias=False))
        self.local_features = nn.Sequential(*local_features)

    def forward(self,lowres_input):
        x = self.rdb1(lowres_input)                          # b, 64, h, w
        splat_features1 = self.splat_features1(x)                    # b, 128, h/2, w/2
        splat_features2 = self.splat_features2(splat_features1)      # b, 128, h/4, w/4
        y = self.rdb2(splat_features2)                    # b, 64, h, w
        features1 = self.reconstruct1(y)                             # b, 64, h/2, w/2
        features1 = torch.cat([features1, splat_features1], 1)       # b, 192, h/2, w/2
        features2 = self.reconstruct2(features1)                     # b, 64, h, w
        features2 = torch.cat([features2, lowres_input], 1)          # b, 128, h, w
        local_features = self.local_features(features2)
        fusion = self.relu(local_features)

        return fusion


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape,self).__init__()
        self.conv_out = nn.Sequential(
            ConvBlock(64, 64, 1, padding=0),
            ConvBlock(64, 648, 1, padding=0, activation=None)
        )
    def forward(self, fusion):
        x = self.conv_out(fusion)
        N, C, H, W = x.shape
        x = x.view(N, 81, 8, H, W)               #B x Channels x D x H x W

        return x

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=1)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, groups=1)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(channels)
        self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)

    def __call__(self, x_left, x_right):
        Q = self.b1(self.rb(x_left))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.b2(self.rb(x_right))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),  # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))  # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)  # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))  # (B*H) * Wr * Wl

        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1  # valid mask 不需要梯度，所以返回一个新的变量 不具有grad
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)

        V_right_to_left = torch.sum(M_left_to_right.detach(), 1) > 0.1  # valid mask 不需要梯度，所以返回一个新的变量 不具有grad
        V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
        V_right_to_left = morphologic_process(V_right_to_left)

        buffer_left = self.b3(x_right).permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_left = torch.bmm(M_right_to_left, buffer_left).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W
        out_left = self.fusion(torch.cat((buffer_left, x_left, V_left_to_right), 1))

        buffer_right = self.b3(x_left).permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_right = torch.bmm(M_left_to_right, buffer_right).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W
        out_right = self.fusion(torch.cat((buffer_right, x_right, V_right_to_left), 1))

        if self.training:
            M_left_right_left = torch.bmm(M_right_to_left, M_left_to_right)
            M_right_left_right = torch.bmm(M_left_to_right, M_right_to_left)
            return out_left, out_right, \
                   (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
                   (M_left_right_left.view(b, h, w, w), M_right_left_right.view(b, h, w, w)), \
                   (V_left_to_right, V_right_to_left)

        else:
            return out_left, out_right


class BSSRnet(nn.Module):
    def __init__(self, params):
        super(BSSRnet, self).__init__()
        self.upscale_factor = params['scale_factor']


        ### 提取特征
        self.init_feature = nn.Conv2d(3,64,3,1,1,bias=True)
        self.deep_feature = RDG(G0=64,C=4,G=24,n_RDB=4)

        #stereo attention
        self.pam = PAM(64)

       # self.fusion = nn.Conv2d(128, 64, 1, 1, 0, bias=True)
        self.rec = RDG(G0=64, C=6, G=16, n_RDB=3)
        self.res1 = ResB(48)
        self.res2 = ResB(48)
        self.res3 = ResB(48)
        self.reconstruct1 = nn.Sequential(
            ResB(112),
            nn.Conv2d(112, 3 * self.upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(self.upscale_factor),
        )
        self.reconstruct2 = nn.Sequential(
            ResB(112),
            nn.Conv2d(112, 3 * self.upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(self.upscale_factor),
        )
        self.reconstruct3 = nn.Sequential(
            ResB(112),
            nn.Conv2d(112, 3 * self.upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(self.upscale_factor),
        )
        # self.upscale = nn.Sequential(
        #     nn.Conv2d(64, 64 * self.upscale_factor ** 2, 1, 1, 0, bias=True),
        #     nn.PixelShuffle(self.upscale_factor),
        #     #nn.Conv2d(64, 3, 3, 1, 1, bias=True),
        # )

        #bilateral operators
        self.coeffs = Coeffs()
        self.reshape1 = Reshape()
        self.reshape2 = Reshape()
        self.reshape3 = Reshape()
        self.guide1 = GuideNN()
        self.guide2 = GuideNN()
        self.guide3 = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, x_left, x_right):
        #bicubic upsampling
        x_left_upscale = F.interpolate(x_left, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        #align_corners如果为True，
        # 则图片角点像素的中心对齐，否则为角点对齐，插值使用边界外插值填充。

        #feature extraction
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left = self.deep_feature(buffer_left)
        buffer_right = self.deep_feature(buffer_right)

        ##
        if self.training:
            pam_left, pam_right, (M_right_to_left, M_left_to_right),\
            (M_left_right_left,  M_right_left_right),(V_left_to_right, V_right_to_left) \
                = self.pam(buffer_left, buffer_right)
        else:
            pam_left, pam_right \
                = self.pam(buffer_left, buffer_right)

        ##
        feature_left = self.rec(pam_left)
        feature_right = self.rec(pam_right)
        ##
        bilateral_feature_left = self.coeffs(pam_left)
        bilateral_feature_right = self.coeffs(pam_right)

        x_left_feature = self.res1(space_to_depth(x_left_upscale, 4))
        x_right_feature = self.res1(space_to_depth(x_right_upscale, 4))

        res_left1 = self.reconstruct1(torch.cat([x_left_feature, feature_left], 1))
        res_right1 = self.reconstruct1(torch.cat([x_right_feature, feature_right], 1))

        guide_left = self.guide1(x_left_upscale)
        guide_right = self.guide1(x_right_upscale)

        bilateral_grid_left1 = self.reshape1(bilateral_feature_left)
        bilateral_grid_right1 = self.reshape1(bilateral_feature_right)

        slice_coeffs_left1 = self.slice(bilateral_grid_left1, guide_left)
        slice_coeffs_right1 = self.slice(bilateral_grid_right1, guide_right)
        ##

        out_left1 = (self.apply_coeffs(slice_coeffs_left1, x_left_upscale + res_left1) + x_left_upscale + res_left1).contiguous()
        out_right1 = (self.apply_coeffs(slice_coeffs_right1, x_right_upscale + res_right1) + x_right_upscale + res_right1).contiguous()

        ##
        out_left1_feature = self.res2(space_to_depth(out_left1, 4))
        out_right1_feature = self.res2(space_to_depth(out_right1, 4))

        res_left2 = self.reconstruct2(torch.cat([out_left1_feature,feature_left],1))
        res_right2 = self.reconstruct2(torch.cat([out_right1_feature,feature_right],1))

        guide_left = self.guide2(out_left1)
        guide_right = self.guide2(out_right1)

        bilateral_grid_left2 = self.reshape2(bilateral_feature_left)
        bilateral_grid_right2 = self.reshape2(bilateral_feature_right)

        slice_coeffs_left2 = self.slice(bilateral_grid_left2, guide_left)
        slice_coeffs_right2 = self.slice(bilateral_grid_right2, guide_right)

        out_left2 = (self.apply_coeffs(slice_coeffs_left2, out_left1 + res_left2) + out_left1 + res_left2).contiguous()
        out_right2 = (self.apply_coeffs(slice_coeffs_right2, out_right1 + res_right2) + out_right1 + res_right2).contiguous()

        ##
        out_left2_feature = self.res3(space_to_depth(out_left2, 4))
        out_right2_feature = self.res3(space_to_depth(out_right2, 4))

        res_left3 = self.reconstruct3(torch.cat([out_left2_feature, feature_left], 1))
        res_right3 = self.reconstruct3(torch.cat([out_right2_feature, feature_right], 1))

        guide_left = self.guide3(out_left2)
        guide_right = self.guide3(out_right2)

        bilateral_grid_left3 = self.reshape3(bilateral_feature_left)
        bilateral_grid_right3 = self.reshape3(bilateral_feature_right)

        slice_coeffs_left3 = self.slice(bilateral_grid_left3, guide_left)
        slice_coeffs_right3 = self.slice(bilateral_grid_right3, guide_right)

        out_left = (self.apply_coeffs(slice_coeffs_left3, out_left2 + res_left3) + out_left2 + res_left3).contiguous()
        out_right = (self.apply_coeffs(slice_coeffs_right3,
                                        out_right2+ res_right3) + out_right2 + res_right3).contiguous()


        if self.training:
            return out_left, out_right,\
                   (M_right_to_left, M_left_to_right), (M_left_right_left,  M_right_left_right), (V_left_to_right, V_right_to_left)
        else:
            return out_left, out_right

if __name__ == "__main__":
    params={}
    params['scale_factor']=4
    net = BSSRnet(params=params)
    total = sum([param.nelement() for param in net.parameters()])
    print('  Number of params: %.2fM' % (total / 1e6))















