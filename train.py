from model_paper import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import torch
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description='BSSRNet Inference')
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=85, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    ###数据集路径
    parser.add_argument('--trainset_dir', type=str, default='/media/root/f/Qingyu/bsdataset/Flickr1024_patches32')
    parser.add_argument('--model_name', type=str, default='BSSR')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='/media/root/f/Qingyu/BSSR_ablation/log_hdr/BSSR_x4_epoch5.pth.tar')
    return parser.parse_args()

def train(train_loader, cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    net = BSSRnet(vars(cfg)).to(cfg.device)
    # net = nn.DataParallel(net)
    net.apply(weights_init_xavier)
    cudnn.benchmark = True
    scale = cfg.scale_factor

    if cfg.load_pretrain:

        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0':cfg.device})
            net.module.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    criterion_L1 = L1Loss()
    char_loss = L1_Charbonnier_loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad ==True], lr = cfg.lr)    #betas=(beta1,beta2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma= cfg.gamma)

    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs) if cfg.load_pretrain else range(cfg.n_epochs):
        torch.cuda.empty_cache()
        scheduler.step()
        lr = 2e-4 * (2**(-(idx_epoch//30)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for idx_iter, (HR_left,HR_right,LR_left,LR_right) in enumerate(train_loader):

           HR_left, HR_right, LR_left, LR_right = HR_left.to(cfg.device), HR_right.to(cfg.device),LR_left.to(cfg.device), LR_right.to(cfg.device)
           SR_left,SR_right, (M_right_to_left, M_left_to_right), _, (V_left, V_right) \
               = net(LR_left, LR_right)

           b, c, h, w = LR_left.shape
           loss_SR = char_loss(SR_left, HR_left) + char_loss(SR_right, HR_right)

           ### loss_consistency
           SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale,
                                       mode='bicubic', align_corners=False)
           SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale,
                                        mode='bicubic', align_corners=False)
           SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w),
                                    SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
           SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w),
                                     SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                     ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
           loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1),
                                    SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1),
                                    SR_right_resT * V_right.repeat(1, 3, 1, 1))

           ### loss_photometric
           Res_left = torch.abs(
               HR_left - F.interpolate(LR_left, scale_factor=scale, mode='bicubic', align_corners=False))
           Res_left = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
           Res_right = torch.abs(
               HR_right - F.interpolate(LR_right, scale_factor=scale, mode='bicubic', align_corners=False))
           Res_right = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)

           Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                 Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                 ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
           Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                  Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
           Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                      Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
           Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                       Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                       ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

           loss_photo = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                        criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))

           ### loss_cycle
           loss_cycle = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                        criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1),
                                     Res_right_cycle * V_right.repeat(1, 3, 1, 1))

           # ### loss_smoothness
           loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                    criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
           loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                    criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
           loss_smooth = loss_w + loss_h

           ### losses
           loss =  10 * loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           psnr_epoch.append(cal_psnr(HR_left[:, :, :, 64:].data.cpu(), SR_left[:, :, :, 64:].data.cpu()))
           loss_epoch.append(loss.data.cpu())
           if idx_epoch < 2:
               print(
                   'Iter--%4d, loss--%f, psnr--%f, loss_SR--%f, loss_photo--%f, loss_smooth__%f, loss_cycle--%f, loss_cons--%f' %
                   (idx_iter + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean()),
                    float(np.array(loss_SR.data.cpu()).mean()),
                    float(np.array(loss_photo.data.cpu()).mean()), float(np.array(loss_smooth.data.cpu()).mean()),
                    float(np.array(loss_cycle.data.cpu()).mean()), float(np.array(loss_cons.data.cpu()).mean())
                    ))

        scheduler.step()
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch--%4d, loss--%f, psnr--%f, loss_SR--%f, loss_photo--%f, loss_smooth__%f, loss_cycle--%f, loss_cons--%f' %
                  (idx_epoch+1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean()), float(np.array(loss_SR.data.cpu()).mean()),
                   float(np.array(loss_photo.data.cpu()).mean()), float(np.array(loss_smooth.data.cpu()).mean()),
                   float(np.array(loss_cycle.data.cpu()).mean()), float(np.array(loss_cons.data.cpu()).mean())
                   ))
            save_ckpt({
                'epoch':idx_epoch+1,
                'state_dict':net.state_dict(),
                'loss':loss_list,
                'psnr':psnr_list
            },save_path = 'log/',filename=cfg.model_name + '_x' +str(cfg.scale_factor) +
                                      '_epoch' +str(idx_epoch + 1) + '.pth.tar')
            if idx_epoch % 2 == 0:
                torch.save(net.state_dict(), 'log_3layer_1/' + 'BSSRnet_model3_x4' + '.pth')
            psnr_epoch = []
            loss_epoch = []


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)