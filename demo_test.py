from model_paper import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import time
import numpy as np
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='/media/root/f/Qingyu/test')
    parser.add_argument('--dataset', type=str, default='Flickr1024')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()

def test(test_loader, cfg):
    net = BSSRnet(vars(cfg)).to(cfg.device)
    # net = nn.DataParallel(net)
    net = net.cuda()
    cudnn.benchmark = True
    #pretrained_dict = torch.load('./log' + '/BSSRnet_x4' + '.pth')
    #pretrained_dict = torch.load('./log2' + '/BSSR_biPAM_x4_epoch80' + '.pth.tar')
    # '/media/zf/sda3/BSSRNet/loop4/23.35loop4_BSSR_biPAM_x4_epoch80.pth.tar'
    # '/media/zf/sda3/BSSRNet/logloop/BSSR_biPAM_x4_epoch80.pth.tar'
    # '/media/zf/sda3/BSSRNet/log3/BSSR_biPAM_x4_epoch16.pth.tar'
    # '/media/zf/sda3/BSSRNet_ablation/log/BSSR_x4_epoch13.pth.tar'
    pretrained_dict = torch.load('./BSSR_x4_epoch80.pth.tar')
    dict = pretrained_dict['state_dict']
    net.load_state_dict(dict)
    net.eval()
    psnr_list = []
    start = time.time()
    with torch.no_grad():
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(test_loader):
            HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device), \
                                                   Variable(HR_right).to(cfg.device), \
                                                   Variable(LR_left).to(cfg.device), \
                                                   Variable(LR_right).to(cfg.device)
            b, c, h, w = LR_left.shape
            h_modify = h - h % 4
            w_modify = w - w % 4
            LR_left  = LR_left[:, :, : h_modify, : w_modify]
            LR_right = LR_right[:, :, : h_modify, : w_modify]
            HR_left = HR_left[:, :, : 4 * h_modify, : 4 * w_modify]
            # HR_right = HR_right[:, :, : 4 * h_modify, : 4 * w_modify]
            video_name = test_loader.dataset.file_list[idx_iter]
            output = net(LR_left, LR_right)
            SR_left = output[0]
            SR_right = output[1]
            SR_left = torch.clamp(SR_left, 0, 1)
            SR_right = torch.clamp(SR_right, 0, 1)

            psnr_list.append(cal_psnr(HR_left[:,:,:,:].cpu(), SR_left[:,:,:,:].cpu()))  # 计算左图的PSNR

            if not os.path.exists('./results_imgres_img/' + cfg.dataset):
                os.makedirs('./results_imgres_img/' + cfg.dataset)
            #
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save('./results_imgres_img/' + cfg.dataset + '/' + video_name + '_L.png')
            SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            SR_right_img.save('./results_imgres_img/' + cfg.dataset + '/' + video_name + '_R.png')

        ## print results
        # print(float(np.array(psnr_list).mean()))
        print( cfg.dataset + ' mean psnr: ', float(np.array(psnr_list).mean()))
        # print(time.strftime('%Y.%m.%d.%I.%M.%S',time.localtime(time.time())), cfg.dataset + ' mean psnr: ', float((np.array(psnr_list1)/2+np.array(psnr_list2)/2).mean()))
    end = time.time()
    running_time = (end - start) * 100 / (idx_iter+1)
    print("Running time per image pair:%f" % running_time)

def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    # dataset_list = ['Flickr1024', 'KITTI2012','KITTI2015','Middlebury']
    dataset_list = ['Middlebury']
    dataset_num = len(dataset_list)
    print(time.strftime('%Y.%m.%d.%I.%M.%S', time.localtime(time.time())))
    number = 10
    t = []
    for epoch in range(number):
        start = time.time()
        for i in range(dataset_num):
            dataset = dataset_list[i]
            cfg.dataset = dataset
            cfg.scale_factor = 4
            main(cfg)
        end = time.time()
        running_time = (end - start) * 100 / 5
        t.append(running_time)
    T = np.array(t[1:])
    print("Epoch:%f, Running time per image pair:%f" % (epoch,T.mean()))
