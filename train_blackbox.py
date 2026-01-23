# python train_blackbox.py --data-path ./dataset/cm/  -d cm -m resnet50.a1_in1k

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import timm
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score
import copy
from torch.utils.data import DataLoader
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import utils
import matplotlib.pyplot as plt
import os
import sys
import time
import math

from dataset.dataset import SkinDataset
# , busiDataset, cmmdDataset, idridDataset, cmDataset, nctDataset, edemaDataset, siimDataset

DEBUG = False


dataset_dict = {
    'isic2018': SkinDataset,
    # 'busi': busiDataset,
    # 'cmmd': cmmdDataset,
    # 'idrid': idridDataset,
    # 'cm': cmDataset,
    # 'nct': nctDataset,
    # 'edema': edemaDataset,
    # 'siim': siimDataset,
}

def train_net(model, config):

    # 前缀名
    print(config.unique_name)
    
    # 拿到 timm 为每个预训练模型定义的 官方预处理规范
    # 保证你的输入分布和 ImageNet 预训练时的输入分布一致
    # 否则会出现：模型是 ImageNet 预训练的，但输入分布不一致，导致性能莫名下降
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    
    # 构造 数据预处理/数据增强

    # 输入数据转成 PIL 图像对象
    transform_list = [transforms.ToPILImage()]
    # 随机裁剪 + 随机缩放
    transform_list.append(transforms.RandomResizedCrop(size=data_cfg['input_size'][-1], scale=(0.75, 1.0), ratio=(0.75, 1.33), interpolation=utils.get_interpolation_mode(data_cfg['interpolation'])))
    # 随机左右翻转
    transform_list.append(transforms.RandomHorizontalFlip())
    # 随机上下翻转
    transform_list.append(transforms.RandomVerticalFlip())
    # 把 PIL 图像转成 pytorch tensor
    transform_list.append(transforms.ToTensor())
    # isic2018 数据集特有的预处理
    if config.dataset == 'isic2018':
        transform_list.append(utils.gray_world())
    # 归一化 （保证输入落在模型熟悉的数值分布）
    transform_list.append(transforms.Normalize(mean=data_cfg['mean'], std=data_cfg['std']))


    # 训练集的数据预处理 / 数据增强
    train_transforms = transforms.Compose(transform_list)
    # 验证集的数据预处理（不需要数据增强）
    val_transforms = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=int(data_cfg['input_size'][-1]/data_cfg['crop_pct']), interpolation=utils.get_interpolation_mode(data_cfg['interpolation'])),
                                         transforms.CenterCrop(size=data_cfg['input_size'][-1]),
                                         transforms.ToTensor(),
                                         utils.gray_world() if config.dataset=='isic2018' else utils.identity(),
                                         transforms.Normalize(mean=data_cfg['mean'], std=data_cfg['std'])])


    trainset = dataset_dict[config.dataset](config.data_path, mode='train', transforms=train_transforms, flag=config.flag, debug=DEBUG, config=config)
    trainLoader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)

    valset = dataset_dict[config.dataset](config.data_path, mode='val', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config)
    valLoader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    testset = dataset_dict[config.dataset](config.data_path, mode='test', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config)
    testLoader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)


    writer = SummaryWriter(config.log_path+config.unique_name)


    if config.cls_weight == None:
        criterion = nn.CrossEntropyLoss().cuda() 
    else:
        # 加权交叉熵损失函数缓解类别不平衡问题
        lesion_weight = torch.FloatTensor(config.cls_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=lesion_weight).cuda()
    
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # 如果开启混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config.amp else None

    # 在正式训练开始之前，对“初始模型状态”做一次基线评估
    BMAC, acc, _ = validation(model, valLoader, criterion)
    print('BMAC: %.5f, Acc: %.5f'%(BMAC, acc))

    best_acc = 0
    for epoch in range(config.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, config.epochs))
        batch_time = 0
        epoch_loss = 0

        model.train()
        start = time.time()
        exp_scheduler = utils.exp_lr_scheduler_with_warmup(optimizer, init_lr=config.lr, epoch=epoch, warmup_epoch=config.warmup_epoch, max_epoch=config.epochs)

        for i, (data, label) in enumerate(trainLoader, 0):
            x1, target1 = data.float().cuda(), label.long().cuda()
            
            optimizer.zero_grad()

            if config.amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(x1)

                    loss = criterion(output, target1)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                output = model(x1)

                loss = criterion(output, target1)
                loss.backward()
                optimizer.step()
            

            epoch_loss +=  loss.item()

            batch_time = time.time() - start

            end = time.time()


            print(i, 'loss: %.5f, batch_time: %.5f' % (loss.item(), batch_time))
        
        print('[epoch %d] epoch loss: %.5f' % (epoch+1, epoch_loss/(i+1) ))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)



        if not os.path.isdir('%s%s/'%(config.cp_path, config.unique_name)):
            os.makedirs('%s%s/'%(config.cp_path, config.unique_name))
        
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), '%s%s/CP%d.pth'%(config.cp_path, config.unique_name, epoch+1))


        val_BMAC, val_acc, val_loss = validation(model, valLoader, criterion)
        writer.add_scalar('Val/BMAC', val_BMAC, epoch+1)
        writer.add_scalar('Val/Acc', val_acc, epoch+1)
        writer.add_scalar('Val/val_loss', val_loss, epoch+1)
        
        test_BMAC, test_acc, test_loss = validation(model, testLoader, criterion)
        writer.add_scalar('Test/BMAC', test_BMAC, epoch+1)
        writer.add_scalar('Test/Acc', test_acc, epoch+1)
        writer.add_scalar('Test/test_loss', test_loss, epoch+1)
                
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR/lr', lr, epoch+1)


        if val_BMAC >= best_acc:
            best_acc = val_BMAC
            if not os.path.exists(config.cp_path):
                os.makedirs(config.cp_path)
            torch.save(model.state_dict(), '%s%s/best.pth'%(config.cp_path, config.unique_name))
          

        print('save done')
        print('BMAC: %.5f/best BMAC: %.5f, Acc: %.5f'%(val_BMAC, best_acc, val_acc))


        
def validation(model, dataloader, criterion):
    
    net = model

    net.eval()
    losses = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data, label = data.float(), label.long()

            inputs, labels = data.cuda(), label.cuda()
            pred = net(inputs)

            loss = criterion(pred, labels)
            losses += loss.item()
            
            _, label_pred = torch.max(pred, dim=1)
            
            
            pred_list = np.concatenate((pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, label.cpu().numpy().astype(np.uint8)), axis=0)
    
    BMAC = balanced_accuracy_score(gt_list, pred_list)
    correct = np.sum(gt_list == pred_list)
    acc = 100 * correct / len(pred_list)


    return 100 *BMAC, acc, losses/(i+1)




if __name__ == '__main__':
    
    # === 定义命令行参数 ===

    parser = OptionParser()

    # 训练轮次 epochs
    parser.add_option('-e', '--epochs', 
                      dest='epochs', 
                      default=100, type='int',
                      help='number of epochs')
    # 批处理大小 batch size
    parser.add_option('-b', '--batch_size', 
                      dest='batch_size', 
                      default=128, type='int', 
                      help='batch size')
    # 预热轮次 warmup epochs
    parser.add_option('--warmup_epoch', 
                      dest='warmup_epoch', 
                      default=5, type='int')
    # 优化器选择
    parser.add_option('--optimizer', 
                      dest='optimizer', 
                      default='sgd', type='str')
    # 学习率 learning rate
    parser.add_option('-l', '--lr', 
                      dest='lr', 
                      default=0.01, type='float', 
                      help='learning rate')
    # 是否加载，从自己训练好的 checkpoint 继续
    parser.add_option('-c', '--resume', 
                      dest='load', 
                      default=False, type='str', 
                      help='load pretrained model')
    # 模型保存路径
    parser.add_option('-p', '--checkpoint-path', 
                      dest='cp_path', 
                      default='./checkpoint/', type='str', 
                      help='checkpoint path')
    # 日志保存路径
    parser.add_option('-o', '--log-path',  
                      dest='log_path', 
                      default='./log/', type='str',
                      help='log path')
    # 使用的模型
    parser.add_option('-m', '--model', 
                      dest='model',
                      default='resnet50.a1_in1k', type='str', # We find vit.orig_in21k is better than CLIP weights
                      help='use which model in [vit_base_patch16_224.orig_in21k, resnet50.a1_in1k]')
    # 是否使用线性探测微调
    # 人话：是否冻结其它层，只训练最后的分类头，用来评估“特征本身好不好”
    # 不让模型重新学特征，这个预训练 backbone 的特征，能不能直接把类别分开
    parser.add_option('--linear-probe', 
                      dest='linear_probe', 
                      action='store_true', # 默认布尔型为True
                      help='if use linear probe finetuning')
    # 数据集名称
    parser.add_option('-d', '--dataset', 
                      dest='dataset', 
                      default='isic2018', type='str', 
                      help='name of datasets')
    # 数据集路径
    parser.add_option('--data-path', 
                      dest='data_path', 
                      default='./dataset/', type='str', 
                      help='the path of the dataset')
    # 前缀名
    parser.add_option('-u', '--unique_name', 
                      dest='unique_name', 
                      default='test', type='str', 
                      help='name prefix')
    # 交叉验证折数
    parser.add_option('--flag', 
                      dest='flag', 
                      default=2, type='int', 
                      help='fold for cross-validation')
    # gpu id
    parser.add_option('--gpu', 
                      dest='gpu', 
                      default='0', type='str')
    # 是否使用混合精度训练
    parser.add_option('--amp', 
                      action='store_true', 
                      help='if use mixed precision training')

    (config, args) = parser.parse_args()
    
    # 使用哪个 gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    # 日志路径和模型保存路径，按数据集分类存储
    config.log_path = config.log_path + config.dataset + '/'
    config.cp_path = config.cp_path + config.dataset + '/' 
    
    # 打印使用的模型
    print('use model:', config.model)
    
    # 定义每个数据集的类别数
    # 决定模型的输出维度
    num_class_dict = {
        'isic2018': 7,
        'busi': 3,
        'cmmd': 2,
        'idrid': 5,
        'cm': 2,
        'nct': 9,
        'edema': 2,
        'siim':2
    }
    # 定义各数据集的类别权重
    # 决定损失函数对每一类的惩罚力度，权重大，该类更少更重要
    cls_weight_dict = {
        'isic2018': [1.2855, 0.2134, 2.7835, 4.3753, 1.3018,12.4410, 10.0755], 
        'busi': [3.1579,  0.9611, 2.0635], 
        'cmmd': [3.7037, 1.3776],
        'idrid': [2.0000,13.4400, 2.0000, 3.6129, 5.4194 ],
        'cm': [1.985, 0.668],
        'nct': [0.6, 0.94, 2.35, 1.26, 0.77, 1.35, 1.08, 1.89, 0.65],
        'edema': [1.206, 0.854],
        'siim': [0.6423 , 2.2568]
    }
    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]

    # 创建网络模型
    net = timm.create_model(config.model, pretrained=True, num_classes=config.num_class)

    # 如果开启 linear probe，则冻结其它层，只训练最后的分类头
    if config.linear_probe:
        # 对模型中的每一个参数
        for name, param in net.named_parameters():
            # 如果是 Resnet 的 fc 层
            if 'fc' in name and 'resnet' in config.model:
                # 参与训练，更新参数
                param.requires_grad = True
            # 如果是 ViT 的 head 层
            elif 'head' in name and 'vit' in config.model:
                # 参与训练，更新参数
                param.requires_grad = True
            # 如果是其它层
            else:
                # 不参与训练，冻结参数
                param.requires_grad = False

    # 参与训练的参数统计       参数张量p中元素个数  对于net中的参数张量p   如果p是可训练的
    print('num of params', sum(p.numel() for p in net.parameters() if p.requires_grad))

    # 加载已训练好的 checkpoint 继续训练
    if config.load:
        net.load_state_dict(torch.load(config.load))
        print('Model loaded from {}'.format(config.load))

    net.cuda()

    # === 以上为准备工作 ===

    # 开始训练
    train_net(net, config)

    print('done')
        

