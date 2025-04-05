import time
import torch.nn.functional as nn
import torch.nn.functional as F
from tqdm import tqdm

from misc.util import *


def attnDiv(cams): # (128,3,4,4)for resnet|(128,3,17)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    orthogonal_loss = 0
    bs = cams.shape[0] # BatchSize
    num_part = cams.shape[1]
    cams = cams.view(bs, num_part, -1)
    cams = F.normalize(cams, p=2, dim=-1)
    mean = cams.mean(dim=-1).view(bs, num_part, -1).expand(size=[bs, num_part, cams.shape[-1]])
    cams = F.relu(cams-mean)
    
    for i in range(cams.shape[1]):
        for j in range(i+1, cams.shape[1]):
            orthogonal_loss += cos(cams[:,i,:].view(bs,1,-1), cams[:,j,:].view(bs,1,-1)).mean()
    return orthogonal_loss/(i*(i-1)/2)

    
def train(train_loader, model, criterion, optimizer, args):
    model.train()
    loss_keys = args['loss_keys']
    acc_keys  = args['acc_keys']
    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter  = {p: AverageMeter() for p in acc_keys}
    time_start = time.time()

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

    # for i, data in enumerate(train_loader):
    for i, data in progress_bar:
        # print("===", i, "/",len(train_loader),"===")
        inputs = data[0].cuda()
        target = data[1].cuda()
        
        output_dict = model(inputs, target)
        logits      = output_dict['logits']
        loss_values = [criterion['entropy'](logit.float(), target.long()) for logit in logits]
        # CAM（Class Activation Map）是一种可视化技术，用于理解卷积神经网络中的注意力分布。

        branch_cams = output_dict['cams']
        loss_values.append(attnDiv(branch_cams))

        # loss_values.append(args['loss_wgts'][0] * sum(loss_values[:4]) + args['loss_wgts'][1] * loss_values[-2] + args['loss_wgts'][2] * loss_values[-1])
        loss_values.append(args['loss_wgts'][0] * sum(loss_values[:2]) + args['loss_wgts'][1] * sum(loss_values[2:4]) + args['loss_wgts'][2] * loss_values[-2] + args['loss_wgts'][3] * loss_values[-1])
        # 第1~3为三个分支的loss,第4个为gate,第5个为attnDiv,第6个为total，b1-3与gate的logits在模型中给出
        multi_loss = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}
        acc_values = [accuracy(logit, target, topk = (1,))[0] for logit in logits]
        train_accs = {acc_keys[k] : acc_values[k] for k in range(len(acc_keys))}
        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter, train_accs, inputs.size(0))
        tmp_str = "< Training Loss >\n"
        for k, v in loss_meter.items(): 
            tmp_str = tmp_str + f"{k}:{v.value:.4f} "
        tmp_str = tmp_str + "\n< Training Accuracy >\n"
        for k, v in acc_meter.items():
            tmp_str = tmp_str + f"{k}:{v.value:.1f} "
        optimizer.zero_grad()
        # 反向传播
        loss_values[-1].backward()
        optimizer.step()
    
    time_eclapse = time.time() - time_start
    print(tmp_str + f"t:{time_eclapse:.1f}s")
    
    return loss_meter[loss_keys[-1]].value
