import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb
import os
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class ContraLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContraLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def phone_mat(anchor_frame, target_frame):
    """
    """
    anchor_size = len(anchor_frame)
    target_size = len(target_frame)
    mat = torch.randn(anchor_size, target_size)
    for i in range(anchor_size):
        for j in range(0, target_size):
            if anchor_frame[i] == target_frame[j]:
                mat[i][j] = 0
            else:
                mat[i][j] = 1
    return mat


def helper(x, y):
    n = x.shape[0]
    m = y.shape[0]
    x = x.unsqueeze(1).repeat(1, m)
    y = y.unsqueeze(0).repeat(n, 1)
    return (x != y) + 0


class MyContraLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, context_feature, target_feature, mask_index, phone_label):
        """
        compute contrastive loss
        step1：Context_feature, Target_feature, Mask_index ——> anchors, negative
            step1.1: 根据Mask_index从Context_feature中取出anchor并拼在一起
                Context_feature,Mask_index ——> anchors(B*N,dim)  #假设每个Context_feature有N个Mask
            step1.2: 直接拼在一起
                Target_feature ——> negative(B*T,dim)
            step1.3: 根据Mask_index从target_feature中取出positive并拼在一起,用作后面算分子
                Target_feature,Mask_index ——> positive_sam(B*N,dim)
        #通过step1得到了两个二维矩阵，下面用F.cosine_similarity算出余弦相似度
        step2: anchors, negative ——> sim_mat(B*N,B*T)
        step3: sim_mat, phone_label ——> final_sim(B*N,B*T-same_phone) ——> loss #用(B*N,B*T)的phone矩阵去做mask
        Args:
            context_feature: (B,len,dim)
            target_feature: (B,len,dim)
            mask_index: (B,len)
            phone_label: (B,len)

        Returns:loss

        """
        #device = torch.device('cuda')
        device = (torch.device('cuda')
                  if context_feature.is_cuda
                  else torch.device('cpu'))

        if len(context_feature.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(target_feature.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(context_feature.shape) > 3:
            context_feature = context_feature.view(context_feature.shape[0], context_feature.shape[1], -1)

        if len(target_feature.shape) > 3:
            target_feature = target_feature.view(target_feature.shape[0], target_feature.shape[1], -1)

        # step1
        k=100 #负例数
        target_length = target_feature.shape[1]
        phone_length = phone_label.shape[1]

        diff = phone_length - target_length
        if diff >= 0:
            phone_label = phone_label[:, 0:target_length]
        else:
            p = torch.zeros(target_feature.shape[0], -diff).to(device)
            phone_label = torch.cat((phone_label, p), 1)
        mask_index = torch.unique(mask_index)
        batch = context_feature.shape[0]
        #mask_index = mask_index.unsqueeze(0).repeat(batch, 1)
        num = mask_index.numel()

        phone_label_n = phone_label[:, mask_index]
        #phone_label_n = phone_label[torch.arange(batch)[:, None], mask_index]
        phone_t = torch.cat(torch.unbind(phone_label, dim=0), dim=0)  # phone_label (B,T)-->(B*T)
        phone_n = torch.cat(torch.unbind(phone_label_n, dim=0), dim=0)
        #phone_label_mat = torch.from_numpy(helper(phone_n, phone_t))
        phone_label_mat = helper(phone_n, phone_t)  # (B*N, B*T)
        # #print("phone_label_mat:",phone_label_mat)

        context_feature_mask = context_feature[:, mask_index, :]  # (B,N,dim)
        #context_feature_mask = context_feature[torch.arange(batch)[:, None], mask_index]
        anchors = torch.cat(torch.unbind(context_feature_mask, dim=0), dim=0)  # (B*N,dim)
        negative = torch.cat(torch.unbind(target_feature, dim=0), dim=0)  # (B*T,dim)
        positive = target_feature[:, mask_index, :]  # (B,N,dim)
        #positive = target_feature[torch.arange(batch)[:, None], mask_index]
        positive_sam = torch.cat(torch.unbind(positive, dim=0), dim=0)  # (B*N,dim)

        # step2
        # res_numerator = F.cosine_similarity(anchors.unsqueeze(1), positive_sam.unsqueeze(0), dim=-1)
        # res_denominator_before = F.cosine_similarity(anchors.unsqueeze(1), negative.unsqueeze(0), dim=-1)

        anchor_norm = torch.norm(anchors, dim=1, keepdim=True)
        contrast_norm = torch.norm(positive_sam, dim=1, keepdim=True)
        ne_contrast_norm = torch.norm(negative, dim=1, keepdim=True)
        anchor_dot_contrast = torch.div(torch.matmul(anchors, positive_sam.T),anchor_norm * contrast_norm.T)
        res_numerator = torch.acos(anchor_dot_contrast)
        anchor_dot_necontrast = torch.div(torch.matmul(anchors, negative.T),anchor_norm * ne_contrast_norm.T)
        res_denominator_before = torch.acos(anchor_dot_necontrast)

        sim_numerator = torch.exp(torch.diag(torch.div(res_numerator, self.temperature)))  # （B*N)
        sim_denominator_before = torch.exp(torch.div(res_denominator_before, self.temperature))  # （B*N,B*T)

        # step3
        sim_denominator = sim_denominator_before * phone_label_mat.to(device)  # (B*N,B*T-same_phone) 再想办法变成(B*N,K)
        #sim_denominator = sim_denominator_before
        sim_sort, _ = sim_denominator.sort(1,True)
        #sim_sort=sim_denominator
        sim_100 = sim_sort[:,0:k]
        # sim_denominator = sim_denominator_before
        sim_denominator_final = sim_100.sum(1)  # (B*N)
        logit = torch.log(torch.div(sim_numerator, sim_denominator_final))
        loss = (-1 / (num*batch)) * logit.sum()  # 随便sum一下，到时候再改

        return loss
