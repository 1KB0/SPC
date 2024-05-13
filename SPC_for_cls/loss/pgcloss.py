import torch
import torch.nn as nn
import torch.nn.functional as F


class PseudoGroupContrast(nn.Module):
    def __init__(self):
        super(PseudoGroupContrast, self).__init__()
        self.projector_dim = 128
        self.class_num = 3
        self.queue_size = 125
        self.register_buffer("queue_list", torch.randn(self.queue_size*self.class_num, self.projector_dim))
        self.queue_list = F.normalize(self.queue_list, dim=1).cuda()
        self.temperature = 0.5

    @torch.no_grad()
    def _dequeue_and_enqueue(self, ema_feature, label):
        temp_list = self.queue_list[label*self.queue_size:(label+1)*self.queue_size, :]

        temp_list = torch.cat([ema_feature, temp_list], dim=0)
        temp_list = temp_list[0:self.queue_size, :]

        self.queue_list[label*self.queue_size:(label+1)*self.queue_size, :] = temp_list

    def forward(self, activation, ema_activation, pseudo_label, mask=None):
        feature = F.normalize(activation, dim=1)
        ema_feature = F.normalize(ema_activation, dim=1)
        label = torch.argmax(pseudo_label, dim=1)
        batch_size = feature.size(0)
        contrast_loss = 0.0

        current_queue_list = self.queue_list.clone().detach()

        l_pos = torch.einsum('nl,nl->n', [feature, ema_feature])
        l_pos = torch.exp(l_pos/self.temperature)

        for i in range(batch_size):
            current_f = feature[i:i+1]
            current_ema_f = ema_feature[i:i+1]
            current_c = label[i]
            ith_ema = l_pos[i:i+1]

            pos_sample = current_queue_list[current_c*self.queue_size:(current_c+1)*self.queue_size, :]
            neg_sample = torch.cat([current_queue_list[0:current_c*self.queue_size, :],
                                    current_queue_list[(current_c+1)*self.queue_size:, :]], dim=0)

            ith_pos = torch.einsum('nl,nl->n', [current_f, pos_sample])
            ith_pos = torch.exp(ith_pos/self.temperature)
            pos = torch.sum(ith_pos)

            ith_neg = torch.einsum('nl,nl->n', [current_f, neg_sample])
            ith_neg = torch.exp(ith_neg/self.temperature)
            neg = torch.sum(ith_neg)

            contrast_D = ith_pos/(ith_ema + pos + neg)
            contrast_1 = ith_ema/(ith_ema + pos + neg)
            contrast = torch.cat([contrast_D, contrast_1], dim=0)
            contrast = -torch.log(contrast + 1e-6)
            contrast = torch.sum(contrast)/(self.queue_size + 1)

            if mask is not None:
                assert batch_size == mask.size(0)
                contrast_loss = contrast_loss + mask[i] * contrast
            else:
                contrast_loss = contrast_loss + contrast

            self._dequeue_and_enqueue(current_ema_f, current_c)

        return contrast_loss/batch_size

