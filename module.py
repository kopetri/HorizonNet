import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model import HorizonNet
from inference import inference
from eval_general import test_general
import numpy as np


class HorizonModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.net = HorizonNet(self.opt.backbone, not self.opt.no_rnn)

        assert -1 <= self.opt.freeze_earlier_blocks and self.opt.freeze_earlier_blocks <= 4
        if self.opt.freeze_earlier_blocks != -1:
            b0, b1, b2, b3, b4 = self.net.feature_extractor.list_blocks()
            blocks = [b0, b1, b2, b3, b4]
            for i in range(self.opt.freeze_earlier_blocks + 1):
                print('Freeze block%d' % i)
                for m in blocks[i]:
                    for param in m.parameters():
                        param.requires_grad = False

        if self.opt.bn_momentum:
            for m in self.net.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.momentum = self.opt.bn_momentum

    def forward(self, x, y_bon, y_cor):
        losses = {}

        y_bon_, y_cor_ = self.net(x)
        losses['bon'] = F.l1_loss(y_bon_, y_bon)
        losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor)
        losses['total'] = losses['bon'] + losses['cor']

        return losses

    def on_train_epoch_start(self) -> None:
        if self.opt.freeze_earlier_blocks != -1:
            b0, b1, b2, b3, b4 = self.net.feature_extractor.list_blocks()
            blocks = [b0, b1, b2, b3, b4]
            for i in range(self.opt.freeze_earlier_blocks + 1):
                for m in blocks[i]:
                    m.eval()
        
    def training_step(self, batch, batch_idx):
        x, y_bon, y_cor = batch

        losses = self(x, y_bon, y_cor)

        loss = losses['total']
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_bon, y_cor, gt_cor_id = batch        
        
        losses = self(x, y_bon, y_cor)

        # True eval result instead of training objective
        true_eval = dict([
            (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
            for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
        ])
        try:
            dt_cor_id = inference(self.net, x, x.device, force_raw=True)[0]
            dt_cor_id[:, 0] *= 1024
            dt_cor_id[:, 1] *= 512
        except Exception as e:
            print(e)
            dt_cor_id = np.array([
                [k//2 * 1024, 256 - ((k%2)*2 - 1) * 120]
                for k in range(8)
            ])
        test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)
        losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
        losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
        losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
        losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']])

    def configure_optimizers(self):
        # Create optimizer
        if self.opt.optim == 'SGD':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.opt.lr, momentum=self.opt.beta1, weight_decay=self.opt.weight_decay)
        elif self.opt.optim == 'Adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=self.opt.weight_decay)
        else:
            raise NotImplementedError()

        return optimizer
