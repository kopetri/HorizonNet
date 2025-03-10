import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model import HorizonNet
from inference import inference
from eval_general import test_general
from eval_cuboid import test
import numpy as np
from misc.utils import save_model
from pathlib import Path

class HorizonModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.net = HorizonNet(self.opt.backbone, not self.opt.no_rnn, self.opt.use_ring_conv if "use_ring_conv" in self.opt else False)

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

    def store_ckpt(self, path):
        path = Path(path).parent/"{}.pth".format(Path(path).stem)
        save_model(self.net, path.as_posix(), self.opt)

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
        self.log("train_bon_loss", losses['bon'])
        self.log("train_cor_loss", losses['cor'])
        self.log("train_loss", losses['total'])
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
            dt_cor_id = np.array([
                [k//2 * 1024, 256 - ((k%2)*2 - 1) * 120]
                for k in range(8)
            ])
        test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)
        losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
        losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
        losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
        losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']])

        self.log("valid_2DIoU",   losses['2DIoU'], prog_bar=True)
        self.log("valid_3DIoU",   losses['3DIoU'], prog_bar=True)
        self.log("valid_rmse",    losses['rmse'], prog_bar=True)
        self.log("valid_delta_1", losses['delta_1'], prog_bar=True)

        return {'valid_2DIoU': losses['2DIoU'], 'valid_3DIoU': losses['3DIoU'], 'valid_rmse': losses['rmse'], 'valid_delta_1': losses['delta_1']}

    def test_step(self, batch, batch_id):
        x, _, _, gt_cor_id = batch
        dt_cor_id, z0, z1, vis_out = inference(net=self.net, x=x, device=self.device, force_cuboid=True)

        dt_cor_id[:, 0] *= 1024
        dt_cor_id[:, 1] *= 512

        losses = {
            'CE': [],
            'PE': [],
            '3DIoU': [],
        }

        test(dt_cor_id, z0, z1, gt_cor_id.cpu().squeeze(0).numpy(), 1024, 512, losses)

        self.log('cuboid_CE',    losses['CE'][0])
        self.log('cuboid_PE',    losses['PE'][0])
        self.log('cuboid_3DIoU', losses['3DIoU'][0])

        dt_cor_id = inference(self.net, x, x.device, force_raw=True)[0]
        dt_cor_id[:, 0] *= 1024
        dt_cor_id[:, 1] *= 512

        # True eval result instead of training objective
        true_eval = dict([
            (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
            for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
        ])

        test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)

        losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
        losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
        losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
        losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']])

        self.log("general_2DIoU",   losses['2DIoU'],   prog_bar=True)
        self.log("general_3DIoU",   losses['3DIoU'],   prog_bar=True)
        self.log("general_rmse",    losses['rmse'],    prog_bar=True)
        self.log("general_delta_1", losses['delta_1'], prog_bar=True)


    def configure_optimizers(self):
        def adjust_learning_rate(cur_iter):
            if cur_iter < self.opt.warmup_iters:
                frac = cur_iter / self.opt.warmup_iters
                #step = self.opt.lr - self.opt.warmup_lr
                #running_lr = self.opt.warmup_lr + step * frac
                return frac - (1/self.opt.lr) * self.opt.warmup_lr * (frac + 1)
            else:
                frac = (float(cur_iter) - self.opt.warmup_iters) / (self.opt.max_iters - self.opt.warmup_iters)
                scale_running_lr = max((1. - frac), 0.) ** self.opt.lr_pow
                #running_lr = self.opt.lr * scale_running_lr
                return scale_running_lr
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

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, adjust_learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }