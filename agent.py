import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
import utils.sd as sd
from utils.networks import get_network
from utils.fisher import MatrixFisherN
from flow.flow import Flow
import pytorch_lightning as pl

class Agent(pl.LightningModule):
    def __init__(self, config, device):
        super(Agent, self).__init__()
        self.config = config
        # self.device = device
        self.flow = Flow(config)
        self.net = get_network(config, self.device) if self.config.condition else None
        if self.config.embedding and (not self.config.pretrain_fisher):
            self.embedding = nn.Parameter(torch.randn(
                (self.config.category_num, self.config.embedding_dim), device=self.device))

    def forward(self, data):
        gt = data.get("rot_mat")  # (b, 3, 3)
        feature, A = self.compute_feature(data, self.config.condition)

        rotation, ldjs = self.flow(gt, feature)
        return rotation, ldjs, feature, A

    def compute_loss(self, rotation, ldjs, feature, A):
        losses_nll = -ldjs

        if self.config.pretrain_fisher:
            A = A.clone()
            pre_distribution = MatrixFisherN(A)  # (N, 3, 3)
            pre_ll = pre_distribution._log_prob(rotation)  # (N, )
            losses_pre_nll = -pre_ll  # probability = pre_ll + ldjs
        else:
            losses_pre_nll = 0 * losses_nll

        loss = losses_nll.mean() + losses_pre_nll.mean()

        result_dict = dict(
            loss=loss,
            losses_nll=losses_nll,
            losses_pre_nll=losses_pre_nll,
            feature=feature,
        )
        return result_dict

    def training_step(self, batch, batch_idx):
        rotation, ldjs, feature, A = self.forward(batch)
        result_dict = self.compute_loss(rotation, ldjs, feature, A)
        self.log('train_loss', result_dict['loss'])
        return result_dict

    def validation_step(self, batch, batch_idx):
        rotation, ldjs, feature, A = self.forward(batch)
        result_dict = self.compute_loss(rotation, ldjs, feature, A)
        self.log('val_loss', result_dict['loss'])
        return result_dict

    def configure_optimizers(self):
        if self.config.use_lr_decay:
            lr_decay = [int(item) for item in self.config.lr_decay.split(',')]
            optimizer_flow_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer_flow, milestones=lr_decay, gamma=self.config.gamma)

            if self.config.condition:
                optimizer_net_list = [*self.net.parameters()]
                if self.config.embedding and (not self.config.pretrain_fisher):
                    optimizer_net_list.append(self.embedding)
                self.optimizer_net = optim.Adam(optimizer_net_list, self.config.lr)
                optimizer_net_scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer_net, milestones=lr_decay, gamma=self.config.gamma)
                return [
                    {"optimizer": self.optimizer_flow, "lr_scheduler": optimizer_flow_scheduler, "monitor": "val_loss"},
                    {"optimizer": self.optimizer_net, "lr_scheduler": optimizer_net_scheduler, "monitor": "val_loss"}
                ]
            else:
                return [
                    {"optimizer": self.optimizer_flow, "lr_scheduler": optimizer_flow_scheduler, "monitor": "val_loss"}
                ]
        else:
            if self.config.condition:
                return [self.optimizer_flow, self.optimizer_net]
            else:
                return [self.optimizer_flow]

    def save_ckpt(self):
        checkpoint = {'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}
        save_path = os.path.join(self.config.model_dir, 'checkpoint.ckpt')
        torch.save(checkpoint, save_path)

    def load_ckpt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def eval_nll(self, flow, data, feature, A):
        # compute gt_nll
        if self.config.dataset == "symsol":
            gt = data.get('rot_mat_all').to(self.device)
            gt_amount = gt.size(1)
            feature_2 = (
                feature[:, None, :]
                .repeat(1, gt_amount, 1)
                .reshape(-1, self.config.feature_dim + (self.config.category_num != 1)*self.config.embedding*self.config.embedding_dim)
            )
            gt_2 = gt.reshape(-1, 3, 3)
            assert gt_2.size(0) == feature_2.size(0)
            _, gt_ldjs = flow(gt_2, feature_2)
            gt_ldjs = gt_ldjs.reshape(-1, gt_amount)
            losses_ll = gt_ldjs.mean(dim=-1)
        else:
            gt = data.get('rot_mat').to(self.device)
            rotation, gt_ldjs = flow(gt, feature)
            losses_ll = gt_ldjs

        if self.config.pretrain_fisher:
            pre_distribution = MatrixFisherN(A.clone())  # (N, 3, 3)
            pre_ll = pre_distribution._log_prob(rotation)  # (N, )
        else:
            pre_ll = 0 * losses_ll

        loss_dict = dict(
            loss=losses_ll.mean() + pre_ll.mean(),
            losses=pre_ll + losses_ll
        )

        if self.config.dataset == "symsol":
            acc_dict = self.eval_acc(flow, data, feature, A)
            loss_dict['est_rotation'] = acc_dict['est_rotation']
            loss_dict['err_deg'] = acc_dict['err_deg']

        return loss_dict

    def eval_acc(self, flow, data, feature, A):
        batch = feature.size(0)
        feature = (
            feature[:, None, :].repeat(1, self.config.number_queries, 1).
            reshape(-1, self.config.feature_dim +
                    self.config.embedding*self.config.embedding_dim)
        )

        if self.config.pretrain_fisher:
            pre_distribution = MatrixFisherN(A.clone())  # (N, 3, 3)
            sample = pre_distribution._sample(
                self.config.number_queries).reshape(-1, 3, 3)  # sample from pre distribution
            base_ll = pre_distribution._log_prob(
                sample).reshape(batch, -1)
        else:
            sample = sd.generate_queries(self.config.number_queries).to(self.device)
            sample = (
                sample[None, ...].repeat(batch, 1, 1, 1).reshape(-1, 3, 3)
            )
            base_ll = torch.zeros(
                (batch, self.config.number_queries), device=feature.device)
        assert feature.size(0) == sample.size(0)

        samples, log_prob = flow.inverse(sample, feature)
        samples = samples.reshape(batch, -1, 3, 3)
        log_prob = - log_prob.reshape(batch, -1) + base_ll
        max_inds = torch.argmax(log_prob, axis=-1)
        batch_inds = torch.arange(batch)
        est_rotation = samples[batch_inds, max_inds]
        if self.config.dataset == 'symsol':
            gt = data.get('rot_mat_all').to(self.device)
            err_rad = utils.min_geodesic_distance_rotmats(
                est_rotation, gt)
        else:
            gt = data.get('rot_mat').to(self.device)
            err_rad = utils.min_geodesic_distance_rotmats(
                gt, est_rotation.reshape(batch, -1, 3, 3))
        err_deg = torch.rad2deg(err_rad)
        result_dict = dict(
            est_rotation=est_rotation,
            err_deg=err_deg,
        )
        if self.config.dataset == 'pascal3d':
            easy = torch.nonzero(data.get('easy').to(self.device)).reshape(-1)
            result_dict = {k: v[easy] for k, v in result_dict.items()}
        return result_dict

    def train_agent(self):
        net_optimizers = []
        if self.config.condition:
            self.net.train()
            net_optimizers.append((self.net, self.optimizer_net))
        self.flow.train()
        net_optimizers.append((self.flow, self.optimizer_flow))
        return net_optimizers

    def eval(self):
        if self.config.condition:
            self.net.eval()
        self.flow.eval()

    def compute_feature(self, data, condition):
        if condition == 0:
            A = None
            feature = None
        else:
            data['cate'] = data['cate'].to(self.device)
            img = data.get("img").to(self.device)
            self.net = self.net.to(self.device)
            if self.config.pretrain_fisher:
                feature, A = net(img, data.get('cate').to(self.device)
                + (1 if self.config.dataset == 'pascal3d' else 0))
                A = A.reshape(-1, 3, 3)
            else:
                feature = self.net(img)  # (b, feature_dim)
                if self.config.category_num != 1 and self.config.embedding:
                    feature = torch.concat(
                        [feature, self.embedding[data.get('cate')]], dim=-1)
                A = None
        return feature, A
