from importlib import import_module
from SPConvNets import Dataloader_ModelNet40
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree

class Trainer(vgtk.Trainer):
    def __init__(self, opt):

        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'
        super(Trainer, self).__init__(opt)

        if self.attention_model:
            self.summary.register(['Loss', 'Acc', 'R_Loss', 'R_Acc'])
        else:
            self.summary.register(['Loss', 'Acc'])
        self.epoch_counter = 0
        self.iter_counter = 0
        self.test_accs = []

    def _setup_datasets(self):
        if self.opt.mode == 'train':
            dataset = Dataloader_ModelNet40(self.opt)
            self.dataset = torch.utils.data.DataLoader(dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)

        dataset_test = Dataloader_ModelNet40(self.opt, 'test')
        self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)


    def _setup_model(self):
        if self.opt.mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

    def _setup_metric(self):
        if self.attention_model:
            self.metric = vgtk.AttentionCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin)
        else:
            self.metric = vgtk.CrossEntropyLoss()

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset)):
            self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
            if data['label'].shape[0] < self.opt.batch_size:
                raise StopIteration
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)

        self._optimize(data)
        self.iter_counter += 1

    def _optimize(self, data):
        in_tensors = data['pc'].to(self.opt.device)
        bdim = in_tensors.shape[0]
        in_label = data['label'].to(self.opt.device).reshape(-1)
        in_Rlabel = data['R_label'].to(self.opt.device) if self.opt.debug_mode == 'knownatt' else None

        pred, feat = self.model(in_tensors, in_Rlabel)

        self.optimizer.zero_grad()

        if self.attention_model:
            in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
            self.loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
        else:
            cls_loss, acc = self.metric(pred, in_label)
            self.loss = cls_loss

        self.loss.backward()
        self.optimizer.step()

        # Log training stats
        if self.attention_model:
            log_info = {
                'Loss': cls_loss.item(),
                'Acc': 100 * acc.item(),
                'R_Loss': r_loss.item(),
                'R_Acc': 100 * r_acc.item(),
            }
        else:
            log_info = {
                'Loss': cls_loss.item(),
                'Acc': 100 * acc.item(),
            }

        self.summary.update(log_info)


    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
        self.eval()
        return None

    def eval(self):
        self.logger.log('Testing','Evaluating test set!')
        self.model.eval()
        self.metric.eval()

        with torch.no_grad():
            accs = []
            all_labels = []
            all_feats = []

            for it, data in enumerate(self.dataset_test):
                in_tensors = data['pc'].to(self.opt.device)
                bdim = in_tensors.shape[0]
                in_label = data['label'].to(self.opt.device).reshape(-1)
                in_Rlabel = data['R_label'].to(self.opt.device) if self.opt.debug_mode == 'knownatt' else None

                pred, feat = self.model(in_tensors, in_Rlabel)

                if self.attention_model:
                    in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
                    loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
                    attention = F.softmax(feat,1)

                    if self.opt.train_loss.attention_loss_type == 'no_cls':
                        acc = r_acc
                        loss = r_loss

                else:
                    cls_loss, acc = self.metric(pred, in_label)
                    loss = cls_loss

                all_labels.append(in_label.cpu().numpy())
                all_feats.append(feat.cpu().numpy())

                accs.append(acc)
                self.logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))
                if self.attention_model:
                    self.logger.log("Testing", "Rot Acc: %.1f, Rot Loss: %.2f!"%(100*r_acc.item(), r_loss.item()))

            accs = np.array(accs, dtype=np.float32)

            self.logger.log('Testing', 'Average accuracy is %.2f!!!!'%(100*accs.mean()))
            self.test_accs.append(100*accs.mean())
            best_acc = np.array(self.test_accs).max()
            self.logger.log('Testing', 'Best accuracy so far is %.2f!!!!'%(best_acc))

            # enable lines below for testing retrieval mAP
            # n = 1
            # mAP = modelnet_retrieval_mAP(all_feats,all_labels,n)
            # self.logger.log('Testing', 'Mean average precision at %d is %f!!!!'%(n, mAP))

        self.model.train()
        self.metric.train()
