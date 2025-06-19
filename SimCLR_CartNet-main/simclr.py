import logging
import os
import sys
import wandb

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, augment_batch
from torch_geometric.data import Batch


torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        #self.writer = SummaryWriter()
        #logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        print("labels", labels)
        
        labels = labels.to(self.args.device)

        print("shape", features.shape)
        print("features", features)


        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        print(similarity_matrix)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        print(positives)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        print(negatives)

        return

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                    name=cfg.name, config=cfg)

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        #save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        #logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        #logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for iter, batch in tqdm(enumerate(train_loader), total=len(train_loader), ncols=50):

                batch_aug_1 = augment_batch(batch.clone())
                batch_aug_2 = augment_batch(batch.clone())

                batch_aug = Batch.from_data_list([batch_aug_1, batch_aug_2]).to("cuda:0")


                #images = torch.cat(images, dim=0)
                #images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    _, features = self.model(batch_aug)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    wandb.log({
                        'loss': loss.item(),
                        'acc/top1': top1[0].item(),
                        'acc/top5': top5[0].item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'epoch': epoch_counter,
                        'step': n_iter
                    }, step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            #logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        #logging.info("Training has finished.")
        # save model checkpoints
        # checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        # save_checkpoint({
        #     'epoch': self.args.epochs,
        #     'arch': self.args.arch,
        #     'state_dict': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        # }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        # logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        ckpt_dir = osp.join(cfg.run_dir,"ckpt/")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = osp.join(ckpt_dir, 'best.ckpt')
            
        torch.save(ckpt, ckpt_path)


        run.finish()



           