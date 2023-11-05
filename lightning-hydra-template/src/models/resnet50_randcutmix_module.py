from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
import random
import torchvision
class Resnet50_randcutmixModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=1000)
        self.val_acc = Accuracy(task="multiclass", num_classes=1000)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.beta = 1.0
        self.cutmix_prob = 0.5
        
        self.transform_cutmix = ['', 'cutmix_imgs[i] = torchvision.transforms.functional.autocontrast(cutmix_imgs[i])','cutmix_imgs[i] = torchvision.transforms.functional.invert(cutmix_imgs[i])',
                    'cutmix_imgs[i] = torchvision.transforms.functional.adjust_brightness(cutmix_imgs[i],2)','cutmix_imgs[i] = torchvision.transforms.functional.adjust_sharpness(cutmix_imgs[i],2)',
                    'cutmix_imgs[i] = torchvision.transforms.RandomRotation(180)(cutmix_imgs[i])',
                    'cutmix_imgs[i] = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(cutmix_imgs[i])',
                    'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,(0.2,0))(cutmix_imgs[i])', 'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,(0,0.2))(cutmix_imgs[i])',
                    'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,shear=(-20,20,0,0))(cutmix_imgs[i])', 'cutmix_imgs[i] = torchvision.transforms.RandomAffine(0,shear=(-0,0,-20,20))(cutmix_imgs[i])']
        
        self.transform_original = ['', 'inputs[i] = torchvision.transforms.functional.autocontrast(inputs[i])','inputs[i] = torchvision.transforms.functional.invert(inputs[i])',
                    'inputs[i] = torchvision.transforms.functional.adjust_brightness(inputs[i],2)','inputs[i] = torchvision.transforms.functional.adjust_sharpness(inputs[i],2)',
                    'inputs[i] = torchvision.transforms.RandomRotation(180)(inputs[i])',
                    'inputs[i] = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(inputs[i])',
                    'inputs[i] = torchvision.transforms.RandomAffine(0,(0.2,0))(inputs[i])', 'inputs[i] = torchvision.transforms.RandomAffine(0,(0,0.2))(inputs[i])',
                    'inputs[i] = torchvision.transforms.RandomAffine(0,shear=(-20,20,0,0))(inputs[i])', 'inputs[i] = torchvision.transforms.RandomAffine(0,shear=(-0,0,-20,20))(inputs[i])']
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int_(W * cut_rat)
        cut_h = np.int_(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2        
        
    def randcutmix(self, inputs, target):
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(inputs.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
        
        # randcutmix
        cutmix_imgs = inputs[rand_index, :, :, :]
        for i in range(inputs.size()[0]):
            chocie_cutmix = random.randrange(0,len(self.transform_cutmix))
            choice_original = random.randrange(0,len(self.transform_original))
            exec(self.transform_cutmix[chocie_cutmix])
            exec(self.transform_original[choice_original])
            
        inputs[:, :, bbx1:bbx2, bby1:bby2] = cutmix_imgs[:, :, bbx1:bbx2, bby1:bby2]
        
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        
        return inputs, target_a, target_b, lam
          
    def randcutmix_model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        x, target_a, target_b, lam = self.randcutmix(x, y)
        logits = self.forward(x)
        loss = self.criterion(logits, target_a) * lam + self.criterion(logits, target_b) * (1. - lam)
        return loss

    def origin_model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        r = np.random.rand(1)
        if r < self.cutmix_prob:
            loss = self.randcutmix_model_step(batch)
        else:
            loss, preds, targets = self.origin_model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("Learning rate", current_lr, on_step=True, on_epoch=False)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.origin_model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/acc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
