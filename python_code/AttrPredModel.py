from torchvision import models
import torch
import pytorch_lightning as pl
from sklearn import metrics
from argparse import ArgumentParser


class AttrPred_Resnet50(pl.LightningModule):
    def __init__(self, n_attributes, prediction_threshold=.0, **kwargs):
        super(AttrPred_Resnet50, self).__init__()
        self.predictor = models.resnet50(pretrained=True)
        self.predictor.fc = torch.nn.Linear(in_features=2048, out_features=n_attributes)
        self.prediction_threshold = prediction_threshold
        self.save_hyperparameters()
        self.checkpoint = pl.callbacks.ModelCheckpoint(filename='{epoch:02d}-{avg_f1_score:.3}.ckpt',
                                                       monitor="avg_f1_score", save_top_k=-1, mode="max")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prediction_threshold", type=float, default=0., help="Threshold to "
                                                                                   "trigger attribute prediction")
        return parser

    def forward(self, x, *args, **kwargs):
        return self.predictor(x)

    def training_step(self, batch, batch_id, *args, **kwargs):
        x, y = batch
        scores = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, y, reduction="mean")
        self.log("train_loss", loss, on_step=True)
        self.log("avg_train_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_id, *args, **kwargs):
        x, y = batch
        scores = self(x)
        y_hat = (scores > self.prediction_threshold).float()
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction="mean")
        f1_score = torch.tensor(metrics.f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro"))
        self.log("avg_f1_score", f1_score, prog_bar=True, on_epoch=True)
        self.log("avg_val_loss", val_loss, prog_bar=True, on_epoch=True)


if __name__ == "__main__":
    model = AttrPred_Resnet50(228, prediction_threshold=.0)
    trainer = pl.Trainer()
