from torchvision import models
import torch
import pytorch_lightning as pl
from sklearn import metrics
from argparse import ArgumentParser


class DebugNet(pl.LightningModule):
    def __init__(self, n_attributes, prediction_threshold=.0, input_dim=(512, 512), **kwargs):
        """
        A very small densely connected prediction network for local debugging of the training pipeline without having
        a gpu at hand. Not meant for actual inference!
        :param n_attributes: number of attributes to predict
        :param prediction_threshold: threshold score above which attribute is counted as predicted
        :param input_dim:
        :param kwargs: buffers hyperparameters for saving (see self.save_hyperparameters)
        """
        super(DebugNet, self).__init__()
        self.input_dim = input_dim
        self.predictor = torch.nn.Linear(3*input_dim[0]*input_dim[1], n_attributes)
        self.prediction_threshold = prediction_threshold
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prediction_threshold", type=float, default=0., help="Threshold to "
                                                                                    "trigger attribute prediction")
        return parser

    def forward(self, x, *args, **kwargs):
        return self.predictor(x.view(-1, self.input_dim[0]*self.input_dim[1]*3))

    def training_step(self, batch, batch_id, *args, **kwargs):
        x, y = batch
        scores = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, y, reduction="mean")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_id, *args, **kwargs):
        x, y = batch
        scores = self(x)
        y_hat = (scores > self.prediction_threshold).float()
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction="mean")
        f1_score = metrics.f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro")

        return {"val_loss": val_loss, "f1_score": f1_score}


if __name__ == "__main__":
    model = DebugNet(228, prediction_threshold=.0)
    trainer = pl.Trainer()
