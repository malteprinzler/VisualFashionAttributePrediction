from python_code.iMatDataset import iMatDataModule
from python_code.AttrPredModel import AttrPred_Resnet50
import pytorch_lightning as pl
from argparse import ArgumentParser
from torchvision.transforms import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--description", type=str, required=False, help="Description")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AttrPred_Resnet50.add_model_specific_args(parser)
    parser = iMatDataModule.add_dataset_specific_args(parser)

    args = parser.parse_args()

    image_augmentations = [ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), RandomHorizontalFlip()]
    dm = iMatDataModule(**vars(args))
    dm.prepare_data()
    dm.setup()

    model = AttrPred_Resnet50(228, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model.checkpoint])
    trainer.fit(model, train_dataloader=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
