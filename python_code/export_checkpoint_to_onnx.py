from AttrPredModel import AttrPred_Resnet50
import argparse
import torch
from torchsummary import summary
import sys


def export2sas(argstring=None):
    parser = argparse.ArgumentParser("Extracting a trained Attribute Prediction Model to ONNX format")
    parser.add_argument("-c", type=str, help="path to checkpoint file", required=True)
    parser.add_argument("-o", type=str, help="path to output file", required=True)

    if argstring:
        args = parser.parse_args(argstring)
    else:
        args = parser.parse_args()

    checkpoint_path = args.c
    output_path = args.o

    model = AttrPred_Resnet50.load_from_checkpoint(checkpoint_path)
    model.cpu()
    # print(summary(model, input_size=(3, 512, 512), device="cpu"))

    input_sample = torch.rand((1, 3, 512, 512))
    model.to_onnx(output_path, input_sample, export_params=True, input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: 'batch_size'}, "output": {0: "batch_size"}})

    print(f"Exported Model successfully to {output_path}")


if __name__ == "__main__":
    export2sas()

