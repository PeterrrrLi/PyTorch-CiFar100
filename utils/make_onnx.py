import argparse
import os
import sys
import onnx
from onnx import shape_inference
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from load_model import load_model


def main():
    parser = argparse.ArgumentParser("A Program that Receives a Checkpoint File and Export an ONNX file")
    parser.add_argument("--net", required=True, type=str, help="Type of network")
    parser.add_argument("--check_point", required=True, type=str, help="Path to checkpoint file")
    parser.add_argument("--output_path", required=True, type=str, help="Output path for the onnx file")
    args = parser.parse_args()

    net = load_model(args.net)
    state_dict = torch.load(args.check_point)
    net.load_state_dict(state_dict)
    net.eval()
    batch_size = 128
    input_shape = (3, 32, 32)
    x = torch.randn(batch_size, *input_shape)
    torch.onnx.export(net,
                      x,
                      args.output_path,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    onnx.save(shape_inference.infer_shapes(onnx.load(args.output_path)), args.output_path)


if __name__ == "__main__":
    main()
