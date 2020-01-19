import argparse
import importlib

from torchsummary import summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--network_name", type=str)
    parser.add_argument("--network_module", type=str)
    args = parser.parse_args()
    model = getattr(importlib.import_module(args.network_module),
                    args.network_name)()

    summary(model, input_size=(3, 512, 512))
