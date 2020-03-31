import argparse
import importlib

import torch
from torch.autograd import Variable
from torchsummary import summary
from torchviz import make_dot, make_dot_from_trace



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--network_name", type=str)
    parser.add_argument("--network_module", type=str)
    args = parser.parse_args()
    model = getattr(importlib.import_module(args.network_module),
                    args.network_name)()

    summary(model, input_size=(3, 512, 512))

    model.eval()
    x = torch.randn(1, 3, 512, 512)
    grp = make_dot(model(x), params=dict(model.named_parameters()))
    grp.view()
