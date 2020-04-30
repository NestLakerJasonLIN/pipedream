# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stage0 import Stage0
from .stage1 import Stage1
from .vgg16 import VGG16Partitioned


def arch():
    return "vgg16"


def model(criterion):
    return [
        (Stage0(),
         ["input0"],
            ["out0_b0",
             "out0_b1",
             "out0_b2",
             "out0_b3"]),
        # stage 0
        (Stage1(),
         ["out0_b0",
          "out0_b1",
          "out0_b2",
          "out0_b3"],
            ["out1"]),
        # stage 1
        (criterion, ["out1"], ["loss"])  # stage 1
    ]


def full_model():
    return VGG16Partitioned()
