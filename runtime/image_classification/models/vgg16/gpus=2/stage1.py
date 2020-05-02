# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import concurrent.futures
import time


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.downstream_head = Downstream_Head(inplace=True)
        self.layer1 = torch.nn.ReLU(inplace=True)
        self.layer2 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            ceil_mode=False)
        self.layer3 = torch.nn.Conv2d(
            128, 256, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(
            256, 256, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Conv2d(
            256, 256, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            ceil_mode=False)
        self.layer10 = torch.nn.Conv2d(
            256, 512, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.Conv2d(
            512, 512, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.Conv2d(
            512, 512, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            ceil_mode=False)
        self.layer17 = torch.nn.Conv2d(
            512, 512, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer18 = torch.nn.ReLU(inplace=True)
        self.layer19 = torch.nn.Conv2d(
            512, 512, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer20 = torch.nn.ReLU(inplace=True)
        self.layer21 = torch.nn.Conv2d(
            512, 512, kernel_size=(
                3, 3), stride=(
                1, 1), padding=(
                1, 1))
        self.layer22 = torch.nn.ReLU(inplace=True)
        self.layer23 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            ceil_mode=False)
        self.layer26 = torch.nn.Linear(
            in_features=25088, out_features=4096, bias=True)
        self.layer27 = torch.nn.ReLU(inplace=True)
        self.layer28 = torch.nn.Dropout(p=0.5)
        self.layer29 = torch.nn.Linear(
            in_features=4096, out_features=4096, bias=True)
        self.layer30 = torch.nn.ReLU(inplace=True)
        self.layer31 = torch.nn.Dropout(p=0.5)
        self.layer32 = torch.nn.Linear(
            in_features=4096, out_features=1000, bias=True)

        self._initialize_weights()


    def forward(self, input0, forward_minibatch_id=-1, backward_minibatch_id=-1, r=None):
        start_time = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        if r is None: 
            out0_b0 = input0[0].clone()
            out0_b1 = input0[1].clone()
            out0_b2 = input0[2].clone()
            out0_b3 = input0[3].clone()
            out1 = self.layer1(_combine(out0_b0, out0_b1, out0_b2, out0_b3))
        else: 
            out1 = self.downstream_head(
                forward_minibatch_id, backward_minibatch_id, r)

        elapsed = (
            time.clock_gettime(
                time.CLOCK_THREAD_CPUTIME_ID) - start_time) * 1000

        print(" -> Stage1 1st layer:", "%.20fms" % elapsed)

        start_time = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = out23.size(0)
        out25 = out23.view(out24, -1)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)

        elapsed = (
            time.clock_gettime(
                time.CLOCK_THREAD_CPUTIME_ID) - start_time) * 1000

        print(" -> Stage1 other layers:", "%.20fms" % elapsed)

        return out32

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


class Downstream_Head(torch.nn.Module):
    def __init__(self, inplace):
        super(Downstream_Head, self).__init__()
        self.relu = torch.nn.ReLU(inplace=inplace)

    def forward(self, forward_minibatch_id, backward_minibatch_id, r):

        print(" -> Stage1 Downstream_Head:")

        start_time = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            b0 = executor.submit(self.thread_function, "out0_b0",
                                 r)
            b1 = executor.submit(self.thread_function, "out0_b1",
                                 r)
            b2 = executor.submit(self.thread_function, "out0_b2",
                                 r)
            b3 = executor.submit(self.thread_function, "out0_b3",
                                 r)

            block0_out = b0.result()
            block1_out = b1.result()
            block2_out = b2.result()
            block3_out = b3.result()

        elapsed = (
            time.clock_gettime(
                time.CLOCK_THREAD_CPUTIME_ID) - start_time) * 1000

        print("  -> time elapsed:", "%.20fms" % elapsed)

        # Used to track where to receive forward from.
        r.comm_handler.increment_messaging_index(
            sending=False)

        start_time = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        relu_out = _combine(
            block0_out,
            block1_out,
            block2_out,
            block3_out)

        elapsed = (
            time.clock_gettime(
                time.CLOCK_THREAD_CPUTIME_ID) - start_time) * 1000

        print("  ->_combine elapsed:", "%.20fms" % elapsed)

        return relu_out

    def thread_function(
            self,
            tensor_name,
            r):

        block_in = r.comm_handler.recv_block(tensor_name)

        r.tensors[-1][tensor_name] = block_in

        block_out = self.relu(block_in.clone())
        return block_out

def _combine(block0_out, block1_out, block2_out, block3_out):
    block_upper = torch.cat((block0_out, block1_out), dim=3)
    block_lower = torch.cat((block2_out, block3_out), dim=3)
    combined_inp = torch.cat((block_upper, block_lower), dim=2)

    return combined_inp
