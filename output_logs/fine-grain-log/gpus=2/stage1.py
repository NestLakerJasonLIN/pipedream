# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import sys
sys.path.append("/home/ubuntu/pipedream/runtime")
from runtime_utilities import t_start, t_stop

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.downstream_head = Downstream_Head(inplace=True)
        self.layer2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer3 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer10 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer17 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer18 = torch.nn.ReLU(inplace=True)
        self.layer19 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer20 = torch.nn.ReLU(inplace=True)
        self.layer21 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer22 = torch.nn.ReLU(inplace=True)
        self.layer23 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer26 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer27 = torch.nn.ReLU(inplace=True)
        self.layer28 = torch.nn.Dropout(p=0.5)
        self.layer29 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer30 = torch.nn.ReLU(inplace=True)
        self.layer31 = torch.nn.Dropout(p=0.5)
        self.layer32 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

        self._initialize_weights()

    def forward(self, forward_minibatch_id, backward_minibatch_id, r):
        start_time = t_start()
        
        out1 = self.downstream_head(forward_minibatch_id, backward_minibatch_id, r)
        
        t_stop(start_time, " -> Stage1 1st layer:")

        start_time = t_start()
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

        t_stop(start_time, " -> Stage1 other layers:")

        return out32

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

        block_num = 4
        block_out_relu = []
        
        for block_id in range(block_num):
            start_time = t_start()
            block_inp_relu = r.comm_handler.recv_block(forward_minibatch_id, backward_minibatch_id)
            t_stop(start_time, "  -> bid: {} recv elapsed:".format(block_id))

            # store block_inp_relu into buffer
            # slice and clone buffer and pass into ReLU
            # return buffer as input_tensor

            start_time = t_start()
            if (block_id == 0):
                # infer shape from the first recv block
                batch_size, channel_size = block_inp_relu.size(0), block_inp_relu.size(1)
                block_buffer = torch.cuda.FloatTensor(batch_size, channel_size, 112, 112).fill_(0)

                block_buffer[:, :, :57, :57] = block_inp_relu
                block_out_relu.append(self.relu(block_buffer[:, :, :57, :57].clone()))
            elif (block_id == 1):
                block_buffer[:, :, :57, 57:] = block_inp_relu
                block_out_relu.append(self.relu(block_buffer[:, :, :57, 57:].clone()))
            elif(block_id == 2):
                block_buffer[:, :, 57:, :57] = block_inp_relu
                block_out_relu.append(self.relu(block_buffer[:, :, 57:, :57].clone()))
            else:
                block_buffer[:, :, 57:, 57:] = block_inp_relu
                block_out_relu.append(self.relu(block_buffer[:, :, 57:, 57:].clone()))

            t_stop(start_time, "  -> bid: {} fill elapsed:".format(block_id))


        # Used to track where to receive forward from.
        r.comm_handler.increment_messaging_index(
            sending=False)
        
        start_time = t_start()
        relu_out = self._combine(block_out_relu)
        t_stop(start_time, " -> combine elapsed:".format(block_id))

        r.tensors[-1]["out0"] = block_buffer

        return relu_out

    def _combine(self, block_list):
        block_upper = torch.cat((block_list[0], block_list[1]), dim=3)
        block_lower = torch.cat((block_list[2], block_list[3]), dim=3)
        combined_inp = torch.cat((block_upper, block_lower), dim=2)

        return combined_inp  