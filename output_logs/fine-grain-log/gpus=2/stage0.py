# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from datetime import datetime

class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer7 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.upstream_tail = Upstream_Tail(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self._initialize_weights()

    def forward(self, input0, forward_minibatch_id, backward_minibatch_id, comm_handler):
        start_time = datetime.now()
    
        out0 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)

        dt = datetime.now() - start_time
        elapsed = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        print("Stage0 before last layer:", "%.20fms" % elapsed)

        start_time = datetime.now()
        
        out9 = self.upstream_tail(out8, forward_minibatch_id, backward_minibatch_id, comm_handler)
        
        dt = datetime.now() - start_time
        elapsed = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        print("Stage0 last layer:", "%.20fms" % elapsed)
        
        return out9

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

class Upstream_Tail(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Upstream_Tail, self).__init__()
        self.orig_padding = padding
        self.kernel_size = kernel_size[0]
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=kernel_size,
                                      stride=stride)

        self.padder = torch.nn.ZeroPad2d(padding[0])
    
    def forward(self, inp, forward_minibatch_id, backward_minibatch_id, comm_handler):
        
        block_out_list = []
                
        inp = self.padder(inp)
        h_pad, w_pad = inp.size(2), inp.size(3)
        block_height, block_width = h_pad // 2,  w_pad // 2
        
        # block_0
        start_time = datetime.now()

        h_start, h_end = 0, block_height + self.kernel_size-1
        w_start, w_end = 0, block_width + self.kernel_size-1

        block_inp = inp[:, :, h_start:h_end, w_start:w_end]
        
        block_out = self.conv2d(block_inp)
        block_out_list.append(block_out)
        
        comm_handler.send_block(block_out, forward_minibatch_id=forward_minibatch_id,
                                     backward_minibatch_id=backward_minibatch_id)

        dt = datetime.now() - start_time
        elapsed = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        print(" ->block 0 time:", "%.20fms" % elapsed)

        # block_1
        start_time = datetime.now()

        h_start, h_end = 0, block_height + self.kernel_size-1
        w_start, w_end = block_width, w_pad

        block_inp = inp[:, :, h_start:h_end, w_start:w_end]

        block_out = self.conv2d(block_inp)
        block_out_list.append(block_out)

        comm_handler.send_block(block_out, forward_minibatch_id=forward_minibatch_id,
                                     backward_minibatch_id=backward_minibatch_id)

        dt = datetime.now() - start_time
        elapsed = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        print(" ->block 1 time:", "%.20fms" % elapsed)

        # block_2
        start_time = datetime.now()

        h_start, h_end = block_height, h_pad
        w_start, w_end = 0, block_width + self.kernel_size-1

        block_inp = inp[:, :, h_start:h_end, w_start:w_end]

        block_out = self.conv2d(block_inp)
        block_out_list.append(block_out)

        comm_handler.send_block(block_out, forward_minibatch_id=forward_minibatch_id,
                             backward_minibatch_id=backward_minibatch_id)
        
        dt = datetime.now() - start_time
        elapsed = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        print(" ->block 2 time:", "%.20fms" % elapsed)

        # block_3
        start_time = datetime.now()

        h_start, h_end = block_height, h_pad
        w_start, w_end = block_width, w_pad

        block_inp = inp[:, :, h_start:h_end, w_start:w_end]

        block_out = self.conv2d(block_inp)
        block_out_list.append(block_out)

        comm_handler.send_block(block_out, forward_minibatch_id=forward_minibatch_id,
                                     backward_minibatch_id=backward_minibatch_id)

        dt = datetime.now() - start_time
        elapsed = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        print(" ->block 3 time:", "%.20fms" % elapsed)

        return self._combine(block_out_list)
    
    def _combine(self, block_list):
        block_upper = torch.cat((block_list[0], block_list[1]), dim=3)
        block_lower = torch.cat((block_list[2], block_list[3]), dim=3)
        combined_inp = torch.cat((block_upper, block_lower), dim=2)

        return combined_inp  