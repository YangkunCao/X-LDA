# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class MetaPaths(nn.Module):
    def __init__(self, outchannel=16, dilation=1, padding=0):
        super(MetaPaths, self).__init__()
        self.conv_three_node_links = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=outchannel,
                kernel_size=(2, 1),
                stride=(1, 1),
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),

        )

    def forward(self, x):
        x = self.conv_three_node_links(x)
        return x


class ModulePatches(nn.Module):
    def __init__(self, outchannel=16, padding=1, dilation=(1, 2), stride=(1, 1)):
        super(ModulePatches, self).__init__()
        self.conv_four_node_links = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=outchannel,
                kernel_size=(2, 2),
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),

        )

    def forward(self, x):
        x = self.conv_four_node_links(x)
        return x


class X_LDA(nn.Module):

    def __init__(self, multi_nums=16):
        super(X_LDA, self).__init__()

        self.three_node_links_ls_adjT = MetaPaths(outchannel=multi_nums)
        self.three_node_links_adj_ds = MetaPaths(outchannel=multi_nums)
        self.three_node_links_lm_dm = MetaPaths(outchannel=multi_nums)

        self.four_interval0_node_links_ls_adjT = ModulePatches(outchannel=multi_nums, padding=0, dilation=1)
        self.four_interval0_node_links_adj_ds = ModulePatches(outchannel=multi_nums, padding=0, dilation=1)
        self.four_interval0_node_links_lm_dm = ModulePatches(outchannel=multi_nums, padding=0, dilation=1)

        self.four_interval1_node_links_ls_adjT = ModulePatches(outchannel=multi_nums, padding=0, dilation=(1, 2))
        self.four_interval1_node_links_adj_ds = ModulePatches(outchannel=multi_nums, padding=0, dilation=(1, 2))
        self.four_interval1_node_links_lm_dm = ModulePatches(outchannel=multi_nums, padding=0, dilation=(1, 2))

        self.four_interval1_node_links_LLD_LDD = ModulePatches(outchannel=multi_nums, padding=0, dilation=1,
                                                               stride=(1, 2))
        self.four_interval1_node_links_LMD_LLD = ModulePatches(outchannel=multi_nums, padding=0, dilation=1,
                                                               stride=(1, 2))
        self.four_interval1_node_links_LDD_LMD = ModulePatches(outchannel=multi_nums, padding=0, dilation=1,
                                                               stride=(1, 2))
        self.four_interval1_node_links_LMD_LDD = ModulePatches(outchannel=multi_nums, padding=0, dilation=1,
                                                               stride=(1, 2))

        self.conv_high_cat = nn.Sequential(
            nn.Conv2d(multi_nums, multi_nums, (2, 2), (1, 1), 1),
            nn.BatchNorm2d(multi_nums),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),
        )

        self.out = nn.Sequential(nn.Linear(291840, 2), nn.BatchNorm1d(2))

    def forward(self, x):
        ls_A_two_link = x[:, :, :, :240]
        A_ds_two_link = x[:, :, :, 240:645]
        lm_dm_two_link = x[:, :, :, 645:1140]

        ls_A_interval0_four_link = x[:, :, :, 1140:1380]
        A_ds_interval0_four_link = x[:, :, :, 1380:1785]
        lm_dm_interval0_four_link = x[:, :, :, 1785:2280]

        ls_A_interval1_four_link = x[:, :, :, 2280:2520]
        A_ds_interval1_four_link = x[:, :, :, 2520:2925]
        lm_dm_interval1_four_link = x[:, :, :, 2925:3420]

        LLD_LDD_interval1_four_link = x[:, :, :, 3420:3900]
        LDD_LMD_interval1_four_link_1 = x[:, :, :, 3900:4230]
        LMD_LLD_interval1_four_link = x[:, :, :, 4230:4710]

        LMD_LDD_interval1_four_link = x[:, :, :, 4710:4890]
        LDD_LMD_interval1_four_link_2 = x[:, :, :, 4890:5520]

        ls_A_three_node = self.three_node_links_ls_adjT(ls_A_two_link)
        A_ds_three_node = self.three_node_links_adj_ds(A_ds_two_link)
        lm_dm_three_node = self.three_node_links_lm_dm(lm_dm_two_link)

        ls_A_four_interval0 = self.four_interval0_node_links_ls_adjT(ls_A_interval0_four_link)
        A_ds_four_interval0 = self.four_interval0_node_links_adj_ds(A_ds_interval0_four_link)
        lm_dm_four_interval0 = self.four_interval0_node_links_lm_dm(lm_dm_interval0_four_link)

        LLD_LDD_four_interval1 = self.four_interval1_node_links_LLD_LDD(LLD_LDD_interval1_four_link)
        LMD_LLD_four_interval1 = self.four_interval1_node_links_LMD_LLD(LMD_LLD_interval1_four_link)

        LDD_LMD_four_interval1_1 = self.four_interval1_node_links_LDD_LMD(LDD_LMD_interval1_four_link_1)
        LDD_LMD_four_interval1_2 = self.four_interval1_node_links_LDD_LMD(LDD_LMD_interval1_four_link_2)
        LMD_LDD_four_interval1 = self.four_interval1_node_links_LMD_LDD(LMD_LDD_interval1_four_link)

        ls_A_four_interval1 = self.four_interval1_node_links_ls_adjT(ls_A_interval1_four_link)
        A_ds_four_interval1 = self.four_interval1_node_links_adj_ds(A_ds_interval1_four_link)
        lm_dm_four_interval1 = self.four_interval1_node_links_lm_dm(lm_dm_interval1_four_link)

        pad_one_zero_right = nn.ConstantPad2d((0, 1, 0, 0), 0.)
        ls_A_four_interval0 = pad_one_zero_right(ls_A_four_interval0)
        A_ds_four_interval0 = pad_one_zero_right(A_ds_four_interval0)
        lm_dm_four_interval0 = pad_one_zero_right(lm_dm_four_interval0)

        pad_two_zero_right = nn.ConstantPad2d((0, 2, 0, 0), 0.)
        ls_A_four_interval1 = pad_two_zero_right(ls_A_four_interval1)
        A_ds_four_interval1 = pad_two_zero_right(A_ds_four_interval1)
        lm_dm_four_interval1 = pad_two_zero_right(lm_dm_four_interval1)

        out_three_node = torch.cat((ls_A_three_node, A_ds_three_node, lm_dm_three_node), 3)
        out_four_interval0 = torch.cat((ls_A_four_interval0, A_ds_four_interval0, lm_dm_four_interval0), 3)
        out_four_interval1_1 = torch.cat((ls_A_four_interval1, A_ds_four_interval1, lm_dm_four_interval1), 3)
        out_four_interval1_2 = torch.cat((LLD_LDD_four_interval1, LMD_LLD_four_interval1,
                                          LDD_LMD_four_interval1_1, LDD_LMD_four_interval1_2, LMD_LDD_four_interval1),
                                         3)
        pad_more_zero_right = nn.ConstantPad2d((0, 1140 - 1052 + 2, 0, 0), 0.)

        out_four_interval1_2 = pad_more_zero_right(out_four_interval1_2)
        out = torch.cat((out_three_node, out_four_interval0, out_four_interval1_1, out_four_interval1_2), 2)
        out = self.conv_high_cat(out)

        out = out.view(out.size()[0], -1)
        out = self.out(out)

        return out
