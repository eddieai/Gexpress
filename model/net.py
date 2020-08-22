import itertools
import numpy as np
import torch


class HandGestureNet(torch.nn.Module):
    """
    [Devineau et al., 2018] Deep Learning for Hand Gesture Recognition on Skeletal Data

    Summary
    -------
        Deep Learning Model for Hand Gesture classification using pose data only (no need for RGBD)
        The model computes a succession of [convolutions and pooling] over time independently on each of the 66 (= 22 * 3) sequence channels.
        Each of these computations are actually done at two different resolutions, that are later merged by concatenation
        with the (pooled) original sequence channel.
        Finally, a multi-layer perceptron merges all of the processed channels and outputs a classification.

    TL;DR:
    ------
        input ------------------------------------------------> split into n_channels channels [channel_i]
            channel_i ----------------------------------------> 3x [conv/pool/dropout] low_resolution_i
            channel_i ----------------------------------------> 3x [conv/pool/dropout] high_resolution_i
            channel_i ----------------------------------------> pooled_i
            low_resolution_i, high_resolution_i, pooled_i ----> output_channel_i
        MLP(n_channels x [output_channel_i]) -------------------------> classification

    Article / PDF:
    --------------
        https://ieeexplore.ieee.org/document/8373818

    Please cite:
    ------------
        @inproceedings{devineau2018deep,
            title={Deep learning for hand gesture recognition on skeletal data},
            author={Devineau, Guillaume and Moutarde, Fabien and Xi, Wang and Yang, Jie},
            booktitle={2018 13th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2018)},
            pages={106--113},
            year={2018},
            organization={IEEE}
        }
    """

    def __init__(self, n_channels=66, n_classes=14, dropout_probability=0.2):

        super(HandGestureNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability

        # Layers ----------------------------------------------
        self.all_conv_high = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.all_conv_low = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=9 * n_channels * 12, out_features=1936),
            # <-- 12: depends of the sequences lengths (cf. below)
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1936, out_features=n_classes)
        )

        # Initialization --------------------------------------
        # Xavier init
        for module in itertools.chain(self.all_conv_high, self.all_conv_low, self.all_residual):
            for layer in module:
                if layer.__class__.__name__ == "Conv1d":
                    torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                    torch.nn.init.constant_(layer.bias, 0.1)

        for layer in self.fc:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.

        Arguments
        ---------
            input: a tensor of gestures of shape (batch_size, duration, n_channels)
                   (where n_channels = 3 * n_joints for 3D pose data)
        """

        # Work on each channel separately
        all_features = []

        for channel in range(0, self.n_channels):
            input_channel = input[:, :, channel]

            # Add a dummy (spatial) dimension for the time convolutions
            # Conv1D format : (batch_size, n_feature_maps, duration)
            input_channel = input_channel.unsqueeze(1)

            high = self.all_conv_high[channel](input_channel)
            low = self.all_conv_low[channel](input_channel)
            ap_residual = self.all_residual[channel](input_channel)

            # Time convolutions are concatenated along the feature maps axis
            output_channel = torch.cat([
                high,
                low,
                ap_residual
            ], dim=1)
            all_features.append(output_channel)

        # Concatenate along the feature maps axis
        all_features = torch.cat(all_features, dim=1)

        # Flatten for the Linear layers
        all_features = all_features.view(-1,
                                         9 * self.n_channels * 12)  # <-- 12: depends of the initial sequence length (100).
        # If you have shorter/longer sequences, you probably do NOT even need to modify the modify the network architecture:
        # resampling your input gesture from T timesteps to 100 timesteps will (surprisingly) probably actually work as well!

        # Fully-Connected Layers
        output = self.fc(all_features)

        return output