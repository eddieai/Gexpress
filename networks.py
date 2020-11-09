import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


# class EmbeddingNet(nn.Module):
#     def __init__(self):
#         super(EmbeddingNet, self).__init__()
#         self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2),
#                                      nn.Conv2d(32, 64, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2))
#
#         self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 2)
#                                 )
#
#     def forward(self, x):
#         output = self.convnet(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         return output
#
#     def get_embedding(self, x):
#         return self.forward(x)


# class EmbeddingNetL2(EmbeddingNet):
#     def __init__(self):
#         super(EmbeddingNetL2, self).__init__()
#
#     def forward(self, x):
#         output = super(EmbeddingNetL2, self).forward(x)
#         output /= output.pow(2).sum(1, keepdim=True).sqrt()
#         return output
#
#     def get_embedding(self, x):
#         return self.forward(x)


class EmbeddingNet(torch.nn.Module):
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
    """

    def __init__(self, n_channels, n_frames, dropout_probability=0.2):

        super(EmbeddingNet, self).__init__()

        self.n_channels = n_channels
        self.n_frames = n_frames
        self.dropout_probability = dropout_probability

        # Layers ----------------------------------------------

        self.timewisetrans1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.n_channels, out_features=64, bias=False),  # 128    64
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.LeakyReLU()
        )

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
        ) for _ in range(n_channels)])

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
        ) for _ in range(n_channels)])

        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2)
        ) for _ in range(n_channels)])

        self.timewisetrans2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=9 * 64, out_features=64, bias=False),  # 128    256
            torch.nn.ReLU()
        )

        # self.bdrnn = torch.nn.GRU(input_size=128, hidden_size=128, bidirectional=True)

        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(in_features=9 * n_channels * (self.n_frames//8), out_features=384),
            torch.nn.Linear(in_features=64 * (self.n_frames//8), out_features=384),
            # <-- 12: depends of the sequences lengths (cf. below)
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=384, out_features=128),
            torch.nn.Tanh()
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

        for layer in self.timewisetrans1:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        for layer in self.timewisetrans2:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        # for module in itertools.chain(self.bdrnn):
        #     for layer in module:
        #         if layer.__class__.__name__ == "BDRNN":
        #             torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        #             torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.

        Arguments
        ---------
            input: a tensor of gestures of shape (batch_size, duration, n_channels)
                   (where n_channels = 3 * n_joints for 3D pose data)
        """

        batch_size = input.shape[0]
        input = input.view(-1, self.n_channels)
        input = self.timewisetrans1(input)
        input = input.view(batch_size, self.n_frames, -1)

        # Work on each channel separately
        all_features = []

        for channel in range(0, input.shape[2]):
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

        n_features = all_features.shape[1]
        all_features = all_features.permute(0,2,1)
        all_features = all_features.reshape(-1, n_features)
        all_features = self.timewisetrans2(all_features)
        all_features = all_features.view(batch_size, -1)

        # Fully-Connected Layers
        output = self.fc(all_features)

        return output


# class ClassificationNet(nn.Module):
#     def __init__(self, embedding_net, n_classes):
#         super(ClassificationNet, self).__init__()
#         self.embedding_net = embedding_net
#         self.n_classes = n_classes
#         self.nonlinear = nn.PReLU()
#         self.fc1 = nn.Linear(2, n_classes)
#
#     def forward(self, x):
#         output = self.embedding_net(x)
#         output = self.nonlinear(output)
#         scores = F.log_softmax(self.fc1(output), dim=-1)
#         return scores
#
#     def get_embedding(self, x):
#         return self.nonlinear(self.embedding_net(x))

class ClassificationNet(nn.Module):
    def __init__(self, n_channels=66, n_frames=100, n_classes=14, dropout_probability=0.2):

        super(ClassificationNet, self).__init__()

        self.n_channels = n_channels
        self.n_frames = n_frames
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
        ) for _ in range(n_channels)])

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
        ) for _ in range(n_channels)])

        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2)
        ) for _ in range(n_channels)])

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=9 * n_channels * (self.n_frames//8), out_features=1936),
            # <-- 12: depends of the sequences lengths (cf. below)
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=self.dropout_probability),
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
                                         9 * self.n_channels * (self.n_frames//8))  # <-- 12: depends of the initial sequence length (100).
        # If you have shorter/longer sequences, you probably do NOT even need to modify the modify the network architecture:
        # resampling your input gesture from T timesteps to 100 timesteps will (surprisingly) probably actually work as well!

        # Fully-Connected Layers
        output = self.fc(all_features)

        return output


class SIGNNet(nn.Module):
    def __init__(self, n_channels=252, n_classes=11, dropout_probability=0.2):

        super(SIGNNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.n_channels, out_features=512),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.Linear(in_features=64, out_features=n_classes),
        )

        # Initialization --------------------------------------
        # Xavier init
        for layer in self.fc:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        output = self.fc(input)
        return output


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
