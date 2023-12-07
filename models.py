import torch.nn as nn
import torch.nn.init as init
import torch

class ConvLSTM(nn.Module):
    def __init__(self, input_dim_mfb, input_dim_f0,input_dim_w2v,dropout_rate=0.4):
        super(ConvLSTM, self).__init__()

         # Convolution for MFB features
        self.conv_mfb = nn.Sequential(
            nn.Conv1d(input_dim_mfb, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Convolution for F0 features
        self.conv_f0 = nn.Sequential(
            nn.Conv1d(input_dim_f0, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )


        # Convolution for F0 features
        self.conv_w2v = nn.Sequential(
            nn.Conv1d(input_dim_w2v, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # LSTM layer
        self.lstm = nn.LSTM(192, 64, batch_first=True)  # 128 for concatenated features

        # Fully Connected layers for classification
        self.fc_fluent = nn.Linear(64, 2)  # Binary classification: Fluent/Dysfluent
        self.fc_events = nn.Linear(64, 6)  # 6 dysfluency types

         # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x_mfb, x_f0, x_w2v):
        #print("x_mfb shape:", x_mfb.shape)
        #print("x_f0 shape:", x_f0.shape)
        x_mfb = self.conv_mfb(x_mfb)
        x_f0 = self.conv_f0(x_f0)
        x_w2v = self.conv_w2v(x_w2v)
        x_w2v = x_w2v.transpose(1, 2)
        x_mfb = x_mfb.transpose(1, 2)  # Now shape: (batch, 100, 94)
        x_f0 = x_f0.transpose(1, 2)
        #print("x_mfb after conv shape:", x_mfb.shape)
        #print("x_f0 after conv shape:", x_f0.shape)

        # Combine features and LSTM processing
        x_combined = torch.cat((x_mfb, x_f0,x_w2v), dim=2)
        #print("x_combined shape:", x_combined.shape)
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x_combined)
        #print("Output shape after LSTM:", h_n.shape)
        lstm_out = h_n.squeeze(0)

        # Classification
        out_fluent = self.fc_fluent(lstm_out)
        out_events = self.fc_events(lstm_out)

        return out_fluent, out_events

class LSTM_base(nn.Module):
    def __init__(self):
        super(LSTM_base, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(135, 64, batch_first=True)  # 128 for concatenated features

        # Fully Connected layers for classification
        self.fc_fluent = nn.Linear(64, 2)  # Binary classification: Fluent/Dysfluent
        self.fc_events = nn.Linear(64, 6)  # 6 dysfluency types

    def forward(self, x_mfb, x_f0, x_w2v):
        #print("x_mfb shape:", x_mfb.shape)
        #print("x_f0 shape:", x_f0.shape)
        #x_mfb = self.conv_mfb(x_mfb)
        #x_f0 = self.conv_f0(x_f0)
        x_mfb = x_mfb.transpose(1, 2)  # Now shape: (batch, 100, 94)
        x_f0 = x_f0.transpose(1, 2)
        x_w2v = x_w2v.transpose(1, 2)
        #print("x_mfb after conv shape:", x_mfb.shape)
        #print("x_f0 after conv shape:", x_f0.shape)

        # Combine features and LSTM processing
        x_combined = torch.cat((x_mfb, x_f0,x_w2v), dim=2)
        #print("x_combined shape:", x_combined.shape)
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x_combined)
        #print("Output shape after LSTM:", h_n.shape)
        lstm_out = h_n.squeeze(0)

        # Classification
        out_fluent = self.fc_fluent(lstm_out)
        out_events = self.fc_events(lstm_out)

        return out_fluent, out_events