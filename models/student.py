import torch
import torch.nn as nn
import torch.nn.functional as F
from .mel import MelSpectrogram
from .constants import *
import random
import random


class DepthwiseSeparableConv(nn.Module):
    """
    Splits convolution into two parts:
    1. Spatial (Depthwise) - Looks at 3x3 patch per channel
    2. Pointwise (1x1) - Mixes channels together
    Drastically reduces parameters and FLOPs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CompactConvStack(nn.Module):
    def __init__(self, input_features, output_features, complexity=24):
        super().__init__()
        
        c = complexity * 16 # Base channel width
        
        # We replace the heavy ConvStack with a MobileNet-style stack
        self.cnn = nn.Sequential(
            # Layer 0: Keep standard conv for first layer (better feature extraction)
            nn.Conv2d(1, c, (3, 3), padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            
            # Layer 3: Depthwise Separable
            DepthwiseSeparableConv(c, c, (3, 3), padding=1),
            
            # Layer 4: Pooling
            nn.MaxPool2d((1, 2)),
            # Layer 5: Dropout
            nn.Dropout(0.2),
            # Layer 6: DSC
            DepthwiseSeparableConv(c, c * 2, (3, 3), padding=1),
            
            # Layer 7: Pooling
            nn.MaxPool2d((1, 2)),
            # Layer 8: Dropout
            nn.Dropout(0.2),
        )
        
        # Projection
        # Input features are reduced by 4 due to two poolings
        linear_input = (c * 2) * (input_features // 4)
        self.fc = nn.Sequential(
            nn.Linear(linear_input, output_features),
            nn.Dropout(0.25)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

class Student_AR_Transcriber(nn.Module):
    def __init__(self, input_features, output_features=256, model_complexity=24):
        super().__init__()
        
        # Significantly smaller architecture
        self.input_features = input_features
        self.language_hidden_size = model_complexity * 16 # e.g. 24*16 = 384 units
        
        # 1. Compact CNN
        self.acoustic_model = CompactConvStack(input_features, output_features, complexity=model_complexity)
        
        # 2. Efficient GRU (Faster than LSTM)
        # 1 Layer is usually enough for Student if Teacher is good
        self.language_model = nn.GRU(
            input_size=output_features + 88*2, 
            hidden_size=self.language_hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=False
        )

        # 3. Simple Projection
        # We keep the SKIP CONNECTION from your thesis model because it helps accuracy 
        # with almost zero cost.
        self.language_post = nn.Sequential(
            nn.Linear(self.language_hidden_size + output_features, 256),
            nn.ReLU(),
            nn.Linear(256, 88 * 5)
        )

        self.class_embedding = nn.Embedding(5, 2)
        self.melspectrogram = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)

    def forward(self, mel, gt_label=None, teacher_forcing_ratio=1.0):
        # 1. Acoustic Features
        acoustic_out = self.acoustic_model(mel)
        
        if gt_label is None or teacher_forcing_ratio < 1.0:
            # === Slow Path (Inference) ===
            h = self.init_rnn_hidden(mel.shape[0], mel.device)
            prev_out = torch.zeros((mel.shape[0], 1, 88 * 2)).to(mel.device)
            all_logits = []

            for i in range(acoustic_out.shape[1]):
                current_audio = acoustic_out[:, i:i+1, :]
                current_data = torch.cat((current_audio, prev_out), dim=2)
                
                # GRU returns (output, h) - No cell state c
                current_rnn_out, h = self.language_model(current_data, h)
                
                # Skip connection
                combined = torch.cat((current_rnn_out, current_audio), dim=2)
                logits = self.language_post(combined).view((mel.shape[0], 1, 88, 5))
                all_logits.append(logits)

                if gt_label is not None and random.random() < teacher_forcing_ratio:
                    current_gt = gt_label[:, i:i+1, :].type(torch.LongTensor).to(mel.device)
                    prev_out = self.class_embedding(current_gt).view(mel.shape[0], 1, 88 * 2)
                else:
                    pred_labels = torch.argmax(logits, dim=3)
                    prev_out = self.class_embedding(pred_labels).view(mel.shape[0], 1, 88 * 2)

            return torch.cat(all_logits, dim=1)
        else:
            # === Fast Path (Training) ===
            prev_gt = torch.cat((
                torch.zeros((gt_label.shape[0], 1, gt_label.shape[2]), device=mel.device, dtype=torch.long), 
                gt_label[:, :-1, :].type(torch.LongTensor).to(mel.device)
            ), dim=1)
            prev_embedded = self.class_embedding(prev_gt).view(mel.shape[0], -1, 88 * 2)
            concated_data = torch.cat((acoustic_out, prev_embedded), dim=2)
            
            rnn_out, _ = self.language_model(concated_data)
            
            combined = torch.cat((rnn_out, acoustic_out), dim=2)
            return self.language_post(combined).view(mel.shape[0], -1, 88, 5)

    def init_rnn_hidden(self, batch_size, device):
        # GRU has only hidden state h, shape (num_layers, batch, hidden_size)
        return torch.zeros(1, batch_size, self.language_hidden_size, device=device)
