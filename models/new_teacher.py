import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .mel import MelSpectrogram
from .constants import *

# ==========================================
# 1. Squeeze-and-Excitation Block (The "Harmonic Filter")
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # Compresses features to understand global distribution
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # Expands back to weight the channels
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Dynamic Feature Re-weighting
        return x * y.expand_as(x)

# ==========================================
# 2. Gated Acoustic Model
# ==========================================
class GatedConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, output_features // 16, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(output_features // 16)
        self.se1 = SEBlock(output_features // 16) # Gating

        # Layer 2
        self.conv2 = nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(output_features // 16)
        self.pool1 = nn.MaxPool2d((1, 2))
        self.drop1 = nn.Dropout(0.25)

        # Layer 3
        self.conv3 = nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(output_features // 8)
        self.se2 = SEBlock(output_features // 8) # Gating

        # Layer 4 (Pooling)
        self.pool2 = nn.MaxPool2d((1, 2))
        self.drop2 = nn.Dropout(0.25)

        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        
        # Convolution with Attention
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x) 
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se2(x)
        
        x = self.drop2(self.pool2(x))

        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

# ==========================================
# 3. Main Architecture (Fully Causal)
# ==========================================
class Thesis_AR_Transcriber(nn.Module):
    def __init__(self, input_features, output_features,
                 model_complexity_conv=48, model_complexity_lstm=48):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        
        model_size_conv = model_complexity_conv * 16 
        model_size_lstm = model_complexity_lstm * 16 
        self.language_hidden_size = model_size_lstm

        # 1. Acoustic Model with Gating
        self.acoustic_model = GatedConvStack(input_features, model_size_conv)
        
        # 2. Causual AR LSTM
        self.language_model = nn.LSTM(
            input_size=model_size_conv + 88*2, 
            hidden_size=model_size_lstm, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=False # STRICTLY CAUSAL
        )
        
        # 3. Output Projection with SKIP CONNECTION
        # Input size is: [LSTM State] + [Original Acoustic Features]
        self.language_post = nn.Sequential(
            nn.Linear(model_size_lstm + model_size_conv, 512),
            nn.ReLU(),
            nn.Linear(512, 88 * 5)
        )

        self.class_embedding = nn.Embedding(5, 2)
        self.melspectrogram = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)

    def forward(self, mel, gt_label=None, teacher_forcing_ratio=1.0):
        # 1. Acoustic Feature Extraction
        acoustic_out = self.acoustic_model(mel) 
        
        # --- AR LOOP ---
        
        if gt_label is None or teacher_forcing_ratio < 1.0:
            # === Slow Path (Inference) ===
            h, c = self.init_lstm_hidden(mel.shape[0], mel.device)
            prev_out = torch.zeros((mel.shape[0], 1, 88 * 2)).to(mel.device)
            all_logits = []

            for i in range(acoustic_out.shape[1]):
                # Current Step Audio
                current_audio = acoustic_out[:, i:i+1, :]
                
                # Input to LSTM: Audio + History
                current_data = torch.cat((current_audio, prev_out), dim=2)
                
                # LSTM Step
                current_lstm_out, (h, c) = self.language_model(current_data, (h, c))
                
                # NOVELTY: CONCATENATE AUDIO AGAIN
                # We feed the raw audio directly to the classifier alongside the LSTM thought.
                # This grounds the prediction in reality.
                combined_output = torch.cat((current_lstm_out, current_audio), dim=2)
                
                logits = self.language_post(combined_output)
                logits = logits.view((mel.shape[0], 1, 88, 5))
                all_logits.append(logits)

                # Sampling Logic
                if gt_label is not None and random.random() < teacher_forcing_ratio:
                    current_gt = gt_label[:, i:i+1, :].type(torch.LongTensor).to(mel.device)
                    prev_out = self.class_embedding(current_gt).view(mel.shape[0], 1, 88 * 2)
                else:
                    probs = torch.softmax(logits, dim=3)
                    pred_labels = torch.argmax(probs, dim=3)
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
            
            lstm_out, _ = self.language_model(concated_data)
            
            # Skip connection for Fast Path
            combined_output = torch.cat((lstm_out, acoustic_out), dim=2)
            
            total_result = self.language_post(combined_output).view(mel.shape[0], -1, 88, 5)
            
            return total_result

    def init_lstm_hidden(self, batch_size, device):
        h = torch.zeros(2, batch_size, self.language_hidden_size, device=device)
        c = torch.zeros(2, batch_size, self.language_hidden_size, device=device)
        return (h, c)
