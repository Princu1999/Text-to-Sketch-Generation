from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchRNNEncoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, latent_size=128):
        super(SketchRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,num_layers=3, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_size * 2, latent_size)  # Bidirectional -> double size
        self.fc_sigma = nn.Linear(hidden_size * 2, latent_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)  # Concatenate bidirectional outputs
        mu = self.fc_mu(h_n)
        sigma_hat = self.fc_sigma(h_n)
        sigma = torch.exp(torch.clamp(sigma_hat/2, min=-5, max=5)) + 1e-8
        z = mu + sigma * torch.randn_like(sigma)  # Reparameterization trick
        return z, mu, sigma



class SketchRNNDecoder(nn.Module):
    def __init__(self, latent_size=128, hidden_size=256, M=2):
        super(SketchRNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(latent_size + 5, hidden_size,num_layers=3, batch_first=True,dropout=.3)
        self.fc_out = nn.Linear(hidden_size, 6*M + 3)
        self.init_weights()

    def init_weights(self):
      for name, param in self.named_parameters():
          if "weight_hh" in name:
              nn.init.orthogonal_(param)
          elif "weight_ih" in name:
              nn.init.xavier_uniform_(param)
          elif "bias" in name:
              param.data.fill_(0)

    def forward(self, z, inputs):
        batch_size, seq_len, _ = inputs.shape
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_input = torch.cat((inputs, z), dim=-1)
        h, _ = self.lstm(lstm_input)
        output = self.fc_out(h)

        # Extract GMM parameters
        M = output.shape[-1] // 6
        pi = F.softmax(torch.clamp(output[..., :M], min=-10, max=10), dim=-1)  # Mixture weights
        mu_x = output[..., M:2*M]
        mu_y = output[..., 2*M:3*M]
        sigma_x = torch.exp(torch.clamp(output[..., 3*M:4*M], min=-5, max=5)) + 1e-8
        sigma_y = torch.exp(torch.clamp(output[..., 4*M:5*M], min=-5, max=5)) + 1e-8
        rho_xy = torch.tanh(output[..., 5*M:6*M])

        # Extract categorical probabilities
        q_logits = output[..., 6*M:]
        q = F.softmax(torch.clamp(q_logits, min=-10, max=10), dim=-1)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q



class SketchRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, latent_size=128, M=20):
        super(SketchRNN, self).__init__()
        self.encoder = SketchRNNEncoder(input_size, hidden_size, latent_size)
        self.decoder = SketchRNNDecoder(latent_size, hidden_size * 2, M)

    def forward(self, x, decoder_input):
        z, mu, sigma = self.encoder(x)
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = self.decoder(z, decoder_input)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma





