from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, optimizer, device, epochs=10, w_kl=0.5):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    train_loss, val_loss = [] , []
    for epoch in range(epochs):
        w_kl = min(0.5, epoch / 20)
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            inputs, _ = batch

            inputs = inputs.to(device)
            s_0 = torch.zeros((inputs.shape[0], 1, 5), device=device)
            s_0[... , 2] = 1
            decoder_input = torch.cat((s_0, inputs[:, :-1, : ]), dim=1)

            optimizer.zero_grad()
            # The issue was with the slicing of the inputs and decoder inputs in the model call and the compute_loss call.
            # Original: pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma = model(inputs, decoder_input)
            # Original: loss = compute_loss(inputs[:, 1:, :], pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma, w_kl)
            # Modified to ensure consistent sequence lengths between target and predictions
            with torch.cuda.amp.autocast():
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma = model(inputs, decoder_input)
                #print(pi.shape)
                # Slice the decoder outputs to match the target sequence length
                loss = compute_loss(inputs, pi, mu_x, mu_y,
                                    sigma_x, sigma_y, rho_xy,
                                    q, mu, sigma, w_kl) #changing inputs[: , 1: , :] to inputs[: , :-1 ,:]
                # if (torch.isnan(loss)):
                #   print('loss is nan')

            scaler.scale(loss).backward()  # Scale gradients for stability


            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.abs().mean()}")


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()


            #optimizer.zero_grad()

            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # Clipping applied
            # optimizer.step()
            total_train_loss += loss.item()

        train_loss.append(total_train_loss/len(train_loader))
        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, _ = batch
                inputs = inputs.to(device)
                decoder_input = torch.cat((torch.zeros((inputs.shape[0], 1, 5), device=device), inputs[:, :-1, :]), dim=1)
                # Similar modifications for validation loop
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma = model(inputs, decoder_input)
                # Slice the decoder outputs to match the target sequence length
                loss = compute_loss(inputs, pi, mu_x, mu_y,
                                    sigma_x, sigma_y, rho_xy,
                                    q, mu, sigma, w_kl)
                total_val_loss += loss.item()


        val_loss.append(total_val_loss/len(val_loader))
        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_loader)}, Val Loss: {total_val_loss/len(val_loader)}")
    return train_loss, val_loss




def plot_train_val_loss(train_loss, val_loss):
  plt.figure(figsize=(10, 5))
  plt.plot(train_loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.show()

