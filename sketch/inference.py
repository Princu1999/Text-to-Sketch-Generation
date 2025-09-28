from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_sketch(model, device, max_len=200):
    """ Generate a sketch from the trained model. """
    model.to(device)
    model.eval()

    with torch.no_grad():
        real_sketch = torch.tensor(test_dataset[4][0]).unsqueeze(0)
        input_sketch = real_sketch.to(device)  # Pass a real sketch
        _, z, _ = model.encoder(input_sketch)
        #z = torch.randn(1, model.encoder.latent_size).to(device)
        seq = [torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32, device=device)]  # Start token

        for _ in range(max_len):
            decoder_input = torch.cat(seq, dim=0).unsqueeze(0)
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = model.decoder(latent_vector.unsqueeze(0), decoder_input)

            # Sample from the mixture model
            idx = torch.multinomial(pi[0, -1], 1).item()
            dx = torch.normal(mu_x[0, -1, idx], sigma_x[0, -1, idx]).item()
            dy = torch.normal(mu_y[0, -1, idx], sigma_y[0, -1, idx]).item()
            pen = torch.multinomial(q[0, -1], 1).item()

            seq.append(torch.tensor([[dx, dy, pen == 0, pen == 1, pen == 2]], dtype=torch.float32, device=device))
            if pen == 2:  # End of sequence
                break

        return torch.cat(seq, dim=0).cpu().numpy()





def sketch_latent(model, device, latent,max_len = 100):
    model.to(device)
    model.eval()

    with torch.no_grad():

        seq = [torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32, device=device)]  # Start token

        for _ in range(max_len):
            decoder_input = torch.cat(seq, dim=0).unsqueeze(0)
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = model.decoder(latent.unsqueeze(0) , decoder_input)

            # Select max probability index instead of sampling
            idx = torch.argmax(pi[0, -1]).item()
            dx = mu_x[0, -1, idx].item()  # Use mean instead of sampling
            dy = mu_y[0, -1, idx].item()  # Use mean instead of sampling
            pen = torch.argmax(q[0, -1]).item()  # Select the most probable pen state

            seq.append(torch.tensor([[dx, dy, pen == 0, pen == 1, pen == 2]], dtype=torch.float32, device=device))
            if pen == 2:  # End of sequence
                break

        return torch.cat(seq, dim=0).cpu().numpy()



def animate_sketch(stroke_data):
    """Create an animation of a sketch."""

    fig, ax = plt.subplots()
    x, y = 0, 0
    strokes = []
    for i in range(len(stroke_data)):
        dx, dy, p1, p2, p3 = stroke_data[i]
        x += dx
        y -= dy
        strokes.append((x, y, p1, p2, p3))

    line, = ax.plot([], [], 'k-')  # Create an empty line object
    ax.set_xlim([min(s[0] for s in strokes) - 10, max(s[0] for s in strokes) + 10])  # Adjust x limits
    ax.set_ylim([min(s[1] for s in strokes) - 10, max(s[1] for s in strokes) + 10])  # Adjust y limits


    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x_seq, y_seq = [], []
        for j in range(i + 1):
            x, y, p1, p2, p3 = strokes[j]
            x_seq.append(x)
            y_seq.append(y)
            if p1 == 0:  # End of stroke
                line.set_data(x_seq, y_seq)
                x_seq, y_seq = [], []

        if len(x_seq) > 0:  # Plot remaining points if the loop ends before a complete stroke
            line.set_data(x_seq, y_seq)

        return line,

    ani = animation.FuncAnimation(fig, animate, frames=len(strokes), blit=True, interval=50, init_func=init)

    plt.close(fig) #added to prevent showing static image
    return ani




def plot_sketch(stroke_data):
    """ Convert stroke sequence into a plot. """
    x, y = 0, 0
    strokes = []
    for i in range(len(stroke_data)):
        dx, dy, p1, p2, p3 = stroke_data[i]
        x += dx
        y -= dy
        strokes.append((x, y, p1, p2, p3))

    x_seq, y_seq = [], []
    for x, y, p1, p2, p3 in strokes:
        x_seq.append(x)
        y_seq.append(y)
        if p1 == 0:  # End of stroke
            plt.plot(x_seq, y_seq, 'k-')
            x_seq, y_seq = [], []
        if p3 ==1:
          break
    if len(x_seq) == len(stroke_data):
        plt.plot(x_seq, y_seq, 'k-')
    plt.show()




def most_probable_latent(gmm):
    """
    Find the most probable latent vector from the GMM.
    """
    best_latent = None
    best_prob = -float('inf')

    # Sample multiple latent vectors from the GMM
    samples, _ = gmm.sample(1000)  # Generate 1000 samples
    for sample in samples:
        prob = gmm.score_samples(sample.reshape(1, -1))  # Compute log probability
        if prob > best_prob:
            best_prob = prob
            best_latent = sample
    #print(best_prob)

    return torch.tensor(best_latent, dtype=torch.float32)  # Shape: (128,)





