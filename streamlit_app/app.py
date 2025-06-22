# streamlit_app/app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- MODEL DEFINITION ---
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    
        x = F.relu(self.conv2(x))    
        x = F.max_pool2d(x, 2)       
        x = self.dropout1(x)
        x = torch.flatten(x, 1)      
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- LOAD MODEL ---
@st.cache_resource
def load_model(path='digit_cnn.pth'):
    model = DigitCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- LOAD MNIST TEST DATA ---
@st.cache_resource
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    return test_dataset

# --- FILTER SAMPLES ---
def get_samples(model, dataset, digit, num_samples=5):
    matched_images = []
    with torch.no_grad():
        for image, label in dataset:
            output = model(image.unsqueeze(0))
            pred = torch.argmax(output, dim=1).item()
            if pred == digit:
                matched_images.append(image.squeeze(0).numpy())
            if len(matched_images) == num_samples:
                break
    return matched_images

# --- PLOT IMAGES ---
def display_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)

# --- STREAMLIT UI ---
st.title("MNIST Digit Generator (CNN Classifier)")

selected_digit = st.selectbox("Select a digit (0–9)", list(range(10)))

if st.button("Generate 5 Images"):
    with st.spinner("Generating..."):
        model = load_model()
        test_data = load_test_data()
        images = get_samples(model, test_data, selected_digit)
        if images:
            display_images(images)
        else:
            st.warning("Couldn't find enough samples for this digit.")
