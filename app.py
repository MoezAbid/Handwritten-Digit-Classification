# Imports
import gradio as gr
import torch
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LABELS = Path('class_names.txt').read_text()

# CNN
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x

MODEL_PATH = "model/cnn_model.pt"
IMAGE_SIZE = (28,28)
OUTPUT_DIM = 10
INPUT_DIM = IMAGE_SIZE
model = CNN(INPUT_DIM, OUTPUT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)

def transform_image(image):
    INPUT_HEIGHT = 28
    INPUT_WIDTH = 28
    SIZE_IN_PIXELS = (INPUT_HEIGHT, INPUT_WIDTH)
    NORMALIZATION_MEAN = 0.1307
    NORMALIZATION_STD = 0.3081
    transformations = T.Compose([
        T.ToTensor(),
        T.Resize(SIZE_IN_PIXELS),
        T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)])
    return transformations(image).unsqueeze(0)

def predict_digit(image):
    model.eval()
    image = transform_image(image)
    image = image.to(device)
    result = model(image)
    probabilities = torch.nn.functional.softmax(result[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    confidences = {LABELS[i]: v.item() for i, v in zip(indices, values)}
    return confidences

gr.Interface(fn=predict_digit,
title="Handwritten digit classification",
             inputs="sketchpad",
             outputs=["label"]
             ).launch()