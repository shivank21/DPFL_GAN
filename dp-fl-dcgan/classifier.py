import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, classification_report
import os

model = AutoModelForSequenceClassification.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")
tokenizer = AutoTokenizer.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")

directory_path = "../images"  

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root=directory_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
predictions = []
ground_truth_labels = []

for image, label in dataloader:
    # Tokenize the input
    inputs = tokenizer(["8"], padding=True, truncation=True, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = torch.argmax(logits, dim=1).item()
    
    # Append predictions and ground truth labels
    predictions.append(predicted_label)
    ground_truth_labels.append(label.item())

f1 = f1_score(ground_truth_labels, predictions, average='micro')

# Print F1 score and other metrics
print("F1 Score:", f1)
print(classification_report(ground_truth_labels, predictions))
