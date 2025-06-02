import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import time

# الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# التحويلات
transform = T.Compose([
    T.RandomResizedCrop(96),
    T.RandomHorizontalFlip(),
    T.ColorJitter(),
    T.ToTensor()
])

# تحميل البيانات
print("📥 Loading STL-10...")
train_dataset = datasets.STL10(root='./data', split='unlabeled', transform=transform, download=True)
eval_dataset = datasets.STL10(root='./data', split='train', transform=T.ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

# نموذج ResNet50
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return z

# Contrastive Loss
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature
    labels = torch.arange(len(z1)).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# تدريب
model = Encoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5
losses = []
f1_scores = []

start = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (x1, _), (x2, _) in zip(train_loader, train_loader):
        x1, x2 = x1.to(device), x2.to(device)
        z1 = model(x1)
        z2 = model(x2)
        loss = contrastive_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    # تقييم F1 بعد كل Epoch
    model.eval()
    features, true_labels = [], []
    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(device)
            z = model(x)
            features.append(z.cpu().numpy())
            true_labels.extend(y.numpy())

    features = np.concatenate(features)
    true_labels = np.array(true_labels)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(features, true_labels)
    preds = clf.predict(features)

    f1 = f1_score(true_labels, preds, average='macro')
    f1_scores.append(f1)
    print(f" Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | F1 Score: {f1:.4f}")

end = time.time()
print(f" Total Training Time: {end - start:.2f} seconds")

# Evaluation Final
precision = precision_score(true_labels, preds, average='macro')
recall = recall_score(true_labels, preds, average='macro')

print(f" Final Precision: {precision:.4f}")
print(f" Final Recall:    {recall:.4f}")

# Loss Curve
plt.figure()
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# F1 Score Curve
plt.figure()
plt.plot(range(1, epochs + 1), f1_scores, marker='o', color='green')
plt.title("F1 Score vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid()
plt.show()

# Confusion Matrix
cm = confusion_matrix(true_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()
