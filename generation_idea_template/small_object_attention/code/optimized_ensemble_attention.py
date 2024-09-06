"""
Implement an ensemble of SEAttention models, each trained with different configurations such as varied data augmentations or initializations
Apply a specific ensemble strategy like boosting, where models are trained sequentially with a focus on the errors of previous models
Develop a mechanism to optimize the weights assigned to each model's output, such as using a meta-learning approach
Evaluate the ensemble's performance on small target detection tasks using precision, recall, and F1-score, comparing against individual SEAttention models and other enhanced models

"""

# Modified code

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnsembleSEAttention(nn.Module):
    def __init__(self, num_models, channel=512, reduction=16):
        super().__init__()
        self.models = nn.ModuleList([SEAttention(channel, reduction) for _ in range(num_models)])
        self.weights = nn.Parameter(torch.ones(num_models, requires_grad=True))

    def forward(self, x):
        preds = torch.stack([model(x) for model in self.models], dim=0)
        weighted_preds = self.weights.view(-1, 1, 1, 1, 1) * preds
        return weighted_preds.sum(dim=0)

    def train_ensemble(self, train_loader, criterion, optimizer, epochs=10):
        self.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                # Update weights based on errors (Boosting)
                with torch.no_grad():
                    errors = (output - target).abs().sum(dim=[1,2,3])
                    self.weights -= 0.1 * errors
                    self.weights = torch.clamp(self.weights, min=0)

    def evaluate(self, test_loader):
        self.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, target in test_loader:
                output = self.forward(data)
                preds = torch.round(output).cpu().numpy()
                y_true.extend(target.cpu().numpy())
                y_pred.extend(preds)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return precision, recall, f1


# Example usage
if __name__ == '__main__':
    ensemble_model = EnsembleSEAttention(num_models=3)
    ensemble_model.train_ensemble(train_loader=None, criterion=None, optimizer=None)  # Replace with actual data
    precision, recall, f1 = ensemble_model.evaluate(test_loader=None)  # Replace with actual data
    print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')