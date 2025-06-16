from avalanche.models import SimpleMLP, SimpleCNN
from torchvision.models import resnet18
import torch.nn as nn
import torch
import pickle

model_name = "simplemlp"

def get_model(model_name: str, num_classes: int):
    if model_name.lower() == "simplemlp":
        return SimpleMLP(num_classes=num_classes)
    elif model_name.lower() == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name.lower() == "resnet18":
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

with open("scenario.pkl", "rb") as f:
    scenario = pickle.load(f)

model = get_model(model_name, num_classes=scenario.n_classes)
torch.save(model.state_dict(), "model.pt")
