import torch
import pickle
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, timing_metrics,
    forgetting_metrics, cpu_usage_metrics, confusion_matrix_metrics,
    disk_usage_metrics
)
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.models import SimpleMLP  # Needed to load state_dict correctly

with open("scenario.pkl", "rb") as f:
    scenario = pickle.load(f)

model = SimpleMLP(num_classes=scenario.n_classes)
model.load_state_dict(torch.load("model.pt"))

tb_logger = TensorboardLogger()
text_logger = TextLogger(open('log.txt', 'a'))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False, stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=64, train_epochs=20, eval_mb_size=64,
    evaluator=eval_plugin
)

with open("strategy.pkl", "wb") as f:
    pickle.dump(cl_strategy, f)
