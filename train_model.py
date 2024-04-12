import os
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
from itertools import product
from gatedConformer import Conformer
from gatedConformer import EmotionClassifier
from gatedConformer import GateLoss

use_cpu = False

if torch.cuda.is_available() and not use_cpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

# Parameters

# Parameters

num_labels = 7
learning_rate = 0.0001
batch_size_train = 32
batch_size_inference = 1
blocks = 1  # conformer blocks
dropout = 0.1  # effects the dropout all over the model
inputdim = 80  # based on filter banks from the mfbs and potential delta / delta delta
modeldim = 80  # hidden number of features the model works with
subsamplereduction = (
    4  # reduces the time dimension at the start of the model by this factor
)
feedforwardexpansion = 4  # increases the hidden dimension in the feed forward modules
feedforwardweight = 0.5  # weight to ff modules
pointwiseexpansion = 2  # don't alter, brought back by glu
kernel_size = 31  # needs to be odd since we are using 'same' padding
attentionheads = 4  # feel free to change
temperature = 1  # if temp is low, the probabilities take more extreme values.
gatethreshold = 0.5  # if Prob of execute > gatethreshold, execute
gateregularizer = 0.1  # loss function parameter on gate_loss
gate_activation = 1.1  # Determines when to turn gates on during training epochs, make greater than 1 for no gates
epochs = 10
k_folds = 5
epochs_kfolds = 3
freeze = False

label_to_int = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Sad": 4,
    "Neutral": 5,
    "Background": 6,
}

BED_train_features = np.load("dataformodel/BED_train_features.npy")
BED_testseen_features = np.load("dataformodel/BED_testseen_features.npy")
BED_testunseen_features = np.load("dataformodel/BED_testunseen_features.npy")
CREMAD_train_features = np.load("dataformodel/CREMAD_train_features.npy")
CREMAD_testseen_features = np.load("dataformodel/CREMAD_testseen_features.npy")
CREMAD_testunseen_features = np.load("dataformodel/CREMAD_testunseen_features.npy")
RAVDESS_train_features = np.load("dataformodel/RAVDESS_train_features.npy")
RAVDESS_testseen_features = np.load("dataformodel/RAVDESS_testseen_features.npy")
RAVDESS_testunseen_features = np.load("dataformodel/RAVDESS_testunseen_features.npy")
MSSNSD_train_features = np.load("dataformodel/MSSNSD_train_features.npy")
MSSNSD_testseen_features = np.load("dataformodel/MSSNSD_testseen_features.npy")
BED_train_labels = np.load("dataformodel/BED_train_labels.npy")
BED_testseen_labels = np.load("dataformodel/BED_testseen_labels.npy")
BED_testunseen_labels = np.load("dataformodel/BED_testunseen_labels.npy")
CREMAD_train_labels = np.load("dataformodel/CREMAD_train_labels.npy")
CREMAD_testseen_labels = np.load("dataformodel/CREMAD_testseen_labels.npy")
CREMAD_testunseen_labels = np.load("dataformodel/CREMAD_testunseen_labels.npy")
RAVDESS_train_labels = np.load("dataformodel/RAVDESS_train_labels.npy")
RAVDESS_testseen_labels = np.load("dataformodel/RAVDESS_testseen_labels.npy")
RAVDESS_testunseen_labels = np.load("dataformodel/RAVDESS_testunseen_labels.npy")
MSSNSD_train_labels = np.load("dataformodel/MSSNSD_train_labels.npy")
MSSNSD_testseen_labels = np.load("dataformodel/MSSNSD_testseen_labels.npy")

# convert labels to integers so torch can use them
BED_train_labels = np.array([label_to_int[label] for label in BED_train_labels])
BED_testseen_labels = np.array([label_to_int[label] for label in BED_testseen_labels])
BED_testunseen_labels = np.array(
    [label_to_int[label] for label in BED_testunseen_labels]
)
CREMAD_train_labels = np.array([label_to_int[label] for label in CREMAD_train_labels])
CREMAD_testseen_labels = np.array(
    [label_to_int[label] for label in CREMAD_testseen_labels]
)
CREMAD_testunseen_labels = np.array(
    [label_to_int[label] for label in CREMAD_testunseen_labels]
)
RAVDESS_train_labels = np.array([label_to_int[label] for label in RAVDESS_train_labels])
RAVDESS_testseen_labels = np.array(
    [label_to_int[label] for label in RAVDESS_testseen_labels]
)
RAVDESS_testunseen_labels = np.array(
    [label_to_int[label] for label in RAVDESS_testunseen_labels]
)
MSSNSD_train_labels = np.array([label_to_int[label] for label in MSSNSD_train_labels])
MSSNSD_testseen_labels = np.array(
    [label_to_int[label] for label in MSSNSD_testseen_labels]
)

BED_train_features_tensor = torch.tensor(BED_train_features, dtype=torch.float32)
BED_testseen_features_tensor = torch.tensor(BED_testseen_features, dtype=torch.float32)
BED_testunseen_features_tensor = torch.tensor(
    BED_testunseen_features, dtype=torch.float32
)
CREMAD_train_features_tensor = torch.tensor(CREMAD_train_features, dtype=torch.float32)
CREMAD_testseen_features_tensor = torch.tensor(
    CREMAD_testseen_features, dtype=torch.float32
)
CREMAD_testunseen_features_tensor = torch.tensor(
    CREMAD_testunseen_features, dtype=torch.float32
)
RAVDESS_train_features_tensor = torch.tensor(
    RAVDESS_train_features, dtype=torch.float32
)
RAVDESS_testseen_features_tensor = torch.tensor(
    RAVDESS_testseen_features, dtype=torch.float32
)
RAVDESS_testunseen_features_tensor = torch.tensor(
    RAVDESS_testunseen_features, dtype=torch.float32
)
MSSNSD_train_features_tensor = torch.tensor(MSSNSD_train_features, dtype=torch.float32)
MSSNSD_testseen_features_tensor = torch.tensor(
    MSSNSD_testseen_features, dtype=torch.float32
)
BED_train_labels_tensor = torch.tensor(BED_train_labels, dtype=torch.long)
BED_testseen_labels_tensor = torch.tensor(BED_testseen_labels, dtype=torch.long)
BED_testunseen_labels_tensor = torch.tensor(BED_testunseen_labels, dtype=torch.long)
CREMAD_train_labels_tensor = torch.tensor(CREMAD_train_labels, dtype=torch.long)
CREMAD_testseen_labels_tensor = torch.tensor(CREMAD_testseen_labels, dtype=torch.long)
CREMAD_testunseen_labels_tensor = torch.tensor(
    CREMAD_testunseen_labels, dtype=torch.long
)
RAVDESS_train_labels_tensor = torch.tensor(RAVDESS_train_labels, dtype=torch.long)
RAVDESS_testseen_labels_tensor = torch.tensor(RAVDESS_testseen_labels, dtype=torch.long)
RAVDESS_testunseen_labels_tensor = torch.tensor(
    RAVDESS_testunseen_labels, dtype=torch.long
)
MSSNSD_train_labels_tensor = torch.tensor(MSSNSD_train_labels, dtype=torch.long)
MSSNSD_testseen_labels_tensor = torch.tensor(MSSNSD_testseen_labels, dtype=torch.long)

# BED_train_dataset = TensorDataset(BED_train_features_tensor, BED_train_labels_tensor)
# BED_testseen_dataset = TensorDataset(
#     BED_testseen_features_tensor, BED_testseen_labels_tensor
# )
# BED_testunseen_dataset = TensorDataset(
#     BED_testunseen_features_tensor, BED_testunseen_labels_tensor
# )
# CREMAD_train_dataset = TensorDataset(
#     CREMAD_train_features_tensor, CREMAD_train_labels_tensor
# )
# CREMAD_testseen_dataset = TensorDataset(
#     CREMAD_testseen_features_tensor, CREMAD_testseen_labels_tensor
# )
# CREMAD_testunseen_dataset = TensorDataset(
#     CREMAD_testunseen_features_tensor, CREMAD_testunseen_labels_tensor
# )
# RAVDESS_train_dataset = TensorDataset(
#     RAVDESS_train_features_tensor, RAVDESS_train_labels_tensor
# )
# RAVDESS_testseen_dataset = TensorDataset(
#     RAVDESS_testseen_features_tensor, RAVDESS_testseen_labels_tensor
# )
# RAVDESS_testunseen_dataset = TensorDataset(
#     RAVDESS_testunseen_features_tensor, RAVDESS_testunseen_labels_tensor
# )
# MSSNSD_train_dataset = TensorDataset(
#     MSSNSD_train_features_tensor, MSSNSD_train_labels_tensor
# )
# MSSNSD_testseen_dataset = TensorDataset(
#     MSSNSD_testseen_features_tensor, MSSNSD_testseen_labels_tensor
# )


# Inference batch size is 1
# BED_train_loader = DataLoader(
#     BED_train_dataset, batch_size=batch_size_train, shuffle=True
# )
# BED_train_loader_test = DataLoader(
#     BED_train_dataset, batch_size=batch_size_inference, shuffle=False
# )
# BED_testseen_loader = DataLoader(
#     BED_testseen_dataset, batch_size=batch_size_inference, shuffle=False
# )
# BED_testunseen_loader = DataLoader(
#     BED_testunseen_dataset, batch_size=batch_size_inference, shuffle=False
# )
# CREMAD_train_loader = DataLoader(
#     CREMAD_train_dataset, batch_size=batch_size_train, shuffle=True
# )
# CREMAD_train_loader_test = DataLoader(
#     CREMAD_train_dataset, batch_size=batch_size_inference, shuffle=False
# )
# CREMAD_testseen_loader = DataLoader(
#     CREMAD_testseen_dataset, batch_size=batch_size_inference, shuffle=False
# )
# CREMAD_testunseen_loader = DataLoader(
#     CREMAD_testunseen_dataset, batch_size=batch_size_inference, shuffle=False
# )
# RAVDESS_train_loader = DataLoader(
#     RAVDESS_train_dataset, batch_size=batch_size_train, shuffle=True
# )
# RAVDESS_train_loader_test = DataLoader(
#     RAVDESS_train_dataset, batch_size=batch_size_inference, shuffle=False
# )
# RAVDESS_testseen_loader = DataLoader(
#     RAVDESS_testseen_dataset, batch_size=batch_size_inference, shuffle=False
# )
# RAVDESS_testunseen_loader = DataLoader(
#     RAVDESS_testunseen_dataset, batch_size=batch_size_inference, shuffle=False
# )
# MSSNSD_train_loader = DataLoader(
#     MSSNSD_train_dataset, batch_size=batch_size_train, shuffle=True
# )
# MSSNSD_train_loader_test = DataLoader(
#     MSSNSD_train_dataset, batch_size=batch_size_inference, shuffle=False
# )
# MSSNSD_testseen_loader = DataLoader(
#     MSSNSD_testseen_dataset, batch_size=batch_size_inference, shuffle=False
# )

# BED and MSSNSD for quick testing
# bedms_train_features = torch.cat(
#     (
#         BED_train_features_tensor,
#         MSSNSD_train_features_tensor,
#     )
# )

# bedms_train_labels = torch.cat(
#     (
#         BED_train_labels_tensor,
#         MSSNSD_train_labels_tensor,
#     )
# )

# bedms_testseen_features = torch.cat(
#     (
#         BED_testseen_features_tensor,
#         MSSNSD_testseen_features_tensor,
#     )
# )
# bedms_testseen_labels = torch.cat(
#     (
#         BED_testseen_labels_tensor,
#         MSSNSD_testseen_labels_tensor,
#     )
# )

# bedms_testunseen_features = torch.cat(
#     (
#         BED_testunseen_features_tensor,
#         MSSNSD_testseen_features_tensor,
#     )
# )
# bedms_testunseen_labels = torch.cat(
#     (
#         BED_testunseen_labels_tensor,
#         MSSNSD_testseen_labels_tensor,
#     )
# )

# bedms_train_dataset = TensorDataset(bedms_train_features, bedms_train_labels)
# bedms_testseen_dataset = TensorDataset(bedms_testseen_features, bedms_testseen_labels)
# bedms_testunseen_dataset = TensorDataset(
#     bedms_testunseen_features, bedms_testunseen_labels
# )

# bedms_train_loader = DataLoader(
#     bedms_train_dataset, batch_size=batch_size_train, shuffle=True
# )
# bedms_train_loader_test = DataLoader(
#     bedms_train_dataset, batch_size=batch_size_inference, shuffle=False
# )
# bedms_testseen_loader = DataLoader(
#     bedms_testseen_dataset, batch_size=batch_size_inference, shuffle=False
# )
# bedms_testunseen_loader = DataLoader(
#     bedms_testunseen_dataset, batch_size=batch_size_inference, shuffle=False
# )

# aggregate datasets below
train_features = torch.cat(
    (
        BED_train_features_tensor,
        CREMAD_train_features_tensor,
        RAVDESS_train_features_tensor,
        MSSNSD_train_features_tensor,
    )
)
train_labels = torch.cat(
    (
        BED_train_labels_tensor,
        CREMAD_train_labels_tensor,
        RAVDESS_train_labels_tensor,
        MSSNSD_train_labels_tensor,
    )
)

testseen_features = torch.cat(
    (
        BED_testseen_features_tensor,
        CREMAD_testseen_features_tensor,
        RAVDESS_testseen_features_tensor,
        MSSNSD_testseen_features_tensor,
    )
)
testseen_labels = torch.cat(
    (
        BED_testseen_labels_tensor,
        CREMAD_testseen_labels_tensor,
        RAVDESS_testseen_labels_tensor,
        MSSNSD_testseen_labels_tensor,
    )
)

testunseen_features = torch.cat(
    (
        BED_testunseen_features_tensor,
        CREMAD_testunseen_features_tensor,
        RAVDESS_testunseen_features_tensor,
        MSSNSD_testseen_features_tensor,
    )
)
testunseen_labels = torch.cat(
    (
        BED_testunseen_labels_tensor,
        CREMAD_testunseen_labels_tensor,
        RAVDESS_testunseen_labels_tensor,
        MSSNSD_testseen_labels_tensor,
    )
)

train_dataset = TensorDataset(train_features, train_labels)
testseen_dataset = TensorDataset(testseen_features, testseen_labels)
testunseen_dataset = TensorDataset(testunseen_features, testunseen_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
train_loader_test = DataLoader(
    train_dataset, batch_size=batch_size_inference, shuffle=False
)
testseen_loader = DataLoader(
    testseen_dataset, batch_size=batch_size_inference, shuffle=False
)
testunseen_loader = DataLoader(
    testunseen_dataset, batch_size=batch_size_inference, shuffle=False
)


def initialize_model(gates_on):
    conformer = Conformer(
        inputdim,
        modeldim,
        subsamplereduction,
        feedforwardexpansion,
        feedforwardweight,
        pointwiseexpansion,
        kernel_size,
        attentionheads,
        blocks,
        dropout,
        temperature,
        gatethreshold,
        gates_on=gates_on,
    )
    model = EmotionClassifier(conformer, num_labels, modeldim)
    criterion = GateLoss(gateregularizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, train_loader, epochs, toprint=True):
    model.train()
    gates_on = False
    gate_activation_epoch = int(epochs * gate_activation)
    for epoch in range(epochs):
        start_time = time.time()
        current_loss = 0.0
        if epoch == gate_activation_epoch:
            model.conformer.enable_gates()
            gates_on = True
            if freeze:
                for param in model.parameters():
                    param.requires_grad = False
                for name, param in model.named_parameters():
                    if "gated" in name:
                        param.requires_grad = True
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model.conformer.reset_gate_values()
            outputs = model(inputs)
            loss = criterion(outputs, labels, model.conformer.gate_values)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        epoch_time = time.time() - start_time
        if toprint:
            print(
                f"Epoch {epoch + 1} Gates {'On' if gates_on else 'Off'}, Loss: {(current_loss / len(train_loader)):.2f}, Time: {epoch_time:.2f} seconds"
            )


skips = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
}

module_count = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
}

labels_actual = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Neutral",
    6: "Background",
}


def evaluate_model(model, loader, toprint=True, name=""):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for input, label in loader:  # Batch size always 1
            input = input.to(device)
            label = label.to(device)
            model.conformer.reset_gate_values()
            logits = model(input)
            probabilities = F.softmax(logits, dim=-1)
            _, predicted = torch.max(probabilities, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # calculate skips
            if model.conformer.gate_values.nelement() != 0:
                emotion_key = label.item()
                skips_current = torch.sum(
                    model.conformer.gate_values <= gatethreshold
                ).item()
                module_count_current = model.conformer.gate_values.numel()
                skips[emotion_key] += skips_current
                module_count[emotion_key] += module_count_current
    skip_emotion_total = 0
    skip_emotion_mean = 0
    skip_background = 0
    skip_difference = 0
    if model.conformer.gate_values.nelement() != 0:
        for i in range(num_labels):
            skip_percentage = skips[i] / module_count[i]
            if labels_actual[i] == "Background":
                skip_background = skip_percentage
            else:
                skip_emotion_total += skip_percentage
            if toprint:
                print(
                    f"{name} {labels_actual[i]} skip percentage: {skip_percentage:.2f}"
                )
        skip_emotion_mean = skip_emotion_total / (num_labels - 1)
        skip_difference = skip_background - skip_emotion_mean
        if toprint:
            print(f"Skip difference: {skip_difference}")
    accuracy = 100 * correct / total
    epoch_time = time.time() - start_time
    if toprint:
        print(f"Accuracy on {name} Data: {accuracy}, Time: {epoch_time:.2f} seconds")
    return accuracy, skip_difference


def count_parameters(model, long):
    all_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {all_params}")
    if long:
        for name, module in model.named_modules():
            if isinstance(module, nn.Module):
                total_params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                print(
                    f"{module.__class__.__name__} {name} - Total Parameters: {total_params}, Trainable Parameters: {trainable_params}"
                )


def save_weights(model):
    path = "gated_conformer_model_weights.pth"
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_weights(model):
    path = "gated_conformer_model_weights.pth"
    model.load_state_dict(torch.load(path))
    print(f"Model weights loaded from {path}")


hyperparameters = {
    "learning_rate": [0.00005],
    "dropout": [0.1],
    "modeldim": [100],
}

best_accuracy = 0
best_skip_difference = 0
best_params_accuracy = {}
best_params_skip_difference = {}
param_options = list(product(*hyperparameters.values()))
for option in param_options:
    current_parameters = dict(zip(hyperparameters.keys(), option))
    print(f"Testing {current_parameters}")
    if "learning_rate" in current_parameters:
        learning_rate = current_parameters["learning_rate"]
    if "batch_size_train" in current_parameters:
        batch_size_train = current_parameters["batch_size_train"]
    if "dropout" in current_parameters:
        dropout = current_parameters["dropout"]
    if "modeldim" in current_parameters:
        modeldim = current_parameters["modeldim"]
    if "temperature" in current_parameters:
        temperature = current_parameters["temperature"]
    if "gateregularizer" in current_parameters:
        gateregularizer = current_parameters["gateregularizer"]
    # add as needed

    kfold = KFold(n_splits=k_folds, shuffle=True)
    scores = []
    skip_differences = []
    start_time = time.time()
    for i, (train_ids, valid_ids) in enumerate(kfold.split(train_features)):
        print(f"Fold {i + 1}")
        train_load = DataLoader(
            Subset(TensorDataset(train_features, train_labels), train_ids),
            batch_size=batch_size_train,
            shuffle=True,
        )
        valid_load = DataLoader(
            Subset(TensorDataset(train_features, train_labels), valid_ids),
            batch_size=batch_size_inference,
            shuffle=False,
        )
        model, criterion, optimizer = initialize_model(gates_on=False)
        model = model.to(device)
        criterion = criterion.to(device)
        count_parameters(model, long=False)
        train_model(model, criterion, optimizer, train_load, epochs_kfolds, True)
        accuracy, skip_difference = evaluate_model(
            model, valid_load, True, f"Fold {i + 1}"
        )
        scores.append(accuracy)
        skip_differences.append(skip_difference)
    avg_accuracy = sum(scores) / len(scores)
    avg_skip_difference = sum(skip_differences) / len(skip_differences)
    kfold_time = time.time() - start_time

    print(
        f"Avg accuracy and skip dif for {k_folds} folds with {current_parameters}: {avg_accuracy}, {avg_skip_difference}, Time: {kfold_time:.2f} seconds"
    )
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_params_accuracy = current_parameters
    if avg_skip_difference > best_skip_difference:
        best_skip_difference = avg_skip_difference
        best_params_skip_difference = current_parameters
print(f"Best accuracy params: {best_params_accuracy}")
print(f"Best avg accuracy: {best_accuracy}")
print(f"Best skip difference params: {best_params_skip_difference}")
print(f"Best avg skip difference: {best_skip_difference}")
