import os

import torch
from PredictorWrapper import Predictor
from Data import compute_mean_and_std, get_data_loaders
from TransferModel import get_model_transfer_learning
from torch.utils.mobile_optimizer import optimize_for_mobile

# First let's get the class names from our data loaders
class_names = get_data_loaders()["train"].dataset.classes

model_transfer = get_model_transfer_learning(model_name="resnet152", n_classes=50)
model_transfer = model_transfer.cpu()
# Let's make sure we use the right weights by loading the
# best weights we have found during training
model_transfer.load_state_dict(
    torch.load("checkpoints/model_transfer152.pt", map_location="cpu")
)

# Wrap with predictor
mean, std = compute_mean_and_std()
predictor = Predictor(model_transfer, class_names, mean, std).cpu()

# Export using torch.jit.script
scripted_predictor = torch.jit.script(predictor)
# Generate normal predictor
scripted_predictor.save("checkpoints/transfer_exported.pt")

# Generate mobile optimized predictor
model_transfer.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model_transfer, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("checkpoints/model_mobile.pt")

# Print all labels to txt file
labels = []
for dir in os.listdir("landmark_images/train"):
    labels.append(dir)
labels.sort()
with open("checkpoints/labels.txt", "w") as f:
    for label in labels:
        splitted = label.split(".")
        f.write(splitted[1] + "\n")
