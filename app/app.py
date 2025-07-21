import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load class labels for CIFAR-10
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load model architecture and weights
def get_model():
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10)
    )

    state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Define transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Grad-CAM setup
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        handle_fw = self.target_layer.register_forward_hook(forward_hook)
        handle_bw = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle_fw, handle_bw])

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = F.interpolate(grad_cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        return grad_cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# Instantiate model and GradCAM
model = get_model()
target_layer = model.layer4[-1]
grad_cam = GradCAM(model, target_layer)

def clear():
    return "", None, ""

def submit(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
         output = model(image_tensor)
         probs = F.softmax(output[0], dim=0)
         top_probs, top_idxs = torch.topk(probs, 3)

    top_preds = {classes[idx]: float(prob) for idx, prob in zip(top_idxs, top_probs)}
    top1_class = max(top_preds, key=top_preds.get)

    grad_cam_map = grad_cam.generate(image_tensor, top_idxs[0].item())
    grad_cam_map = np.uint8(255 * grad_cam_map)
    cam_img = Image.fromarray(grad_cam_map).resize(image.size).convert("L")

    heatmap = np.array(cam_img)
    orig_np = np.array(image.convert("RGB"))
    overlay_np = np.uint8(0.6 * orig_np + 0.4 * np.stack([heatmap]*3, axis=-1))
    overlay_img = Image.fromarray(overlay_np)
    return top_preds, overlay_img, top1_class

# Gradio Blocks interface with buttons and status
with gr.Blocks() as demo:
    gr.Markdown("## CIFAR-10 Image Classifier with Grad-CAM")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload CIFAR-10 Image")
        status = gr.Label(value="")  # Status label (empty by default)

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        top_preds = gr.Label(num_top_classes=3, label="Top 3 Predictions")
        cam_output = gr.Image(type="pil", label="Grad-CAM Overlay")

    # Make sure status is the third output here
    submit_btn.click(
        fn=submit,
        inputs=image_input,
        outputs=[top_preds, cam_output, status]
    )

    clear_btn.click(
        fn=lambda: ("", None, "Cleared"),  # Empty predictions, cleared Grad-CAM, status="Cleared"
        inputs=None,
        outputs=[top_preds, cam_output, status]
    )

demo.launch(share = True)

    


