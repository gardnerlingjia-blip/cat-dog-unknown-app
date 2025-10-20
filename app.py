
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -----------------------------
# 1. Define Your Model Class
# -----------------------------
class YourModelClass(nn.Module):
    def __init__(self, num_classes=3):  # cat, dog, unknown
        super(YourModelClass, self).__init__()
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 2. Load Model with Prefix Fix
# -----------------------------
@st.cache_resource
def load_model():
    model = YourModelClass(num_classes=3)
    state_dict = torch.load("best_model.pt", map_location="cpu")

    # Fix key mismatch (add/remove 'model.' prefix)
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if all(k.startswith("model.") for k in model_keys) and not all(k.startswith("model.") for k in ckpt_keys):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif not any(k.startswith("model.") for k in model_keys) and any(k.startswith("model.") for k in ckpt_keys):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# -----------------------------
# 3. Define Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("🐶🐱 Cat-Dog-Unknown Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

    classes = ["Cat", "Dog", "Unknown"]
    threshold = 0.6  # Confidence threshold

    if confidence < threshold:
        st.write(f"**Prediction:** Unknown (Confidence too low: {confidence*100:.2f}%)")
    else:
        st.write(f"**Prediction:** {classes[pred_idx]} ({confidence*100:.2f}%)")

    # Optional: Show all class probabilities
    st.write("Class probabilities:")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probs[0][i]*100:.2f}%")

