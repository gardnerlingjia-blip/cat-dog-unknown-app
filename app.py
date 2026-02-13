import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# -----------------------------
# 0. App Page Setup + Header
# -----------------------------
st.set_page_config(page_title="Pet Image Classifier", page_icon="🐾", layout="centered")

st.title("🐶🐱 Pet Image Classifier")
st.markdown("""
Upload an image to classify it as **Cat**, **Dog**, or **Other (Unknown)**.

**Model:** ResNet18 (3 classes)  
**Deployment:** Docker + Google Cloud Run  
""")

# -----------------------------
# 1. Define Your Model Class
# -----------------------------
class YourModelClass(nn.Module):
    def __init__(self, num_classes: int = 3):  # Cat, Dog, Unknown
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 2. Load Model with Prefix Fix (cached)
# -----------------------------
MODEL_PATH = Path(__file__).parent / "best_model.pt"

@st.cache_resource
def load_model():
    model = YourModelClass(num_classes=3)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 4. Inference UI
# -----------------------------
classes = ["Cat", "Dog", "Unknown"]
threshold = 0.6  # Confidence threshold

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # shape: [3]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    st.subheader("Prediction Result")

    if confidence < threshold:
        st.markdown(f"### 🏆 Top Prediction: **Unknown** ({confidence*100:.2f}%)")
        st.caption(f"Confidence is below the threshold ({threshold*100:.0f}%).")
    else:
        st.markdown(f"### 🏆 Top Prediction: **{classes[pred_idx]}** ({confidence*100:.2f}%)")

    st.write("Class probabilities:")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {float(probs[i].item())*100:.2f}%")

else:
    st.info("Tip: Try uploading a random non-pet image — it should ideally land in **Unknown**.")

# -----------------------------
# 5. Footer / Explanation
# -----------------------------
st.markdown("---")
st.markdown("### How it works")
st.markdown("""
- The uploaded image is resized to **224×224** and normalized.
- A **ResNet18** model predicts one of three classes: **Cat**, **Dog**, **Unknown**.
- Probabilities are computed using a **softmax** over model outputs.
- The app is packaged with **Docker** and deployed on **Google Cloud Run**.
""")


