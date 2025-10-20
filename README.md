🐶🐱 Cat-Dog-Unknown Classifier
A simple Streamlit app that classifies uploaded images as Cat, Dog, or Unknown using a PyTorch model.

✅ Features

Upload an image and get predictions instantly.
Displays class probabilities for transparency.
Includes a confidence threshold (default: 0.6) to reduce misclassification.
Built with Streamlit, PyTorch, and Torchvision.


📂 Project Structure
app.py              # Streamlit app
train.py            # Model training script
best_model.pt       # Trained PyTorch model weights
requirements.txt    # Dependencies
README.md           # Project documentation
.gitignore          # Ignore unnecessary files


⚙️ Installation
Clone the repo and install dependencies:
Shellgit clone https://github.com/gardnerlingjia-blip/cat-dog-unknown-app.gitcd cat-dog-unknown-apppip install -r requirements.txtShow more lines

▶️ Run Locally
Shellstreamlit run app.pyShow more lines
Then open the URL shown in your terminal (usually http://localhost:8501).

🌐 Live Demo
https://gardnerlingjia-blip-cat-dog-unknown-app.streamlit.app
(Replace with your actual deployed link)

🛠 How It Works

Model: ResNet18 backbone fine-tuned for 3 classes.
Preprocessing: Resize → Normalize → Tensor.
Confidence threshold: If max probability < 0.6 → Unknown.


✅ Future Improvements

Add a slider in the UI to adjust confidence threshold.
Deploy to Hugging Face Spaces for wider sharing.
Add top-3 predictions display.
