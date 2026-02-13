🐶🐱 Pet Image Classifier (Cat · Dog · Unknown)
A production-ready AI web application that classifies uploaded images as Cat, Dog, or Unknown using a fine-tuned ResNet18 model.
The app is fully containerized with Docker and deployed on Google Cloud Run.

🚀 Live Demo
🔗 Cloud Run Deployment:
https://pet-app-891324924713.europe-west1.run.app

🎯 Project Highlights
Real-time image classification via Streamlit UI
Confidence-based thresholding to reduce misclassification
Top-class prediction + full probability distribution
Dockerized for reproducibility
Deployed to Google Cloud Run
Versioned deployment workflow via scripted CI-like process

🧠 Model Details
Backbone: ResNet18
Output Classes:
Cat
Dog
Unknown (non-pet / low confidence fallback)
Preprocessing:
Resize to 224×224
Normalize using ImageNet statistics
Softmax probabilities for interpretability
Confidence threshold: 0.6

🏗 Architecture
Local Development
→ Docker Container
→ Google Cloud Build
→ Artifact Registry
→ Cloud Run Service
→ Public HTTPS Endpoint
This architecture enables:
Reproducibility
Scalability
Cloud-native deployment
Version-controlled releases

📂 Project Structure
app.py              # Streamlit inference app
train.py            # Model training script
best_model.pt       # Trained model weights
deploy.sh           # Versioned Cloud Run deployment script
Dockerfile          # Container configuration
requirements.txt    # Python dependencies
README.md

⚙️ Run Locally
pip install -r requirements.txt
streamlit run app.py

🐳 Build & Deploy (Cloud Run)
./deploy.sh 4
This:
Builds a Docker image
Pushes to Artifact Registry
Deploys a new Cloud Run revision

🛠 Key Engineering Decisions
CPU-based inference for cost-efficient serverless deployment
Cached model loading via st.cache_resource
Threshold-based unknown class to improve robustness
Versioned image tagging (:1, :2, :3, …)

🔮 Future Improvements
Move model weights to Google Cloud Storage
Add structured logging + monitoring
Add performance metrics dashboard
Add CI/CD via GitHub Actions
Extend to multi-class pet recognition



