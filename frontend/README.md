# AgroCheck

AgroCheck is an **end-to-end crop disease detection system** built with a modern ML & full-stack pipeline:  
- **Data preprocessing** & cleaning  
- **CNN model training (TensorFlow/Keras)**  
- **Model serving (TensorFlow Serving)**  
- **Inference API (FastAPI)**  
- **User-facing frontend (React.js)**  
- **Cloud deployment (Google Cloud Run + Artifact Registry + Cloud Storage)**  

> The goal is to provide **fast, reliable plant disease predictions** from leaf images with production-grade MLOps.

---

## 🚀 Architecture

```mermaid
flowchart LR
    A[Raw Data (PlantVillage, etc.)] --> B[Data Cleaning & Preprocessing]
    B --> C[CNN Model Training (TensorFlow/Keras)]
    C --> D[Export SavedModel / .keras]
    D --> E[TensorFlow Serving (Docker/Cloud Run)]
    E --> F[FastAPI Backend]
    F --> G[React.js Frontend]
    G --> H[GCP Deployment]
📂 Repository Structure
graphql
Copy
Edit
AgroCheck/
├─ api/                 # FastAPI app: inference endpoints & schemas
├─ frontend/            # React web app (upload + results UI)
├─ gcp/                 # GCP deployment scripts/configs
├─ models/              # Exported TF SavedModels / versioned dirs
├─ training/            # Data preprocessing & model training scripts
├─ main.py              # Entrypoint / helper for local API
├─ models.config        # TensorFlow Serving multi-model config
├─ requirements.txt     # Python dependencies
└─ README.md
🧹 Data & Preprocessing
Input: PlantVillage leaf images

Steps:

Deduplicate & clean images

Resize/crop to 224×224

Normalize pixels (0–1)

Data augmentation: rotations, flips, shifts

Train/val/test split

Example:

python
Copy
Edit
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale=1./255,
  validation_split=0.2,
  rotation_range=15,
  width_shift_range=0.1,
  height_shift_range=0.1,
  horizontal_flip=True
)
🧠 Model (CNN)
Backbone: MobileNetV2 / EfficientNet (transfer learning)

Loss: categorical cross-entropy

Metrics: accuracy, F1-score

Export for serving:

markdown
Copy
Edit
models/
  plantdisease/
    1/
      saved_model.pb
      variables/
📦 TensorFlow Serving
models.config example:

text
Copy
Edit
model_config_list: {
  config: {
    name: 'plantdisease',
    base_path: '/models/plantdisease',
    model_platform: 'tensorflow'
  }
}
Run locally:

bash
Copy
Edit
docker run --rm -p 8501:8501 \
  -v $(pwd)/models/plantdisease:/models/plantdisease \
  -v $(pwd)/models.config:/models/models.config \
  -e MODEL_CONFIG_FILE=/models/models.config \
  tensorflow/serving
⚡ FastAPI Backend
Responsibilities:

Accept image upload

Preprocess to tensor

Call TF Serving REST/gRPC

Return prediction JSON

Run locally:

bash
Copy
Edit
pip install -r requirements.txt
export TF_SERVING_URL=http://localhost:8501
uvicorn api.app:app --reload --port 8000
Response:

json
Copy
Edit
{
  "prediction": "Late_Blight",
  "confidence": 0.93,
  "top_k": [
    {"label": "Late_Blight", "prob": 0.93},
    {"label": "Early_Blight", "prob": 0.05},
    {"label": "Healthy", "prob": 0.02}
  ]
}
💻 React Frontend
Upload & preview images

Display prediction + confidence

Simple UI for farmers & experts

Run locally:

bash
Copy
Edit
cd frontend
npm install
npm run dev
🌐 Deployment on Google Cloud
1. Build & Push Images
bash
Copy
Edit
gcloud auth login
gcloud config set project <PROJECT_ID>

# FastAPI API
gcloud builds submit --tag \
  us-central1-docker.pkg.dev/<PROJECT_ID>/agrocheck/fastapi:latest .

# TensorFlow Serving
gcloud builds submit --tag \
  us-central1-docker.pkg.dev/<PROJECT_ID>/agrocheck/tfserving:latest serving/
2. Deploy to Cloud Run
bash
Copy
Edit
gcloud run deploy agrocheck-api \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/agrocheck/fastapi:latest \
  --platform managed --region us-central1 --allow-unauthenticated \
  --set-env-vars TF_SERVING_URL=<TF_SERVING_URL>,MODEL_NAME=plantdisease
Frontend can be deployed via Firebase Hosting or Cloud Storage + CDN.

🔑 Environment Variables
Backend:

TF_SERVING_URL – URL of TensorFlow Serving

MODEL_NAME – e.g., plantdisease

TOP_K – top predictions to return

Frontend:

VITE_API_URL – Base URL for FastAPI

✅ API Reference
GET /healthz → Health check

POST /predict → Image upload, returns predictions

🛠 Roadmap
 Add evaluation metrics & confusion matrix

 CI/CD with GitHub Actions

 Canary deployment for model versions

 Grad-CAM explanations in frontend