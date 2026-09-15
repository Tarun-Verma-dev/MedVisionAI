# MedVision AI

**A Modular, Explainable Deep Learning Framework for Multi-Modal Medical Image Classification**

MedVision AI is a research-oriented platform that routes medical images (X-ray, MRI, retina, skin, etc.) to specialized deep learning models for disease classification, and pairs every prediction with a visual explanation (heatmap) showing *why* the model made that decision.

Instead of building one monolithic model, MedVision AI is designed as a **platform**: a modality router dispatches each uploaded image to a dedicated model trained for that specific imaging type, and an Explainable AI (XAI) engine generates interpretability overlays for every prediction.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Supported Modalities](#supported-modalities)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Backend](#backend)
  - [Backend Setup](#backend-setup)
  - [API Endpoints](#api-endpoints)
  - [Model Pipeline](#model-pipeline)
  - [Explainability Engine](#explainability-engine)
- [Frontend](#frontend)
  - [Frontend Setup](#frontend-setup)
  - [Frontend Pages](#frontend-pages)
- [Datasets](#datasets)
- [Installation (Full Stack)](#installation-full-stack)
- [Environment Variables](#environment-variables)
- [Research Objective](#research-objective)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Overview

Medical imaging AI research is often narrow — one dataset, one modality, one model. MedVision AI proposes a **modular architecture** where:

1. A single entry point accepts any medical image.
2. A **modality detector** identifies the image type (X-ray, MRI, retina, skin, etc.).
3. The image is routed to a **specialized classifier** trained specifically for that modality.
4. Every prediction is passed through an **XAI engine** (Grad-CAM and comparable methods) to generate a visual explanation.
5. The user receives a structured report: prediction, confidence score, and heatmap overlay.

The core research question this project investigates is not just *"how accurate is the model?"* but:

> **Does the model's explanation correspond to medically relevant regions of the image?**

---

## System Architecture

```
                         ┌─────────────────┐
                         │  Medical Image  │
                         │     Upload      │
                         └────────┬────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │ Image Modality Detector │
                    └────────────┬────────────┘
                                 ↓
             ┌───────────────────┼──────────────────┐
             ↓                   ↓                  ↓
          X-RAY                 MRI               RETINA / SKIN
             ↓                   ↓                  ↓
       Chest X-Ray          Brain MRI          Retina / Skin
          Model                Model             Model
             ↓                   ↓                  ↓
       Disease Class        Tumor Class        DR Grade / Lesion
             └───────────────────┼──────────────────┘
                                 ↓
                         ┌──────────────┐
                         │ XAI Engine   │
                         │ Grad-CAM etc │
                         └──────┬───────┘
                                ↓
                       Prediction + Heatmap
                                ↓
                         Explainable Report
```

---

## Key Features

- 🧠 **Multi-modal support** — chest X-ray, brain MRI, retina, and skin lesion classification in one platform
- 🔀 **Automatic modality routing** — no need for the user to manually select the model
- 🔍 **Built-in explainability** — Grad-CAM, Grad-CAM++, Integrated Gradients, and SHAP support
- 📊 **Confidence-scored predictions** — every output includes class probabilities, not just a single label
- 🧩 **Modular model design** — each modality's model can be trained, evaluated, and swapped independently
- ⚡ **REST API backend** — FastAPI service that can be consumed by any frontend or third-party client
- 🖥️ **Modern web frontend** — React/Next.js interface for upload, visualization, and report generation

---

## Supported Modalities

| Module | Dataset | Task | Model (proposed) |
|---|---|---|---|
| Chest X-ray | ChestX-ray14 / CheXpert | Thoracic disease (multi-label, 14 classes) | DenseNet121 |
| Brain MRI | Brain Tumor MRI Dataset | Tumor classification (glioma, meningioma, pituitary, none) | EfficientNet / ResNet |
| Retinal Image | EyePACS (or similar) | Diabetic retinopathy grading (none → severe) | EfficientNet |
| Skin Image | ISIC | Skin lesion classification | EfficientNet / ResNet |

> Future additions: CT scans, ultrasound, dental X-ray.

---

## Tech Stack

### Backend
- **Language:** Python 3.10+
- **API Framework:** FastAPI
- **Deep Learning:** PyTorch, torchvision
- **Explainability:** Grad-CAM / Captum
- **Image Processing:** OpenCV, NumPy
- **Classical ML utilities:** scikit-learn
- **Database:** PostgreSQL (production) / SQLite (development)
- **Server:** Uvicorn / Gunicorn

### Frontend
- **Framework:** React (Next.js)
- **Styling:** Tailwind CSS (or CSS Modules)
- **HTTP Client:** Axios / Fetch API
- **State Management:** React Context or Zustand
- **Visualization:** Canvas/SVG overlay for heatmaps

### Prototyping (optional, early-stage)
- **Streamlit** — for a fast, single-file demo before the full frontend is built

---

## Project Structure

```
medvision-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── predict.py       # /predict endpoint
│   │   │   │   ├── explain.py       # /explain endpoint
│   │   │   │   └── health.py        # /health endpoint
│   │   │   └── deps.py              # shared dependencies
│   │   ├── core/
│   │   │   ├── config.py            # settings/env config
│   │   │   └── logging.py
│   │   ├── models/
│   │   │   ├── modality_router.py   # detects image type
│   │   │   ├── xray_model.py        # chest X-ray classifier
│   │   │   ├── mri_model.py         # brain MRI classifier
│   │   │   ├── retina_model.py      # retina classifier
│   │   │   └── skin_model.py        # skin lesion classifier
│   │   ├── xai/
│   │   │   ├── gradcam.py
│   │   │   ├── gradcam_plus.py
│   │   │   ├── integrated_gradients.py
│   │   │   └── shap_explainer.py
│   │   ├── schemas/
│   │   │   └── prediction.py        # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── preprocessing.py
│   │   │   └── report_generator.py
│   │   └── db/
│   │       ├── database.py
│   │       └── models.py            # SQLAlchemy ORM models
│   ├── weights/                     # trained model checkpoints (.pt)
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── app/                     # Next.js app router pages
│   │   │   ├── page.tsx             # landing / upload page
│   │   │   ├── results/page.tsx     # prediction + heatmap view
│   │   │   └── history/page.tsx     # past reports
│   │   ├── components/
│   │   │   ├── UploadCard.tsx
│   │   │   ├── HeatmapViewer.tsx
│   │   │   ├── ConfidenceChart.tsx
│   │   │   └── ReportCard.tsx
│   │   ├── lib/
│   │   │   └── api.ts               # API client
│   │   ├── styles/
│   │   └── types/
│   ├── package.json
│   └── next.config.js
│
├── notebooks/                       # training/experimentation notebooks
│   ├── chest_xray_training.ipynb
│   ├── brain_mri_training.ipynb
│   ├── retina_training.ipynb
│   └── skin_training.ipynb
│
├── docs/
│   ├── architecture.md
│   └── research_notes.md
│
├── .env.example
├── docker-compose.yml
├── LICENSE
└── README.md
```

---

## Backend

The backend is a **FastAPI** service responsible for image intake, modality detection, model inference, and generating explainability overlays.

### Backend Setup

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env

# 5. Run the development server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Upload an image, returns modality, predicted class(es), and confidence scores |
| `POST` | `/api/v1/explain` | Returns a Grad-CAM (or selected method) heatmap overlay for a given prediction |
| `GET` | `/api/v1/models` | Lists available models and their supported classes |
| `GET` | `/api/v1/reports/{id}` | Retrieves a previously generated report |
| `GET` | `/health` | Health check endpoint |

**Example request:**

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@chest_xray.png"
```

**Example response:**

```json
{
  "modality": "chest_xray",
  "predictions": [
    { "label": "Pneumonia", "confidence": 0.874 },
    { "label": "Effusion", "confidence": 0.122 },
    { "label": "Cardiomegaly", "confidence": 0.041 }
  ],
  "explanation_url": "/api/v1/explain/8f2c1a"
}
```

### Model Pipeline

Each modality follows the same general pipeline pattern:

```
Image → Preprocessing (resize/normalize) → CNN Backbone → Class Probabilities
```

| Modality | Backbone | Output |
|---|---|---|
| Chest X-ray | DenseNet121 | 14-class multi-label probabilities |
| Brain MRI | EfficientNet / ResNet | Tumor type (4-class) |
| Retina | EfficientNet | DR severity grade (5-class) |
| Skin | EfficientNet / ResNet | Lesion type |

The **modality router** runs first, using a lightweight classifier to determine which specialized model should process the image. This can start as a simple CNN classifier and later be upgraded to a vision-language/foundation model for more robust detection.

### Explainability Engine

For every prediction, the backend generates a visual explanation using one or more of:

1. **Grad-CAM** — baseline, fast, class-discriminative localization
2. **Grad-CAM++** — improved localization for multiple instances of a class
3. **Integrated Gradients** — pixel-level attribution via path integrals
4. **SHAP** — game-theoretic feature attribution for comparison

The output is a heatmap overlay blended with the original image, plus a raw attribution map that can be used for quantitative evaluation (e.g., overlap with radiologist-annotated regions of interest).

---

## Frontend

The frontend is a **React/Next.js** application that allows users to upload medical images and view predictions with interactive heatmap overlays.

### Frontend Setup

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Set environment variables
cp .env.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000

# 4. Run the development server
npm run dev
```

The app will be available at `http://localhost:3000`.

### Frontend Pages

| Page | Description |
|---|---|
| `/` | Upload interface — drag-and-drop image upload with modality auto-detection preview |
| `/results` | Displays predicted class(es), confidence scores, and the Grad-CAM heatmap overlay |
| `/history` | List of previously analyzed images and their generated reports |

**Core components:**
- `UploadCard` — handles file selection/drag-drop and sends the image to `/api/v1/predict`
- `HeatmapViewer` — renders the original image with the heatmap overlay, with an opacity slider
- `ConfidenceChart` — bar chart of class probabilities
- `ReportCard` — combines prediction, heatmap, and metadata into a downloadable report

> An optional **Streamlit** app (`streamlit_app.py`) can be used as a lightweight prototype/demo before the full Next.js frontend is built.

---

## Datasets

| Dataset | Modality | Size (approx.) | Link |
|---|---|---|---|
| ChestX-ray14 / CheXpert | Chest X-ray | ~112,000 images, 14 labels | NIH / Stanford ML Group |
| Brain Tumor MRI Dataset | Brain MRI | Varies by source | Kaggle |
| EyePACS | Retina | ~35,000+ images | Kaggle / EyePACS |
| ISIC Archive | Skin | 25,000+ images | ISIC Archive |

> Datasets are not included in this repository due to size and licensing. Download links and preprocessing scripts are provided in `notebooks/`.

---

## Installation (Full Stack)

Using Docker Compose to run both services together:

```bash
# From the project root
docker-compose up --build
```

This will start:
- `backend` on `http://localhost:8000`
- `frontend` on `http://localhost:3000`
- `db` (PostgreSQL) on `localhost:5432`

---

## Environment Variables

**Backend (`backend/.env`):**

```env
ENVIRONMENT=development
DATABASE_URL=postgresql://user:password@localhost:5432/medvision
MODEL_WEIGHTS_DIR=./weights
ALLOWED_ORIGINS=http://localhost:3000
```

**Frontend (`frontend/.env.local`):**

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Research Objective

Beyond building a working classification system, this project is framed as a research contribution around **interpretability in medical AI**. The central research question:

> Does the model's saliency/attribution map correspond to medically relevant regions, as judged against expert-annotated ground truth?

Planned evaluation approaches:
- Quantitative overlap metrics (IoU, pointing game accuracy) between heatmaps and annotated regions
- Comparative analysis across Grad-CAM, Grad-CAM++, Integrated Gradients, and SHAP
- Qualitative review with domain-informed criteria for clinically plausible explanations

---

## Roadmap

- [ ] Phase 1: Implement and validate individual modality models (X-ray, MRI, retina, skin)
- [ ] Phase 2: Build and evaluate the modality router
- [ ] Phase 3: Integrate Grad-CAM as the baseline XAI method
- [ ] Phase 4: Add Grad-CAM++, Integrated Gradients, and SHAP; compare explanation quality
- [ ] Phase 5: Build the FastAPI backend and expose REST endpoints
- [ ] Phase 6: Build the React/Next.js frontend
- [ ] Phase 7: Add CT, ultrasound, and dental X-ray support
- [ ] Phase 8: Explore foundation/vision-language models for modality detection

---

## Disclaimer

MedVision AI is a **research and educational project**. It is **not a certified medical device** and is **not intended for clinical diagnosis or treatment decisions**. All predictions should be reviewed by qualified medical professionals.

---

## License

This project is released under the [MIT License](LICENSE).
