# ❤️ CardioCare Hybrid Predictor Engine

An enterprise-grade, microservices-driven clinical web application that predicts cardiovascular disease risk. The platform uses a hybrid decision framework combining an ensemble Machine Learning pipeline with a deterministic clinical scoring engine, logged into a permanent NoSQL repository, and supplemented by an automated generative AI layer.

## 🚀 Microservices Architecture
The system is fully decoupled into isolated, communication-ready microservices orchestrated via container environments:
* **Frontend UI:** A responsive data dashboard built with **Streamlit**.
* **Backend Engine:** A high-performance, asynchronous REST API powered by **FastAPI**.
* **Database Tier:** A cloud-hosted **MongoDB Atlas** NoSQL cluster for permanent patient log persistence.
* **LLM Intelligence Layer:** Fully integrated with Google's **Gemini 2.5 Flash** API for automated, contextual patient lifestyle advice generation.

---

## 📊 Core Technical Framework

### 1. The Hybrid Decision Pipeline
* **Machine Learning Model:** A regularized **Random Forest Classifier** trained on 68,000+ patient records, handling advanced multivariate data imputation (`IterativeImputer`) and biological validation boundaries. Outperforming traditional linear benchmarks, it achieves a ~72.3% testing accuracy with a tight generalization gap (<2%).
* **Clinical Scoring Engine:** An algorithmic implementation modeling peer-reviewed medical standards (AHA/ACC ASCVD Risk Criteria). It applies strict mathematical weights to behavioral indicators (Smoking Status, Physical Activity Level) to act as a deterministic safety guard against model false-positives.

### 2. Live History & Clinical Analytics Dashboard
* Asynchronous tracking pipelines feed telemetry payloads dynamically to MongoDB Atlas.
* An interactive logs visualizer back-parses unstructured BSON structures on-the-fly, giving healthcare provider views absolute sorting control and quick-KPI calculations.

---

## 🛠️ Advanced Tech Stack
* **Languages & Frameworks:** Python 3.12, FastAPI, Streamlit
* **Data Science Pipeline:** Scikit-Learn, Pandas, NumPy, Joblib
* **Database & Middleware:** PyMongo, Pydantic (Strict Data Type Guarding)
* **Infrastructure & Orchestration:** Docker, Docker Compose (With live hot-reloading volume bindings)
* **Generative AI:** Google GenAI SDK (Gemini-2.5-Flash Engine)

---

## 📦 Containerized Installation & Local Setup

The entire multi-container environment spins up with a single orchestrator instruction.

### Prerequisites
Ensure you have **Docker Desktop** installed and running on your host machine.

### Setup Instructions
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/minshac13/ML-project.git](https://github.com/minshac13/ML-project.git)
    cd ML-project
    ```

2.  **Configure Environment Environment Flags:**
    Create a `.env` file in the root folder directory and inject your remote cluster addresses and keys:
    ```env
    GEMINI_API_KEY=your_actual_gemini_api_key_here
    MONGO_URI=your_mongodb_atlas_connection_string_here
    ```

3.  **Boot the Application Network:**
    Build the underlying container images and bring up the microservice nodes simultaneously:
    ```bash
    docker compose up --build
    ```

4.  **Access the Ports:**
    * **Interactive Frontend UI:** Open `http://localhost:8501`
    * **FastAPI Engine Interactive Swagger Documentation:** Open `http://localhost:8000/docs`

5.  **Hot Reloading Development Mode:**
    Because local host drives are hot-linked directly to internal container directory mounts via Docker Compose volumes, modifying code structures inside your text editor will re-trigger server compilation arrays automatically without needing a container rebuild.