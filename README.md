**# Customer Support AI**



**An AI-powered customer support system that classifies support tickets, predicts their priority, analyzes sentiment, searches a knowledge base, and generates helpful responses.**



**## Features**



**- \*\*Ticket Classification:\*\* Predicts the category of a support ticket.**

**- \*\*Priority Prediction:\*\* Classifies tickets by priority.**

**- \*\*Sentiment Analysis:\*\* Detects positive or negative customer sentiment.**

**- \*\*Semantic Knowledge Search:\*\* Retrieves relevant troubleshooting articles.**

**- \*\*AI Response Generation:\*\* Uses the Gemini API to draft support responses based on relevant knowledge-base articles.**

**- \*\*Ticket Management:\*\* Stores and retrieves tickets using SQLite.**

**- \*\*Interactive Dashboard:\*\* Provides a Streamlit interface for working with the system.**

**- \*\*Docker Support:\*\* Runs the API and dashboard in containers.**



**## Tech Stack**



**- Python**

**- FastAPI**

**- Streamlit**

**- Scikit-learn**

**- Hugging Face Transformers**

**- Sentence Transformers**

**- Google Gemini API**

**- SQLite**

**- Docker and Docker Compose**



**## Project Structure**



**```text**

**customer-support-ai/**

**├── app/**

**│   ├── main.py**

**│   ├── database.py**

**│   ├── schemas.py**

**│   ├── sentiment.py**

**│   ├── knowledge\_base.py**

**│   └── response\_generator.py**

**├── data/**

**│   ├── knowledge\_base.csv**

**│   ├── tickets.csv**

**│   └── tickets.db**

**├── models/**

**│   ├── ticket\_category\_pipeline.joblib**

**│   └── priority\_pipeline.joblib**

**├── dashboard.py**

**├── train\_model.py**

**├── train\_priority\_model.py**

**├── requirements.txt**

**├── Dockerfile**

**├── docker-compose.yml**

**└── README.md**

**```**



**## Run with Docker**



**### Prerequisites**



**- Docker Desktop**

**- A Gemini API key for AI-generated responses**



**### 1. Configure the environment**



**Create a `.env` file in the project root:**



**```text**

**GEMINI\_API\_KEY=your\_gemini\_api\_key\_here**

**```**



**Replace the placeholder with your own API key. Never commit this file or publish your API key.**



**### 2. Build and start the application**



**```powershell**

**docker compose up --build -d**

**```**



**### 3. Open the application**



**- \*\*Streamlit Dashboard:\*\* http://localhost:8501**

**- \*\*FastAPI:\*\* http://localhost:8000**

**- \*\*API Documentation:\*\* http://localhost:8000/docs**



**### 4. View logs**



**```powershell**

**docker compose logs -f**

**```**



**### 5. Stop the application**



**```powershell**

**docker compose down**

**```**



**## API Endpoints**



**| Method | Endpoint | Purpose |**

**|---|---|---|**

**| POST | `/tickets/predict` | Predict ticket category, priority, and sentiment |**

**| POST | `/tickets` | Create and save a support ticket |**

**| GET | `/tickets` | Retrieve saved tickets |**

**| GET | `/knowledge/search` | Search relevant knowledge-base articles |**

**| GET | `/knowledge/response` | Generate a draft support response |**



**Use the FastAPI documentation page to inspect request parameters and test endpoints.**



**## Security**



**- Keep API keys in environment variables.**

**- Do not commit `.env` or other files containing secrets.**

**- Avoid publishing real customer information or sensitive ticket data.**



**## Future Improvements**



**- Add automatic retry and fallback handling for temporary AI API failures.**

**- Improve ticket analytics and reporting.**

**- Add user authentication and role-based access.**

**- Add automated tests and CI/CD deployment.**



**## Disclaimer**



**AI-generated responses should be reviewed before being sent to customers. Predictions may be imperfect and should be validated against real-world support requirements.**

