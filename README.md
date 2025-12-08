# FYP---Wildfire-Monitoring-System

## Setup Instructions:

Open terminal in the folder. Then run the following commands:

````python -m venv .venv````

````.venv\Scripts\activate````

Install dependencies as well by running the following command:

````pip install -r requirements.txt````

Finally:

````uvicorn api.main:app --reload````
