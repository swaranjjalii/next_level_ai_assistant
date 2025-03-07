# AI Assistant

This project is an AI assistant that helps with various tasks, leveraging large language models and vector-based information retrieval.

## Features

- **Text Summarization:** Summarizes input text using the Gemini Pro model.
- **Sentiment Analysis:** Classifies the sentiment of input text (positive, negative, or neutral).
- **Vector Retrieval:** Implements semantic search using sentence embeddings.
- **Data Processing:** Supports loading and saving data in CSV, JSON, and plain text formats.
- **Interactive UI:** Provides a Streamlit-based user interface for interacting with the assistant.
- **Named Entity Recognition:** Extracts entities like people, organizations, locations, and dates.
- **Question Answering:** Uses context-aware responses with RAG (Retrieval-Augmented Generation).
- **Code Generation and Review:** Creates and analyzes Python code based on requirements.

## Requirements

- Python 3.6+
- [Poetry](https://python-poetry.org/) for dependency management (or pip)
- A Google Cloud project with the Gemini API enabled
- An API key for the Gemini API (set as an environment variable `GEMINI_API_KEY`)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/akshayram1/next_level_ai_assistant.git
    cd ai_assistant
    ```

2.  **Install dependencies using Poetry:**

    ```bash
    poetry install
    ```

    Or, if you prefer pip:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set the Gemini API key:**

    Create a `.env` file in the project root with the following content:

    ```properties
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    Replace `YOUR_GEMINI_API_KEY` with your actual API key.

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run src/app.py
    ```

2.  **Interact with the AI Assistant:**

    Open your web browser and go to the address displayed in the terminal (usually `http://localhost:8501`).  Enter your query in the text box and the assistant will process it.

3.  **Working with documents:**

    The assistant can ingest documents through the UI for knowledge retrieval. Upload files in formats like TXT, CSV, JSON, or MD through the sidebar, then click "Process Documents" to add them to the vector index.

4.  **Managing the knowledge base:**

    You can save the current index to disk using the "Save Index" button and reload it later with "Load Index" to persist your document collection.

5.  **Watch the demo:**

    Check out the [demo video]([(https://github.com/swaranjjalii/next_level_ai_assistant/blob/main/demo.mp4)]) to see the AI assistant in action.

## Project Structure

```
ai_assistant/
├── src/
│   ├── app.py             # Streamlit application
│   ├── nlp_tasks.py       # NLP tasks (summarization, sentiment analysis, etc.)
│   ├── retrieval.py       # Vector-based information retrieval
│   ├── llm_abstraction.py # Abstraction layer for LLMs (Gemini)
│   ├── data_processing.py # Data loading and saving utilities
├── docs/                  # Documentation and sample files
├── .env                   # Environment variables (API keys)
├── requirements.txt       # Pip dependencies
├── index.json             # Optional saved vector index
├── README.md              # This file
```

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.


# next_level_ai_assistant
