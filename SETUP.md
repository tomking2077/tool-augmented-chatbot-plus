# Setup and Run Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

### 1. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root directory with your API key:

```bash
# Create .env file
touch .env
```

Add the following to your `.env` file:

```
GOOGLE_API_KEY=your_google_api_key_here
```

**How to get a Google API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and paste it in your `.env` file

### 3. Run the Application

Start the Streamlit application:

```bash
streamlit run main.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## Usage

1. **Upload PDFs**: Click the file uploader to upload one or more PDF documents
2. **Ask Questions**: Type your question in the chat input and press Enter
3. **Multiple Tabs**: Use the ➕ button to create multiple chat tabs
4. **Dual Chat Mode**: Toggle "Enable Dual Chat Mode" in the sidebar to compare FAISS vs Direct document processing

## Features

- **FAISS Vector Search**: Fast semantic search through uploaded PDFs
- **Multi-Query Retrieval**: Enhanced search with query variations
- **Re-ranking**: Cross-encoder re-ranking for better relevance
- **Optimized Processing**: 3x faster PDF processing with parallelization
- **Multiple Chat Tabs**: Organize different conversations

## Troubleshooting

### Port Already in Use

If port 8501 is already in use, specify a different port:

```bash
streamlit run main.py --server.port 8502
```

### Missing Dependencies

If you encounter import errors, make sure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

### API Key Issues

Make sure your `.env` file is in the project root and contains:
```
GOOGLE_API_KEY=your_actual_key_here
```

The application will show an error if the API key is missing or invalid.

## Project Structure

```
tool-augmented-chatbot/
├── main.py                 # Main Streamlit application
├── config.py              # Configuration and session state
├── llm_setup.py            # LLM and embedding setup
├── document_handling.py   # PDF processing and FAISS operations
├── chat_handlers.py       # Chat conversation handlers
├── ui_components.py       # UI components
├── token_tracking.py      # Token usage tracking
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (create this)
```

