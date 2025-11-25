# Tool Augmented Chatbot

A powerful Streamlit-based chatbot application that combines Google's Gemini AI with document retrieval capabilities using FAISS vector search. The application allows users to upload PDF documents and ask questions about their content using advanced semantic search and retrieval techniques.

## Features

- 🤖 **AI-Powered Chat**: Powered by Google Gemini 2.5 Pro with tool augmentation
- 📄 **PDF Document Processing**: Upload and process PDF documents with automatic text extraction
- 🔍 **FAISS Vector Search**: Fast semantic search through uploaded documents using FAISS indexes
- 🔄 **Dual Chat Mode**: Compare FAISS vector search vs direct document upload side-by-side
- 📊 **Token Usage Tracking**: Monitor and compare token usage between different retrieval methods
- 🌐 **Web Search Integration**: DuckDuckGo search tool for real-time information
- 💬 **Multiple Chat Tabs**: Organize different conversations in separate tabs
- ⚡ **Optimized Processing**: Parallel PDF processing for faster document handling
- 🎯 **Re-ranking**: Cross-encoder re-ranking for improved search relevance

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google API Key (for Gemini AI)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tool-augmented-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   Add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   
   **How to get a Google API Key:**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key
   - Copy the key and paste it in your `.env` file

## Usage

1. **Start the application**
   ```bash
   streamlit run main.py
   ```

2. **Upload PDFs**: Click the file uploader to upload one or more PDF documents

3. **Ask Questions**: Type your question in the chat input and press Enter

4. **Multiple Tabs**: Use the ➕ button to create multiple chat tabs for different conversations

5. **Dual Chat Mode**: Toggle "Enable Dual Chat Mode" in the sidebar to compare FAISS vs Direct document processing side-by-side

## Project Structure

```
tool-augmented-chatbot/
├── main.py                 # Main Streamlit application entry point
├── config.py              # Configuration and session state management
├── llm_setup.py           # LLM, graph, and tools setup
├── document_handling.py   # PDF processing and FAISS operations
├── chat_handlers.py       # Chat conversation handlers
├── ui_components.py       # UI components and token tracking display
├── token_tracking.py      # Token usage tracking utilities
├── requirements.txt       # Python dependencies
├── SETUP.md              # Detailed setup instructions
└── .env                  # Environment variables (create this, not in repo)
```

## Technical Details

### Document Processing
- PDFs are processed and split into chunks for vector embedding
- FAISS indexes are created for fast similarity search
- Documents are stored in both vectorized (FAISS) and direct text formats

### Retrieval Methods
- **FAISS Search**: Uses semantic similarity search with embeddings
- **Direct Upload**: Processes entire document text directly
- **Multi-Query Retrieval**: Generates query variations for better results
- **Re-ranking**: Uses cross-encoder models to improve relevance

### Tools Available
- `document_retrieval`: Retrieve information from uploaded PDFs
- `duckduckgo_search`: Search the web for real-time information
- `add`: Mathematical addition tool
- `multiply`: Mathematical multiplication tool

## Troubleshooting

### Port Already in Use
If port 8501 is already in use, specify a different port:
```bash
streamlit run main.py --server.port 8502
```

### Missing Dependencies
If you encounter import errors:
```bash
pip install -r requirements.txt --upgrade
```

### API Key Issues
Make sure your `.env` file is in the project root and contains:
```
GOOGLE_API_KEY=your_actual_key_here
```

The application will show an error if the API key is missing or invalid.

## License

This project is open source and available for use and modification.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
