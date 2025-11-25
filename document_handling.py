"""PDF document handling and FAISS operations"""
import os
import json
import hashlib
import multiprocessing
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# embedding_fn will be imported from llm_setup
from llm_setup import embedding_fn

# Persistent metadata file path
METADATA_FILE = "file_metadata.json"

# Thread pool size configuration
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)
PDF_PARSE_WORKERS = min(multiprocessing.cpu_count(), 6)


def _load_persistent_metadata() -> dict:
    """Load file metadata from persistent storage"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_persistent_metadata(metadata: dict):
    """Save file metadata to persistent storage"""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except IOError:
        st.warning("Could not save file metadata to disk")


def _get_file_metadata(metadata_key: str) -> dict:
    """Get file metadata, loading from persistent storage if needed"""
    if "file_metadata" not in st.session_state:
        st.session_state.file_metadata = _load_persistent_metadata()
    
    if metadata_key not in st.session_state.file_metadata:
        st.session_state.file_metadata[metadata_key] = {}
    
    return st.session_state.file_metadata[metadata_key]


def _update_file_metadata(metadata_key: str, metadata: dict):
    """Update file metadata and persist to disk"""
    if "file_metadata" not in st.session_state:
        st.session_state.file_metadata = {}
    
    st.session_state.file_metadata[metadata_key] = metadata
    _save_persistent_metadata(st.session_state.file_metadata)


def expand_query(query: str) -> list[str]:
    """Expand query with variations for better retrieval
    
    Returns:
        List of query variations including original and expanded versions
    """
    queries = [query]  # Always include original
    
    # Simple query variations
    # Add question form if it's a statement
    if not query.strip().endswith('?'):
        queries.append(f"{query}?")
    
    # Add "what is" prefix for definitions
    if not query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
        queries.append(f"What is {query}?")
    
    # Add "explain" variation
    if 'explain' not in query.lower():
        queries.append(f"Explain {query}")
    
    return queries[:3]  # Limit to 3 variations


def _add_unique_documents(all_results, seen_docs, documents):
    """Add unique documents to results list"""
    for doc in documents:
        doc_id = hash(doc.page_content)
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            all_results.append(doc)


def multi_query_retrieval(db, queries: list[str], k: int = 8, fetch_k: int = 20) -> list[Document]:
    """Retrieve documents using multiple query variations and combine results"""
    all_results = []
    seen_docs = set()
    
    for query in queries:
        if len(all_results) >= k * 2:  # Early termination
            break
            
        try:
            results = db.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=0.5
            )
        except Exception:
            # Fallback to basic search
            results = db.similarity_search(query, k=k)
        
        _add_unique_documents(all_results, seen_docs, results)
    
    return all_results


# Cache reranker to avoid reloading model on each call
_reranker_cache = None

def rerank_results(query: str, documents: list[Document], top_k: int = 8) -> list[Document]:
    """Re-rank documents using cross-encoder for better relevance.
    
    Falls back to original order if re-ranking fails.
    """
    if len(documents) <= top_k:
        return documents
    
    try:
        global _reranker_cache
        if _reranker_cache is None:
            from sentence_transformers import CrossEncoder
            # Use a lightweight cross-encoder for re-ranking
            # Cache the model to avoid reloading (major latency improvement)
            _reranker_cache = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        reranker = _reranker_cache
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [doc for doc, score in scored_docs[:top_k]]
    except Exception:
        # If re-ranking fails, return original order
        return documents[:top_k]


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA256 hash of file content for metadata tracking"""
    return hashlib.sha256(file_content).hexdigest()


def _calculate_chunk_sizes(total_size: int) -> tuple[int, int]:
    """Calculate optimal chunk size and overlap based on total file size.
    
    Args:
        total_size: Total size of all files in bytes
        
    Returns:
        Tuple of (chunk_size, chunk_overlap)
    """
    # Simplified standard chunk sizes
    return 1000, 200


def _format_documents_with_metadata(documents: list[Document]) -> str:
    """Format documents with metadata for display"""
    chunk_texts = []
    for doc in documents:
        text = doc.page_content
        metadata = getattr(doc, 'metadata', {})
        if metadata:
            metadata_parts = []
            if 'page' in metadata:
                metadata_parts.append(f"Page {metadata['page']}")
            if 'source' in metadata:
                metadata_parts.append(f"Source: {metadata['source']}")
            
            if metadata_parts:
                text = f"[{' | '.join(metadata_parts)}]\n{text}"
        chunk_texts.append(text)
    return "\n\n".join(chunk_texts)


def _update_metadata_and_notify(results, file_metadata, conversation, tab_name, metadata_key):
    """Update file metadata and show notifications"""
    for result in results:
        file_name = result['file_name']
        file_hash = result['file_hash']
        file_metadata[file_name] = file_hash
        
        if result['type'] == 'faiss':
            # Only store index hash if it was computed (skipped for large files)
            if result.get('index_hash'):
                file_metadata[f"{file_name}_index_hash"] = result['index_hash']
            st.toast(f"📚 Indexed {file_name} with FAISS in {tab_name}.", icon="✅")
        else:
            conversation.setdefault("direct_docs", []).append(
                f"[{file_name}]\n{result['text_blob']}"
            )
            st.toast(f"📄 Attached {file_name} directly to {tab_name}.", icon="✅")
    
    st.session_state.file_metadata[metadata_key] = file_metadata


def compute_index_hash(index_path: str) -> str:
    """Compute combined hash of FAISS index files to verify integrity"""
    try:
        faiss_file = f"{index_path}/index.faiss"
        pkl_file = f"{index_path}/index.pkl"
        
        if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
            return None
        
        # Compute hash of both files
        combined_hash = hashlib.sha256()
        for file_path in [faiss_file, pkl_file]:
            with open(file_path, "rb") as f:
                combined_hash.update(f.read())
        return combined_hash.hexdigest()
    except Exception:
        return None


def _parse_pdf_file(file_name: str, content: bytes, splitter: RecursiveCharacterTextSplitter, tab_name: str):
    """Parse a single PDF file and return chunks.
    
    Args:
        file_name: Name of the file
        content: File content as bytes
        splitter: Text splitter to use
        tab_name: Tab name for temp file prefix
        
    Returns:
        dict with 'file_name', 'chunks' (on success), or 'error' (on failure), and 'success' bool
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', prefix=f'temp_{tab_name}_') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            chunks = loader.load_and_split(splitter)
            return {'file_name': file_name, 'chunks': chunks, 'success': True}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        return {'file_name': file_name, 'error': str(e), 'success': False}


def _parse_pdf_files_parallel(uploaded_files, file_contents: dict, splitter: RecursiveCharacterTextSplitter, 
                               tab_name: str, status_text=None):
    """Parse multiple PDF files in parallel.
    
    Args:
        uploaded_files: List of uploaded file objects
        file_contents: Dict mapping file names to file content bytes
        splitter: Text splitter to use
        tab_name: Tab name for temp file prefix
        status_text: Optional Streamlit status text element to update
        
    Returns:
        List of parsed file results (dicts with 'file_name' and 'chunks')
    """
    if status_text:
        status_text.text("📄 Parsing PDFs...")
    
    parsed_files = []
    with ThreadPoolExecutor(max_workers=PDF_PARSE_WORKERS) as executor:
        parse_futures = {
            executor.submit(_parse_pdf_file, uploaded_file.name, file_contents[uploaded_file.name], splitter, tab_name): uploaded_file.name
            for uploaded_file in uploaded_files
        }
        
        for future in as_completed(parse_futures):
            result = future.result()
            if result.get('success'):
                parsed_files.append(result)
            else:
                st.error(f"Error parsing {result['file_name']}: {result.get('error', 'Unknown error')}")
    
    return parsed_files


def load_existing_faiss_index(tab_name: str, file_name: str, expected_index_hash: str = None):
    """Load existing FAISS index from disk if it exists and verify integrity.
    
    Uses shared location based on filename only (faiss_indexes/{file_name}).
    
    Returns:
        tuple: (FAISS index or None, current index hash or None)
    """
    # Use shared location: filename only (works for both single and dual chat modes)
    index_path = f"faiss_indexes/{file_name}"
    if not (os.path.exists(f"{index_path}/index.faiss") and os.path.exists(f"{index_path}/index.pkl")):
        return None, None
    
    # Try to load the index first - if it loads successfully, use it
    try:
        index = FAISS.load_local(index_path, embedding_fn, allow_dangerous_deserialization=True)
        
        # Fix docstore if it's a dict (from old manual construction)
        if hasattr(index, 'docstore') and isinstance(index.docstore, dict):
            from langchain_community.docstore.in_memory import InMemoryDocstore
            index.docstore = InMemoryDocstore(index.docstore)
        
        # Compute current index hash after successful load
        current_hash = compute_index_hash(index_path)
        
        # If expected hash provided and doesn't match, warn but still use the index
        # (Hash mismatch could be due to re-processing with different params, but index is still valid)
        if expected_index_hash and current_hash and current_hash != expected_index_hash:
            st.info(f"ℹ️ FAISS index for {file_name} was updated. Using existing index.")
        
        return index, current_hash
    except Exception as e:
        # If loading fails, compute hash for diagnostic purposes
        current_hash = compute_index_hash(index_path)
        st.warning(f"Failed to load existing index for {file_name}: {str(e)}")
        return None, current_hash


def save_direct_text_blob(file_name: str, text_blob: str):
    """Save direct mode text blob to disk.
    
    Uses shared location: direct_texts/{file_name}.txt
    """
    os.makedirs("direct_texts", exist_ok=True)
    text_path = f"direct_texts/{file_name}.txt"
    try:
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_blob)
    except Exception as e:
        st.warning(f"Could not save direct text blob for {file_name}: {str(e)}")


def load_existing_direct_text_blob(file_name: str) -> str:
    """Load existing direct mode text blob from disk if it exists.
    
    Uses shared location: direct_texts/{file_name}.txt
    
    Returns:
        str: Text blob content or None if not found
    """
    text_path = f"direct_texts/{file_name}.txt"
    if not os.path.exists(text_path):
        return None
    
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.warning(f"Failed to load direct text blob for {file_name}: {str(e)}")
        return None


def build_context_blob(conversation: dict, query: str, char_limit: int = 6000, use_multi_query: bool = True, use_rerank: bool = True, k: int = 8, fetch_k: int = 20, enhanced: bool = False) -> str:
    """Build context blob from FAISS or direct documents with advanced retrieval techniques
    
    Args:
        conversation: Conversation dictionary with vectorstores
        query: User query string
        char_limit: Maximum characters in context blob
        use_multi_query: Use multiple query variations for better retrieval (default: True)
        use_rerank: Use cross-encoder re-ranking for better relevance (default: True)
        k: Number of results to return (default: 8)
        fetch_k: Number of candidates to fetch before MMR (default: 20)
        enhanced: If True, use enhanced parameters for better retrieval (more chunks, wider search)
    """
    # Enhanced mode: fetch more chunks with wider search
    if enhanced:
        k = k * 2  # Double the number of results
        fetch_k = fetch_k * 3  # Triple the candidate pool
        char_limit = char_limit * 2  # Allow more context
    if not conversation:
        return ""

    if conversation.get("faiss_enabled", True):
        vectorstores = conversation.get("vectorstores", {})
        if not vectorstores:
            return ""
        snippets = []
        for name, db in vectorstores.items():
            try:
                # Step 1: Query expansion - generate multiple query variations
                if use_multi_query:
                    queries = expand_query(query)
                    # Multi-query retrieval: get results from all query variations
                    results = multi_query_retrieval(db, queries, k=k*2, fetch_k=fetch_k)
                else:
                    # Single query with MMR
                    results = db.max_marginal_relevance_search(
                        query,
                        k=k,
                        fetch_k=fetch_k,
                        lambda_mult=0.5
                    )
                
                if use_rerank and results:
                    results = rerank_results(query, results, top_k=k)
                elif len(results) > k:
                    results = results[:k]
                
                if results:
                    chunk_text = _format_documents_with_metadata(results)
                    snippets.append(f"[{name}]\n{chunk_text}")
            except Exception as e:
                # Fallback to basic search if advanced methods fail
                results = db.similarity_search(query, k=k)
                if results:
                    chunk_text = "\n\n".join(doc.page_content for doc in results)
                    snippets.append(f"[{name}]\n{chunk_text}")
        
        if snippets:
            result = "\n\n---\n\n".join(snippets)
            return result[:char_limit] if len(result) > char_limit else result
        return ""

    # Direct Mode: Use a much larger character limit to simulate unoptimized full-text processing
    direct_char_limit = 200_000  # 200k chars (~50k tokens)
    direct_docs = conversation.get("direct_docs", [])
    if not direct_docs:
        return ""
    result = "\n\n---\n\n".join(direct_docs)
    return result[:direct_char_limit] if len(result) > direct_char_limit else result


def document_retrieval(query: str) -> str:
    """Tool function for document retrieval"""
    tab_name = st.session_state.get("active_tool_context")
    conversations = st.session_state.get("conversations", {})
    conversation = conversations.get(tab_name)

    blob = build_context_blob(conversation, query)
    return blob if blob else "No relevant info found."


def _read_files_and_compute_hashes(uploaded_files):
    """Read file contents and compute hashes in one pass"""
    file_hashes = {}
    file_contents = {}
    for uploaded_file in uploaded_files:
        content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for potential reuse
        file_hashes[uploaded_file.name] = compute_file_hash(content)
        file_contents[uploaded_file.name] = content
    return file_hashes, file_contents


def _check_files_changed(file_hashes: dict, file_metadata: dict) -> bool:
    """Check if any files have changed by comparing hashes
    
    Only checks the files in file_hashes - doesn't require exact set match.
    This allows loading a subset of files even if metadata has more files.
    """
    if not file_metadata:
        return True
    
    # Filter out index hash keys (format: "{file_name}_index_hash") from metadata
    stored_file_set = {k for k in file_metadata.keys() if not k.endswith("_index_hash")}
    current_file_set = set(file_hashes.keys())
    
    # Check if all current files exist in stored metadata
    if not current_file_set.issubset(stored_file_set):
        # Some files are new - they've changed
        return True
    
    # Check if any file hashes changed (only for files being uploaded)
    return any(
        file_metadata.get(file_name) != file_hash
        for file_name, file_hash in file_hashes.items()
    )


def _create_documents_with_metadata(chunks, file_name: str):
    """Create Document objects with metadata from chunks"""
    chunks_per_page = max(1, len(chunks) // 10)
    documents = []
    
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk.page_content,
            metadata={
                **getattr(chunk, 'metadata', {}),
                'source': file_name,
                'page': getattr(chunk, 'metadata', {}).get('page', (i // chunks_per_page) + 1)
            }
        )
        documents.append(doc)
    
    return documents


def _try_quick_load_from_disk(uploaded_files, file_metadata: dict, tab_name: str, 
                               conversation: dict, current_signature: tuple) -> bool:
    """Try to load files from disk without reading file contents first (optimization)"""
    if not file_metadata:
        return False
    
    file_names_in_metadata = {k for k in file_metadata.keys() if not k.endswith("_index_hash")}
    current_file_names = frozenset(f.name for f in uploaded_files)
    
    if file_names_in_metadata != current_file_names:
        return False
    
    # All files exist in metadata - try loading from disk
    all_loaded = True
    index_hashes = {}
    temp_file_hashes = {}
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        expected_hash = file_metadata.get(f"{file_name}_index_hash")
        existing_index, current_hash = load_existing_faiss_index(tab_name, file_name, expected_hash)
        
        if existing_index:
            conversation.setdefault("vectorstores", {})[file_name] = existing_index
            index_hashes[file_name] = current_hash
            temp_file_hashes[file_name] = file_metadata.get(file_name)
        else:
            all_loaded = False
            break
    
    if all_loaded:
        # Update metadata and return
        metadata_key = f"{tab_name}_files"
        updated_metadata = temp_file_hashes.copy()
        for file_name, index_hash in index_hashes.items():
            updated_metadata[f"{file_name}_index_hash"] = index_hash
        _update_file_metadata(metadata_key, updated_metadata)
        conversation["last_upload_signature"] = current_signature
        st.toast(f"📚 Loaded {len(uploaded_files)} file(s) from disk. Ready to query!", icon="✅")
        return True
    
    return False


def _load_existing_indexes(uploaded_files, file_hashes, file_metadata, tab_name, conversation, 
                          show_toasts: bool = True, toast_key: str = None):
    """Try to load existing FAISS indexes for unchanged files"""
    if not conversation.get("faiss_enabled", True):
        return False, {}
    
    index_hashes = {}
    shown_toasts = st.session_state.get(toast_key, set()) if toast_key else set()
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        expected_hash = file_metadata.get(f"{file_name}_index_hash")
        existing_index, current_hash = load_existing_faiss_index(tab_name, file_name, expected_hash)
        
        if existing_index:
            conversation.setdefault("vectorstores", {})[file_name] = existing_index
            index_hashes[file_name] = current_hash
            if show_toasts:
                if toast_key and file_name not in shown_toasts:
                    st.toast(f"📚 Loaded existing FAISS index for {file_name}.", icon="✅")
                    shown_toasts.add(file_name)
                elif not toast_key:
                    st.toast(f"📚 Loaded existing FAISS index for {file_name}.", icon="✅")
        else:
            return False, {}
    
    if toast_key:
        st.session_state[toast_key] = shown_toasts
    
    # Update metadata with index hashes
    updated_metadata = file_metadata.copy() if file_metadata else {}
    updated_metadata.update(file_hashes)
    for file_name, index_hash in index_hashes.items():
        updated_metadata[f"{file_name}_index_hash"] = index_hash
    return True, updated_metadata


def _filter_files_to_process(uploaded_files, file_contents: dict, conversation: dict, 
                             tab_name: str, file_metadata: dict) -> list:
    """Filter out files that are already loaded (in memory or on disk) to avoid re-processing"""
    faiss_enabled = conversation.get("faiss_enabled", True)
    files_to_process = []
    
    if faiss_enabled:
        vectorstores = conversation.get("vectorstores", {})
        loaded_file_names = set(vectorstores.keys())
        
        # Try to load any missing files from disk
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in loaded_file_names:
                # Try loading from disk
                expected_hash = file_metadata.get(f"{uploaded_file.name}_index_hash") if file_metadata else None
                existing_index, _ = load_existing_faiss_index(tab_name, uploaded_file.name, expected_hash)
                if existing_index:
                    vectorstores[uploaded_file.name] = existing_index
                    conversation.setdefault("vectorstores", {}).update(vectorstores)
                    loaded_file_names.add(uploaded_file.name)
                else:
                    # Not in memory or on disk, needs processing
                    files_to_process.append(uploaded_file)
    else:
        direct_docs = conversation.get("direct_docs", [])
        loaded_file_names = set()
        for doc in direct_docs:
            if doc.startswith("[") and "]" in doc:
                file_name = doc.split("]")[0][1:]
                loaded_file_names.add(file_name)
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in loaded_file_names:
                files_to_process.append(uploaded_file)
    
    return files_to_process


def handle_pdf_upload(uploaded_files, conversation: dict, tab_name: str):
    """Handle PDF upload and process based on FAISS setting with hybrid client/server storage"""
    # Handle empty upload (clearing all files)
    if not uploaded_files:
        if conversation.get("vectorstores") or conversation.get("direct_docs"):
            conversation["vectorstores"] = {}
            conversation["direct_docs"] = []
            conversation["last_upload_signature"] = None
            st.toast(f"🗑️ Cleared all documents from {tab_name}.", icon="🗑️")
        return

    # Step 1: Check filename
    current_file_names = frozenset(f.name for f in uploaded_files)
    faiss_enabled = conversation.get("faiss_enabled", True)

    # Sync existing context with current uploads (remove cancelled/deleted files)
    if faiss_enabled:
        vectorstores = conversation.get("vectorstores", {})
        # Create list to avoid runtime error during iteration
        for file_name in list(vectorstores.keys()):
            if file_name not in current_file_names:
                del vectorstores[file_name]
                # st.toast(f"Removed {file_name} from active context", icon="🗑️")
    else:
        direct_docs = conversation.get("direct_docs", [])
        # Filter out docs that are not in current_file_names
        conversation["direct_docs"] = [
            doc for doc in direct_docs 
            if doc.startswith("[") and doc.split("]")[0][1:] in current_file_names
        ]

    current_signature = (current_file_names, faiss_enabled)
    metadata_key = f"{tab_name}_files"
    file_metadata = _get_file_metadata(metadata_key)
    
    # Step 2: See if exists in memory
    if faiss_enabled:
        vectorstores = conversation.get("vectorstores", {})
        if set(vectorstores.keys()) == current_file_names:
            conversation["last_upload_signature"] = current_signature
            return
    else:
        direct_docs = conversation.get("direct_docs", [])
        if direct_docs:
            loaded_file_names = {doc.split("]")[0][1:] for doc in direct_docs if doc.startswith("[") and "]" in doc}
            if loaded_file_names == current_file_names:
                conversation["last_upload_signature"] = current_signature
                return
    
    # Step 2: See if exists on disk (check metadata for filename)
    if not file_metadata:
        # No metadata - files not processed before, need to process
        file_hashes, file_contents = _read_files_and_compute_hashes(uploaded_files)
    else:
        file_names_in_metadata = {k for k in file_metadata.keys() if not k.endswith("_index_hash")}
        if file_names_in_metadata != current_file_names:
            # Filenames don't match - new files, need to process
            file_hashes, file_contents = _read_files_and_compute_hashes(uploaded_files)
        else:
            # Filenames match - try quick load from disk (doesn't require reading files)
            # Only use quick load for FAISS as it verifies index integrity. 
            # Direct mode needs to verify hashes or load text blobs which is handled in "files unchanged" block.
            if faiss_enabled and _try_quick_load_from_disk(uploaded_files, file_metadata, tab_name, conversation, current_signature):
                return
            # Quick load failed - read files to compare hashes
            file_hashes, file_contents = _read_files_and_compute_hashes(uploaded_files)
    
    # Step 3: Compare hashes
    files_changed = _check_files_changed(file_hashes, file_metadata)
    
    # Step 4: If match, use existing; otherwise reprocess
    if not files_changed:
        # Hashes match - files unchanged, load from disk
        if faiss_enabled:
            all_loaded, updated_metadata = _load_existing_indexes(
                uploaded_files, file_hashes, file_metadata, tab_name, conversation
            )
            if all_loaded:
                _update_file_metadata(metadata_key, updated_metadata)
                conversation["last_upload_signature"] = current_signature
                st.toast(f"📚 Loaded {len(uploaded_files)} file(s) from disk. Ready to query!", icon="✅")
                return
        else:
            # Direct mode - files unchanged, load from disk
            all_loaded = True
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                text_blob = load_existing_direct_text_blob(file_name)
                
                if text_blob:
                    conversation.setdefault("direct_docs", []).append(
                        f"[{file_name}]\n{text_blob}"
                    )
                else:
                    all_loaded = False
                    break
            
            if all_loaded:
                conversation["last_upload_signature"] = current_signature
                updated_metadata = file_metadata.copy() if file_metadata else {}
                updated_metadata.update(file_hashes)
                _update_file_metadata(metadata_key, updated_metadata)
                st.toast(f"📄 Loaded {len(uploaded_files)} file(s) from disk. Ready to query!", icon="✅")
                return
    
    # Hashes don't match or files not found - reprocess
    conversation["last_upload_signature"] = current_signature
    
    # Filter out files that are already loaded (in memory or on disk) to avoid re-processing
    files_to_process = _filter_files_to_process(uploaded_files, file_contents, conversation, tab_name, file_metadata)
    
    # If all files are already loaded, nothing to do
    if not files_to_process:
        return
    
    # Calculate optimal chunk sizes based on total file size (only for files to process)
    total_size = sum(len(file_contents[f.name]) for f in files_to_process)
    chunk_size, chunk_overlap = _calculate_chunk_sizes(total_size)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False
    )

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Parse only files that need processing
    parsed_files = _parse_pdf_files_parallel(
        files_to_process, file_contents, splitter, tab_name, status_text
    )
    
    if not parsed_files:
        progress_bar.empty()
        status_text.empty()
        return
    
    # Process documents and create indexes
    if faiss_enabled:
        status_text.text("🔢 Creating vector indexes...")
        progress_bar.progress(0.5)
        
        all_documents = []
        file_doc_map = {}
        
        total_chunks = sum(len(parsed['chunks']) for parsed in parsed_files)
        if total_chunks > 100:
            status_text.text(f"🔢 Processing {total_chunks} chunks...")
        
        for parsed in parsed_files:
            file_name = parsed['file_name']
            chunks = parsed['chunks']
            file_docs = _create_documents_with_metadata(chunks, file_name)
            
            start_idx = len(all_documents)
            all_documents.extend(file_docs)
            file_doc_map[file_name] = list(range(start_idx, len(all_documents)))
        
        progress_bar.progress(0.7)
        
        total_docs = len(all_documents)
        status_text.text(f"🔢 Generating embeddings for {total_docs} chunks...")
        
        def _create_file_index(file_name, doc_indices, file_hash, file_num=0, total_files=1):
            """Create FAISS index for a single file with progress tracking"""
            try:
                file_docs = [all_documents[i] for i in doc_indices]
                
                # Use FAISS.from_documents() which handles docstore creation properly
                file_db = FAISS.from_documents(
                    documents=file_docs,
                    embedding=embedding_fn
                )
                
                # Update progress
                overall_progress = 0.7 + (0.25 * ((file_num + 1) / total_files))
                progress_bar.progress(overall_progress)
                status_text.text(f"✅ Created index for {file_name}...")
                
                # Save index (non-blocking for large files)
                # Use shared location: filename only (works for both single and dual chat modes)
                index_path = f"faiss_indexes/{file_name}"
                try:
                    file_db.save_local(index_path)
                    index_hash = compute_index_hash(index_path)
                except Exception as save_error:
                    st.warning(f"Could not save index for {file_name}, but it's available in memory")
                    index_hash = None
                
                return {
                    'file_name': file_name,
                    'file_hash': file_hash,
                    'index_hash': index_hash,
                    'file_db': file_db,
                    'type': 'faiss'
                }
            except Exception as e:
                st.error(f"Error creating index for {file_name}: {str(e)}")
                return None
        
        # Create indexes sequentially to show progress (parallel doesn't help with CPU-bound embedding)
        results = []
        file_stores = {}
        file_names = list(file_doc_map.keys())
        total_files = len(file_names)
        
        for file_idx, file_name in enumerate(file_names):
            result = _create_file_index(
                file_name, 
                file_doc_map[file_name], 
                file_hashes[file_name], 
                file_num=file_idx,
                total_files=total_files
            )
            if result:
                results.append(result)
                file_stores[result['file_name']] = result['file_db']
                status_text.text(f"✅ Completed {result['file_name']} ({file_idx + 1}/{total_files} files)")
        
        # Show completion message
        status_text.text("✅ All embeddings complete! Finalizing indexes...")
        progress_bar.progress(0.95)
        conversation.setdefault("vectorstores", {}).update(file_stores)
        
    else:
        # Direct mode - simpler processing
        results = []
        for parsed in parsed_files:
            file_name = parsed['file_name']
            chunks = parsed['chunks']
            text_blob = "\n\n".join(chunk.page_content for chunk in chunks)
            
            # Save text blob to disk for persistence
            save_direct_text_blob(file_name, text_blob)
            
            results.append({
                'file_name': file_name,
                'file_hash': file_hashes[file_name],
                'text_blob': text_blob,
                'type': 'direct'
            })
    
    # Final completion
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete! Your PDFs are ready to query.")
    
    # Clear progress indicators (completion message stays visible via toast notifications)
    progress_bar.empty()
    # Keep status text visible briefly, then clear
    status_text.empty()
    
    # Update metadata and show results
    _update_metadata_and_notify(results, file_metadata, conversation, tab_name, metadata_key)
    _update_file_metadata(metadata_key, file_metadata)
    
    # Clear processing flag
    processing_key = f"{tab_name}_processing_pdfs"
    if processing_key in st.session_state:
        del st.session_state[processing_key]
