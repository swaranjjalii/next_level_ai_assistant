from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from .data_processing import DataProcessor

class VectorRetriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = {}  # Maps doc_id to embedding
        self.documents = {}  # Maps doc_id to document text
        self.metadata = {}  # Maps doc_id to document metadata

    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Add a document to the retrieval index"""
        embedding = self.model.encode(text)
        self.index[doc_id] = embedding
        self.documents[doc_id] = text
        self.metadata[doc_id] = metadata or {}

    def get_document_text(self, doc_id: str) -> str:
        """Get the text of a document by its ID"""
        return self.documents.get(doc_id, "")

    def get_document_metadata(self, doc_id: str) -> Dict:
        """Get the metadata of a document by its ID"""
        return self.metadata.get(doc_id, {})

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for documents similar to the query"""
        query_embed = self.model.encode(query)
        scores = {
            doc_id: np.dot(query_embed, doc_embed) 
            for doc_id, doc_embed in self.index.items()
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def ingest_directory(self, directory_path: str, file_extensions: List[str] = None):
        """
        Ingest all documents from a directory
        
        Args:
            directory_path: Path to the directory containing documents
            file_extensions: List of file extensions to include (e.g., ['.txt', '.md'])
        """
        file_extensions = file_extensions or ['.txt', '.md', '.csv', '.json']
        processor = DataProcessor()
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                
                if ext.lower() not in file_extensions:
                    continue
                
                try:
                    if ext.lower() == '.csv':
                        df = processor.load_csv(file_path)
                        for i, row in df.iterrows():
                            doc_id = f"{file}_{i}"
                            text = ' '.join(str(v) for v in row.values)
                            self.add_document(doc_id, text, {"source": file_path, "row": i})
                    
                    elif ext.lower() == '.json':
                        data = processor.load_json(file_path)
                        if isinstance(data, list):
                            for i, item in enumerate(data):
                                doc_id = f"{file}_{i}"
                                text = json.dumps(item)
                                self.add_document(doc_id, text, {"source": file_path, "index": i})
                        else:
                            self.add_document(file, json.dumps(data), {"source": file_path})
                    
                    else:  # .txt, .md, etc.
                        text = processor.load_text(file_path)
                        # Split long documents into chunks
                        chunks = self._chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            doc_id = f"{file}_{i}" if len(chunks) > 1 else file
                            self.add_document(doc_id, chunk, {"source": file_path, "chunk": i})
                
                except Exception as e:
                    print(f"Error ingesting file {file_path}: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to find a good breakpoint
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break != -1 and para_break > start + chunk_size // 2:
                    end = para_break
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def save_index(self, file_path: str):
        """Save the index to disk"""
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "embeddings": {k: v.tolist() for k, v in self.index.items()}
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    def load_index(self, file_path: str):
        """Load the index from disk"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.index = {k: np.array(v) for k, v in data["embeddings"].items()}