import numpy as np
import os
import glob
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from typing import List, Dict, Tuple

class RAGSearch:
    def __init__(self, documents: List[str]):
        """
        Initialization of the RAG Search system.
        """
        self.documents = documents
        
        print(f"Initializing RAG with {len(documents)} document chunks...")

        print("Initializing Text Search (BM25)...")
        # 1. Text Search Setup (BM25) - Simple whitespace tokenizer for demo
        tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Initializing Vector Search Models...")
        # 2. Vector Search Setup - Load embedding model
        # using a small efficient model for demo
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        self.doc_embeddings = self.embedder.encode(documents, convert_to_tensor=True)

        print("Initializing Ranker Model...")
        # 4. Ranker Setup - Load Cross-Encoder for re-ranking
        self.ranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def text_retrieval(self, query: str, top_k: int = 10) -> List[int]:
        """
        1. Text: 關鍵字檢索，精準比對詞彙，演算法為 BM25
        """
        tokenized_query = query.lower().split(" ")
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
        return top_n_indices.tolist()

    def vector_retrieval(self, query: str, top_k: int = 10) -> List[int]:
        """
        2. Vector: 向量/語意檢索，能處理口語化提問、同義詞、模糊描述，演算法為 Vector Retrieval
        """
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        hits = util.semantic_search(query_embedding, self.doc_embeddings, top_k=top_k)
        # hits is a list of lists (one for each query), we take the first
        top_n_indices = [hit['corpus_id'] for hit in hits[0]]
        return top_n_indices

    def reciprocal_rank_fusion(self, bm25_indices: List[int], vector_indices: List[int], k: int = 60) -> List[int]:
        """
        3. Hybrid: 截長補短，融合精準匹配語意相似排名後綜合結果，演算法為 RRF
        RRF Score(d) = 1 / (k + rank(d))
        """
        rrf_scores = {}

        # Process BM25 Ranks
        for rank, index in enumerate(bm25_indices):
            if index not in rrf_scores:
                rrf_scores[index] = 0
            rrf_scores[index] += 1 / (k + rank + 1)

        # Process Vector Ranks
        for rank, index in enumerate(vector_indices):
            if index not in rrf_scores:
                rrf_scores[index] = 0
            rrf_scores[index] += 1 / (k + rank + 1)

        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_indices

    def ranker_rerank(self, query: str, candidate_indices: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        4. Ranker: 第二階段排序，用來對第一階段資料(Usually top 50)更精細的檢查相關性，常用深度學習模型計算
        5. 如果有開啟 Ranker，最終排序會依據 Ranker 分數排序
        """
        # Prepare pairs for Cross-Encoder
        candidate_pairs = [[query, self.documents[idx]] for idx in candidate_indices]
        
        if not candidate_pairs:
            return []

        # Predict scores
        scores = self.ranker_model.predict(candidate_pairs)
        
        # Combine index and score
        results = list(zip(candidate_indices, scores))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]

    def search(self, query: str, use_ranker: bool = True, initial_top_k: int = 50, final_top_k: int = 5) -> List[str]:
        """
        Orchestrate the full search process based on the user provided diagram.
        """
        print(f"\nProcessing Query: {query}")
        
        # 1. Text Retrieval (BM25)
        bm25_results = self.text_retrieval(query, top_k=initial_top_k)
        print(f"- BM25 retrieved {len(bm25_results)} docs")

        # 2. Vector Retrieval
        vector_results = self.vector_retrieval(query, top_k=initial_top_k)
        print(f"- Vector retrieved {len(vector_results)} docs")

        # 3. Hybrid Retrieval (RRF)
        combined_indices = self.reciprocal_rank_fusion(bm25_results, vector_results)
        print(f"- Hybrid Fusion (RRF) combined to {len(combined_indices)} candidates")

        # Select candidates for reranking (e.g., top 50 from RRF)
        candidates_for_reranking = combined_indices[:initial_top_k]

        if use_ranker:
            # 4. Ranker (Re-ranking)
            print("- Running Ranker (Cross-Encoder)...")
            ranked_results = self.ranker_rerank(query, candidates_for_reranking, top_k=final_top_k)
            final_indices = [idx for idx, score in ranked_results]
        else:
            final_indices = candidates_for_reranking[:final_top_k]

        # Return actual documents
        return [self.documents[idx] for idx in final_indices]

def load_documents_from_folder(folder_path: str) -> List[str]:
    """
    Load PDF documents from a folder and chunk them into paragraphs.
    """
    all_chunks = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return []

    # Get all PDF files
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []
        
    print(f"Found {len(pdf_files)} PDF files in {folder_path}")

    # Try importing pypdf
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            print("Error: Please install pypdf to read PDF files: pip install pypdf")
            return []

    for pdf_file in pdf_files:
        print(f"Processing {os.path.basename(pdf_file)}...")
        try:
            reader = PdfReader(pdf_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Simple chunking by paragraphs (double newlines)
            # Filter out very short chunks
            paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
            
            # If no paragraphs found (maybe single newlines), try splitting by single newlines but grouping
            if not paragraphs:
                 paragraphs = [p.strip() for p in full_text.split('\n') if len(p.strip()) > 50]

            for p in paragraphs:
                # Add source metadata to the chunk text for context
                all_chunks.append(f"[Source: {os.path.basename(pdf_file)}] {p}")
                
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")

    return all_chunks

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Define folder path
    # Using the path provided by the user
    pdf_folder_path = r"d:\Users\jiayanjylin\Desktop\ML\paper\Yolo"
    
    # 1. Load Documents
    print(f"Loading documents from {pdf_folder_path}...")
    docs = load_documents_from_folder(pdf_folder_path)

    # Fallback to sample docs if folder is empty or not found (for testing purposes)
    if not docs:
        print("No documents loaded from folder. Using sample data for demonstration.")
        docs = [
            "Python is a high-level, general-purpose programming language.",
            "BM25 is a ranking function used by search engines to estimate the relevance of documents.",
            "Vector databases allow for semantic search by storing embeddings.",
            "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Hybrid search combines keyword search and vector search to improve retrieval accuracy.",
            "Reciprocal Rank Fusion (RRF) is a method to combine multiple result lists.",
            "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and AI.",
            "Transformers are deep learning models that handle sequential data.",
            "Bert is a transformer-based machine learning technique for natural language processing pre-training."
        ]

    try:
        # Initialize RAG System
        rag = RAGSearch(docs)

        while True:
            # interactive loop
            query = input("\nEnter your query (or 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                continue
                
            results = rag.search(query, initial_top_k=20, final_top_k=3)
            
            print("\n=== Final Search Results ===")
            for i, res in enumerate(results):
                print(f"\nResult {i+1}:")
                # Print first 500 chars to avoid flooding terminal
                print(res[:500] + "..." if len(res) > 500 else res)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the required libraries installed:")
        print("pip install rank_bm25 sentence-transformers torch numpy pypdf")
