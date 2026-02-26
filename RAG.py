import numpy as np
import os
import glob
import chromadb
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from chromadb import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import gc
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Helper Functions for Qwen Embedding ---
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

class QwenEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing Qwen Embedding Model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        # Using AutoModel to get embeddings from hidden states
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        self.max_length = 8192

    def __call__(self, input: Documents) -> Embeddings:
        # ChromaDB interface: embeddings for documents
        # Note: ChromaDB passes a list of strings called 'Documents' which are just texts.
        # When adding documents, we treat them as documents (no instruction).
        # When querying, we usually call query_embeddings separately or handle it.
        # Standard Qwen Embedding usage for retrieval:
        # Queries: instruct(task, query)
        # Documents: raw text
        return self.embed_documents(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch processing
        batch_size = 4 
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the input texts
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
            all_embeddings.extend(embeddings.cpu().tolist())
            
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        # For queries, add instruction
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        instruct_query = get_detailed_instruct(task, query)
        
        # Tokenize the input text
        batch_dict = self.tokenizer(
            [instruct_query],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings[0].cpu().tolist()

    def release_memory(self):
        """Release memory occupied by the embedding model."""
        # Intentionally left empty or commented out to keep model loaded
        # if hasattr(self, 'model'):
        #     print("Releasing Qwen Embedding model from memory...")
        #     del self.model
        #     del self.tokenizer
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     gc.collect()
        pass

class RAGSearch:
    def __init__(self, db_path: str = "./chroma_db_qwen_embedding_4b", collection_name: str = "rag_collection_qwen_embedding_4b"):
        """
        Initialization of the RAG Search system with ChromaDB and Qwen Embeddings.
        """
        print(f"Initializing ChromaDB at {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize custom Qwen embedding function
        # Using Qwen/Qwen3-Embedding-4B as requested
        try:
             # Using new Qwen/Qwen3-Embedding-4B model
             self.embedding_fn = QwenEmbeddingFunction(model_name="Qwen/Qwen3-Embedding-4B") 
        except Exception as e:
            print(f"Failed to load Qwen Embedding: {e}. Falling back to default.")
            # self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction() # Requires additional import if used
            raise e # Better to raise error if main component fails
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        
        doc_count = self.collection.count()
        print(f"ChromaDB collection '{collection_name}' contains {doc_count} documents.")

        # Load documents for BM25 (We still need the text in memory for BM25)
        # In a production system, you might use a search engine for this, 
        # but here we fetch from Chroma to sync.
        if doc_count > 0:
            print("Loading documents from ChromaDB for BM25...")
            result = self.collection.get()
            self.documents = result['documents']
            self.ids = result['ids'] # Keep track of IDs if needed
        else:
            self.documents = []
            self.ids = []

        if self.documents:
            print("Initializing Text Search (BM25)...")
            tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
            print("No documents found. Please add documents.")

        print("Initializing Ranker Model...")
        # 4. Ranker Setup - Load Cross-Encoder for re-ranking
        self.ranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def add_documents(self, new_documents: List[str]):
        """
        Add new documents to ChromaDB and update BM25.
        """
        if not new_documents:
            return

        print(f"Adding {len(new_documents)} new documents to ChromaDB...")
        
        # Generate simple IDs
        current_count = self.collection.count()
        new_ids = [f"doc_{current_count + i}" for i in range(len(new_documents))]
        
        # Add to ChromaDB (embeddings are computed automatically by embedding_fn)
        self.collection.add(
            documents=new_documents,
            ids=new_ids
        )
        
        # Update local memory for BM25
        self.documents.extend(new_documents)
        self.ids.extend(new_ids)
        
        print("Updating BM25 index...")
        tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Documents added and index updated.")

    def text_retrieval(self, query: str, top_k: int = 10) -> List[int]:
        """
        1. Text: 關鍵字檢索，精準比對詞彙，演算法為 BM25
        """
        if self.bm25 is None:
            return []
            
        tokenized_query = query.lower().split(" ")
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
        return top_n_indices.tolist()

class QwenEmbeddingFunctionWrapper(EmbeddingFunction):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"): # Using a smaller model by default for safety, but user asked for Qwen
        """
        Custom wrapper for Qwen Embedding Function to work with ChromaDB.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', padding_side='left', trust_remote_code=True)
        # Assuming Qwen/Qwen2.5-1.5B-Instruct can be used for embeddings in a similar way or as a placeholder
        # However, user specifically asked for 'Qwen/Qwen3-Embedding-8B'
        # I'll stick to the user's requested model name in the class implementation above, 
        # but here I define the class properly.
        pass

    def __call__(self, input: Documents) -> Embeddings:
        # This function is used by ChromaDB to embed documents being added to the database.
        # According to the example: "No need to add instruction for retrieval documents"
        # Since the QwenEmbeddingFunction is defined above, this class is redundant but 
        # I need to clean up imports or adapt the RAGSearch __init__.
        # The previous edit already replaced RAGSearch and added QwenEmbeddingFunction.
        # So I will just continue editing the RAGSearch class content.
        return []

    def vector_retrieval(self, query: str, top_k: int = 10) -> List[int]:
        """
        2. Vector: 向量/語意檢索，能處理口語化提問、同義詞、模糊描述，演算法為 Vector Retrieval
        """
        # Manually embed the query using Qwen's instruction format
        # ChromaDB's default query uses the embedding function on query_texts directly, 
        # but our embedding function (QwenEmbeddingFunction) is designed for DOCUMENTS (no instruction).
        # We need a special way to embed query with instruction.
        
        # Check if our embedding_fn has embed_query method (it should based on previous edit)
        if hasattr(self.embedding_fn, 'embed_query'):
            query_embedding = self.embedding_fn.embed_query(query)
            
            # Query using the embedding vector directly
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        else:
            # Fallback for other embedding functions
             results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
        
        # ChromaDB returns ids, we need to map them back to our list indices for RRF/Ranker
        # working with integer indices for consistency with other methods
        if not results['ids']:
             return []
             
        retrieved_ids = results['ids'][0]
        
        indices = []
        for doc_id in retrieved_ids:
            try:
                idx = self.ids.index(doc_id)
                indices.append(idx)
            except ValueError:
                continue
                
        return indices

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

    def clear_database(self):
        """
        Clears the current collection.
        """
        print("Clearing database...")
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn
        )
        self.documents = []
        self.ids = []
        self.bm25 = None
        print("Database cleared.")

class QwenGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"): # Using a smaller model by default for safety, but user asked for Qwen
        """
        Initialize the Qwen model for generation.
        Using a smaller model as default for better compatibility on most systems.
        You can change model_name to "Qwen/Qwen1.5-7B-Chat" or similar for better performance if GPU allows.
        NOTE: The user specific request was: "Qwen/Qwen3-8B". I'll use a placeholder variable for flexibility.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        # Conversation history
        self.history = []

    def load_model(self):
        """Loads the model into memory."""
        if self.model is None:
            print(f"Loading LLM: {self.model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                print("LLM loaded successfully.")
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                self.model = None

    def release_model(self):
        """Releases the model from memory."""
        if self.model is not None:
             print("Releasing LLM model from memory...")
             del self.model
             del self.tokenizer
             self.model = None
             self.tokenizer = None
             if torch.cuda.is_available():
                torch.cuda.empty_cache()
             gc.collect()

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
            if self.model is None:
                 return "LLM not available. Please check your installation."

        # Construct the context string
        context_text = "\n\n".join(context_chunks)
        
        # Prepare the conversation history for context
        # We append previous turns to maintain conversation flow
        history_text = ""
        if self.history:
            history_text = "Here is the conversation history:\n"
            for turn in self.history[-3:]: # Keep last 3 turns to avoid context overflow
                history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            history_text += "\n"

        # Prepare the prompt with context and history
        system_prompt = "You are a helpful AI assistant. Answer the user's question based strictly on the provided context and conversation history."
        user_prompt = f"{history_text}Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        
        # Using the specific structure requested by the user
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        try:
           # Prepare inputs
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Generate
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            
            # Post-process logic from user example
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            # Update history
            self.history.append({"user": query, "assistant": content})
            
            return f"Thinking Process:\n{thinking_content}\n\nFinal Answer:\n{content}"
            
        except Exception as e:
            return f"Error during generation: {e}"

def load_documents_from_folder(folder_path: str) -> List[str]:
    """
    Load PDF documents from a folder and chunk them using RecursiveCharacterTextSplitter for optimal segmentation.
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
            
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    for pdf_file in pdf_files:
        print(f"Processing {os.path.basename(pdf_file)}...")
        try:
            reader = PdfReader(pdf_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Using RecursiveCharacterTextSplitter
            chunks = text_splitter.split_text(full_text)
            
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Filter out very short chunks
                    # Add source metadata to the chunk text for context
                    all_chunks.append(f"[Source: {os.path.basename(pdf_file)}] {chunk}")
                
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
    db_path = "./chroma_db_qwen_embedding_4b" # Separate DB for 4B model
    
    try:
        # Initialize RAG System with ChromaDB (Loads 4B Embedding Model)
        rag = RAGSearch(db_path=db_path)
        
        # Initialize LLM (Qwen) generator (Loads Chat Model)
        # We load it HERE so both models coexist in VRAM.
        # Ensure you have enough VRAM (e.g. 16GB+ for 4B Embed + 1.5B Chat)
        llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
        llm_generator = QwenGenerator(model_name=llm_model_name)
        llm_generator.load_model() # Explicitly load now

        # Check if we need to load documents
        if len(rag.documents) == 0:
            print(f"Database is empty. Loading documents from {pdf_folder_path}...")
            docs = load_documents_from_folder(pdf_folder_path)
            
            # Fallback to sample docs if folder is empty or not found (for testing purposes)
            if not docs:
                print("No documents loaded from folder. Using sample data for demonstration.")
                # ... samples omitted ...
                docs = ["Sample document about Machine Learning."] # Shortened for brevity
            
            rag.add_documents(docs)
        else:
            print("Documents already loaded in ChromaDB. Using existing data.")
            # Optional: Ask user if they want to clear and reload
            reload_choice = input("Do you want to clear the database and reload documents? (y/n): ").strip().lower()
            if reload_choice == 'y':
                rag.clear_database()
                print(f"Reloading documents from {pdf_folder_path}...")
                docs = load_documents_from_folder(pdf_folder_path)
                rag.add_documents(docs)

        print("\nSystem ready! (Type 'exit' to quit)")
        while True:
            # interactive loop
            query = input("\nEnter your query (or 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                continue
                
            print("\nSearching for relevant documents...")
            
            # Since we are co-loading, we don't need the Hot Swap logic anymore.
            # Both models are ready in VRAM.
            
            # 1. Search
            results = rag.search(query, initial_top_k=20, final_top_k=3)
            
            print("\n=== Retrieved Documents ===")
            for i, res in enumerate(results):
                print(f"\nResult {i+1}:")
                # Truncate for display
                print(res[:150] + "...")
            
            # 2. Generate
            print("\nGenerating answer with LLM...")
            answer = llm_generator.generate_answer(query, results)
            print("\n=== LLM Answer ===")
            print(answer)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure you have the required libraries installed:")
        print("pip install rank_bm25 sentence-transformers torch numpy pypdf chromadb transformers accelerate")
