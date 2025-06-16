import os
from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
import numpy as np

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LangChainRAG:
    """RAG implementation using LangChain components"""
    
    def __init__(self, model_name: str = "anthropic/claude-3-haiku:free"):
        """Initialize LangChain RAG with specified model"""
        self.model_name = model_name
        
        # Configure the LLM based on the provider
        if "groq/" in model_name:
            # Groq model
            self.llm = ChatOpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                model=model_name.replace("groq/", ""),
                temperature=0.7,
                max_tokens=1000
            )
        else:
            # Default to OpenRouter for all other models
            self.llm = ChatOpenAI(
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                model=model_name,
                temperature=0.7,
                max_tokens=1000,
                # OpenRouter-specific headers
                default_headers={
                    "HTTP-Referer": "https://finsolve-rag.com",
                    "X-Title": "FinSolve RAG-RBAC"
                }
            )
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def semantic_search(self, query_embedding, documents, top_k=3):
        """Search for documents similar to the query using Google embeddings"""
        results = []
        
        for doc in documents:
            # Check if document has chunks
            if "embedding_chunks" not in doc or not doc["embedding_chunks"]:
                continue
                
            # Find the highest similarity among chunks
            max_similarity = 0
            for chunk_embedding in doc["embedding_chunks"]:
                similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                max_similarity = max(max_similarity, similarity)
                
            if max_similarity > 0:
                results.append({
                    "document": doc,
                    "similarity": max_similarity
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def _format_chat_history(self, chat_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format chat history into a list of messages for LangChain"""
        formatted_history = []
        
        for entry in chat_history:
            # Add user message
            formatted_history.append(HumanMessage(content=entry["query"]))
            
            # Add AI response
            formatted_history.append(AIMessage(content=entry["response"]))
        
        return formatted_history
    
    def _convert_to_langchain_docs(self, mongo_docs: List[Dict[str, Any]]) -> List[Document]:
        """Convert MongoDB documents to LangChain Document objects"""
        langchain_docs = []
        for doc in mongo_docs:
            langchain_docs.append(
                Document(
                    page_content=doc.get("document", ""),
                    metadata={
                        "id": str(doc.get("_id", "")),
                        "title": doc.get("title", "Untitled"),
                        "category": doc.get("category", "Uncategorized"),
                        "description": doc.get("description", "")
                    }
                )
            )
        return langchain_docs
    
    def create_rag_chain_with_history(self, user_role: str, chat_history: Optional[List[Dict[str, Any]]] = None):
        """Create a LangChain RAG chain with chat history"""
        # Format chat history if provided
        formatted_history = []
        if chat_history:
            formatted_history = self._format_chat_history(chat_history)
        
        # Create system prompt template
        system_template = """You are an AI assistant for FinSolve Technologies, providing role-specific information to users. 
You're responding to a user with the role: {role}.

Follow these guidelines:
1. Answer questions based ONLY on the context provided below
2. If the information isn't in the context, say "I don't have that information" - DO NOT make up answers
3. Keep responses professional, clear, and concise
4. Include citations to the source documents when appropriate using [Document Title]
5. Focus on providing factual information relevant to the user's role
6. Consider the conversation history for context
7. For CSV data, interpret the data as structured tables with headers and rows
   - Present tabular data in a readable format
   - If asked for specific data points, extract them precisely
   - For financial data, format numbers appropriately (e.g., currency symbols, decimal places)
8. For Markdown data:
   - Properly interpret headers, lists, tables, and other formatting
   - Preserve the hierarchical structure when relevant to the query
   - Recognize and properly handle code blocks or technical content

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{history}
"""
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])
        
        # Create history formatter
        def format_history(history):
            if not history:
                return "No previous conversation."
            return "\n".join([f"User: {h.content}" if isinstance(h, HumanMessage) else f"Assistant: {h.content}" for h in history])
        
        # Create the chain
        chain = (
            {
                "role": lambda _: user_role,
                "context": RunnablePassthrough(),
                "history": lambda _: format_history(formatted_history),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    async def generate_response(self, 
                               query: str, 
                               retrieved_docs: List[Dict[str, Any]], 
                               user_role: str,
                               chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate response using the RAG chain with chat history"""
        try:
            # Convert MongoDB docs to LangChain documents
            langchain_docs = self._convert_to_langchain_docs(retrieved_docs)
            
            # Create context text from documents
            if not langchain_docs:
                return "I couldn't find any relevant information for your query."
            
            context_text = ""
            for i, doc in enumerate(langchain_docs, 1):
                context_text += f"\n\nDOCUMENT {i}: {doc.metadata['title']} (Category: {doc.metadata['category']})\n{doc.page_content}"
            
            # Create and run the RAG chain with history
            rag_chain = self.create_rag_chain_with_history(user_role, chat_history)
            response = rag_chain.invoke({"context": context_text, "question": query})
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    async def direct_chat_completion(self, prompt: str) -> str:
        """Send a direct chat completion request using LangChain"""
        try:
            # Create a simple chain for direct completion
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant for FinSolve Technologies."),
                    ("human", "{input}")
                ])
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            return chain.invoke({"input": prompt})
            
        except Exception as e:
            print(f"Error in direct chat completion: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    @staticmethod
    def get_available_models():
        """Get list of available models from OpenRouter and Groq"""
        models = [
            # OpenRouter models - Claude
            {"id": "deepseek/deepseek-r1-0528:free", "name": "DeepSeek R1", "description": "Fastest and most accurate model", "provider": "OpenRouter"},
            {"id": "anthropic/claude-3-sonnet:free", "name": "Claude 3 Sonnet", "description": "Balanced Claude model", "provider": "OpenRouter"},
            {"id": "anthropic/claude-3-opus:free", "name": "Claude 3 Opus", "description": "Most powerful Claude model", "provider": "OpenRouter"},
            
            # OpenRouter models - Mistral
            {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B", "description": "Free tier Mistral model", "provider": "OpenRouter"},
            {"id": "mistralai/mistral-large:latest", "name": "Mistral Large", "description": "Powerful Mistral model", "provider": "OpenRouter"},
            
            # OpenRouter models - Meta
            {"id": "meta-llama/llama-3-8b-instruct:free", "name": "LLaMA-3 8B", "description": "Free tier LLaMA 3 model", "provider": "OpenRouter"},
            {"id": "meta-llama/llama-3-70b-instruct:free", "name": "LLaMA-3 70B", "description": "Powerful LLaMA 3 model", "provider": "OpenRouter"},
            
            # Groq models
            {"id": "groq/llama3-8b-8192", "name": "LLaMA-3 8B (Groq)", "description": "Fast inference with LLaMA-3", "provider": "Groq"},
            {"id": "groq/mixtral-8x7b-32768", "name": "Mixtral 8x7B (Groq)", "description": "Mixtral model with fast inference", "provider": "Groq"}
        ]
        return models 