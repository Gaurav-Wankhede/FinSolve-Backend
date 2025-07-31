# FinSolve Technologies RAG-RBAC Assistant

## Project Overview

This project implements a Retrieval Augmented Generation (RAG) based chatbot with Role-Based Access Control (RBAC) for FinSolve Technologies, a leading FinTech company. The system enables different departments to access role-specific information through a natural language interface, helping to reduce communication delays, address data access barriers, and offer secure, department-specific insights on demand.

### Challenge Background

FinSolve Technologies teams were facing delays in communication and difficulty accessing the right data at the right time, which led to inefficiencies and data silos between different departments. Tony Sharma, the Chief Innovation Officer, launched this digital transformation project to address these challenges through AI.

## Architecture

The backend system is built with a modern tech stack:

- **FastAPI**: High-performance web framework for building APIs
- **MongoDB**: Document database for storing user data and document embeddings
- **LangChain**: Framework for building LLM applications with RAG capabilities
- **OpenRouter/Groq**: API gateways for accessing various LLMs
- **Google AI API**: For document embedding generation

### Core Components

1. **Authentication & Authorization**
   - JWT-based authentication system
   - Role-based access control with predefined permission sets
   - Six different user roles with varying access levels

2. **Document Management**
   - Document uploading with automatic chunking and embedding
   - Role-specific document tagging
   - Semantic search across document collection

3. **RAG Implementation**
   - Document retrieval based on semantic similarity
   - Context augmentation with relevant documents
   - Response generation using various LLM models
   - Chat history integration for contextual responses

4. **API Endpoints**
   - User authentication and role management
   - Document upload and management
   - Chatbot query handling with role-based filtering
   - Model selection and testing

## Features

### Role-Based Access Control

The system supports six predefined roles, each with specific access permissions:

- **Finance Team**: Access to financial reports, marketing expenses, equipment costs, reimbursements
- **Marketing Team**: Access to campaign performance data, customer feedback, sales metrics
- **HR Team**: Access to employee data, attendance records, payroll, performance reviews
- **Engineering Department**: Access to technical architecture, development processes, operational guidelines
- **C-Level Executives**: Full access to all company data
- **Employee Level**: Access only to general company information (policies, events, FAQs)

### Retrieval Augmented Generation

The RAG system works through the following process:

1. User query is converted to embedding using Google AI API
2. Semantic search finds relevant document chunks accessible to the user's role
3. Retrieved documents are formatted into a context prompt
4. LLM generates a response based on the context, user's role, and chat history
5. Response is returned with source document references

### Multi-Model Support

The system supports multiple language models through OpenRouter and Groq:

- Claude models (Anthropic)
- DeepSeek models
- Llama models (Meta)
- Gemini models (Google)
- GPT models (OpenAI)

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB server (local or remote)
- API keys for:
  - Google AI API (embeddings)
  - OpenRouter (LLM access)
  - Groq (optional, for faster inference)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Gaurav-Wankhede/gaurav-wankhede-backend.git # Rename with finsolve-rag
   cd finsolve-rag/backend
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp env.template .env
   ```
   Edit the `.env` file with your configuration details.

### Running the Backend

Start the FastAPI server:

```
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000, and the interactive documentation at http://127.0.0.1:8000/docs.

## API Endpoints

### Authentication

- `POST /login`: Authenticate user and get JWT token
- `GET /user/me`: Get current user information

### Document Management

- `POST /upload`: Upload documents with role permissions
- `GET /api/v1/document/{document_id}`: Get document content by ID

### Chatbot

- `POST /api/v1/chat`: Send a query to the chatbot
- `GET /api/v1/history`: Get chat history for current user
- `GET /api/v1/models`: List available LLM models
- `POST /api/v1/test-model`: Test a specific LLM model

## System Flow

1. **Authentication**: User logs in and receives a JWT token with role information
2. **Query Submission**: User submits a natural language query through the chat interface
3. **Role-Based Retrieval**: System retrieves documents that match the query and are accessible to the user's role
4. **LLM Processing**: Selected documents are processed by the LLM to generate a response
5. **Response Delivery**: User receives the response with citations to source documents

## Technical Implementation Details

### Document Processing

Documents are processed in the following way:
1. Text extraction and cleaning
2. Chunking into manageable pieces (typically 512 tokens)
3. Embedding generation for each chunk using Google AI API
4. Storage in MongoDB with role access permissions
5. Indexing for efficient retrieval

### Semantic Search

The system uses cosine similarity between query embeddings and document chunk embeddings to find the most relevant information. The search algorithm:
1. Calculates similarity scores for each document chunk
2. Takes the maximum score for each document
3. Returns the top-k most similar documents

### LLM Chain

The LangChain implementation creates a chain that:
1. Prepares a system prompt with user role and retrieved context
2. Includes chat history for continuity
3. Formats the query for the LLM
4. Processes the response for delivery

## Security Considerations

- JWT-based authentication with secure key handling
- Role-based access control for all documents
- No direct access to source documents without proper authorization
- API rate limiting to prevent abuse
- Input validation to prevent injection attacks

## Future Enhancements

- **Fine-tuning**: Domain-specific model fine-tuning for FinTech content
- **Real-time Updates**: Integration with real-time data sources
- **Advanced Analytics**: Usage analytics and query pattern analysis
- **Multi-language Support**: Support for queries in multiple languages
- **Voice Interface**: Integration with speech-to-text for voice queries

## Contributors

- Peter Pandey (AI Engineer)
- FinSolve Technologies Team

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This README is part of the Codebasics Resume Project Challenge: "Build a RAG-Based Assistant for FinSolve Technologies"* 