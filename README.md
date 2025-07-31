# AI-Powered Offer Letter Generation System

An intelligent system that generates personalized offer letters for candidates based on HR policies, salary bands, and employee data using RAG (Retrieval Augmented Generation) architecture.

## 🚀 Features

- **Intelligent Document Processing**: Advanced chunking strategies for HR policy documents
- **Vector-based Search**: Efficient retrieval of relevant policy information
- **Personalized Offer Letters**: Generated based on employee band, department, and policies
- **Interactive Chat Interface**: Ask questions about HR policies and get contextual answers
- **Band-specific Policies**: Automatic application of correct benefits based on employee level
- **Multi-format Support**: Process PDF, TXT, and CSV files

## 🏗️ Architecture
|─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector Store  │    │   LLM Interface │
│   Processor     │───▶│   (FAISS)      │───▶│   (GPT-4)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                        │                           │
▼                        ▼                           ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HR Policies   │    │   Embeddings    │    │   Generated     │
│   & Templates   │    │   & Metadata    │    │   Offer Letters │
└─────────────────┘    └─────────────────┘    └─────────────────┘