# 🤖 AI-Powered Offer Letter Generation System

An intelligent system that generates personalized offer letters for candidates based on HR policies, salary bands, and employee data using a Retrieval Augmented Generation (RAG) architecture. The system features an interactive Streamlit interface for easy document management, employee data handling, and offer letter creation.

## ✨ Features

* **Interactive UI**: A user-friendly Streamlit application to manage the entire workflow.
* **Intelligent Document Processing**: Advanced chunking and processing for various document formats (`.pdf`, `.txt`, `.docx`).
* **Vector-based Search**: Utilizes FAISS for efficient similarity search across HR policy documents.
* **Dynamic Offer Letter Generation**: Creates personalized offer letters tailored to employee band, department, and specific policies.
* **HR Policy Chatbot**: An interactive chat interface to ask contextual questions about HR policies.
* **Data Management**: Easy upload and management of employee data via CSV files.
* **System Health & Monitoring**: Built-in status indicators and system diagnostics.

## ⚙️ How It Works

The system follows a RAG pipeline:

1. **Data Ingestion**: HR policy documents (e.g., leave, travel policies) and a sample offer letter are uploaded through the UI.
2. **Document Processing**: The `DocumentProcessor` chunks the documents into smaller, manageable pieces to prepare them for embedding.
3. **Vectorization**: The text chunks are converted into numerical vectors (embeddings) using a sentence transformer model.
4. **Indexing**: These embeddings are stored and indexed in a `FAISS` vector store for fast retrieval.
5. **Querying & Retrieval**: When generating an offer letter for a specific employee, the system retrieves the most relevant policy information from the vector store based on the employee's data (e.g., their band level).
6. **Generation**: The retrieved context, along with the employee's details, is fed into a Large Language Model (Gemini 1.5 Pro) which generates a personalized and contextually accurate offer letter.

## 🏍️ Architecture

<details> <summary>Click to expand diagram</summary>

```mermaid
graph TD
    subgraph Input_Layer
        A[📄 HR Policies & Templates]
    end

    subgraph Processing_Core
        B[📝 Document Processor]
        C[📚 Vector Store (FAISS)]
        E[🤖 LLM Interface - Gemini]
    end

    subgraph Output_Layer
        D[✨ Embeddings & Metadata]
        F[📧 Generated Offer Letters]
    end

    A --> B
    B --> C
    C --> E
    C --> D
    E --> F

    style A fill:#D6EAF8,stroke:#3498DB
    style B fill:#D1F2EB,stroke:#1ABC9C
    style C fill:#FDEDEC,stroke:#E74C3C
    style E fill:#FDEBD0,stroke:#F39C12
    style D fill:#E8DAEF,stroke:#8E44AD
    style F fill:#D5F5E3,stroke:#2ECC71
```

</details>

## 🛠️ Technology Stack

* **Backend**: Python
* **Web Framework**: Streamlit
* **LLM**: Google Gemini 1.5 Pro
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **Embeddings**: `text-embedding-ada-002` (or other sentence transformers)
* **Data Handling**: Pandas

## 📁 Project Structure

```
offer_letter_system/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
├── config/
│   └── settings.py                # Configuration settings
├── src/
│   ├── document_processor.py      # Document parsing & chunking
│   ├── vector_store.py           # FAISS vector database operations
│   ├── llm_interface.py          # Gemini integration
│   ├── offer_generator.py        # Offer letter generation logic
│   └── utils.py                  # Utility functions
├── data/
│   ├── documents/                # Input HR policy documents
│   └── employee_data.csv         # Sample employee metadata
└── README.md
```

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* A Google Gemini API Key

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/codeMaestro78/AI-Powered-Offer-Letter-Generation-System.git
   cd offer_letter_system
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the root directory of the project.
2. Add your Google Gemini API key to the file:

   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

### Usage

1. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3. **Using the App:**

   * **Step 1: Upload Documents**: In the sidebar, upload your HR policy documents (PDF, TXT, DOCX) and the employee data CSV file.
   * **Step 2: Process Documents**: Click the `Process Documents` button. The system will parse the documents, create vector embeddings, and build the FAISS index.
   * **Step 3: Generate Offer Letters**: Once processing is complete, select an employee from the dropdown menu in the "Offer Letter Generation" tab. The system will automatically populate their details.
   * **Step 4: Customize and Generate**: Click `✨ Generate Offer Letter` to create the personalized letter.
   * **Step 5: Chat with Policies**: Use the "HR Policy Chatbot" tab to ask questions about the uploaded documents.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or improvements.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Open a Pull Request.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
