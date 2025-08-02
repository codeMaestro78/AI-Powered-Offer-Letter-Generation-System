import streamlit as st
import pandas as pd
from datetime import datetime
import os
import tempfile
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
import traceback
import time
from typing import List, Dict, Any
from pathlib import Path
import re
from typing import Optional, Dict, Any, List

from src.document_processor import DocumentProcessor
from src.cached_loaders import get_vector_store
from src.offer_generator import OfferGenerator
from src.utils import Utils
from config.settings import Config

st.set_page_config(
    page_title="AI Offer Letter Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OfferLetterApp:
    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor()
        self.vector_store = get_vector_store()
        self.offer_generator = None
        
        # Initialize session state
        self._initialize_session_state()
        
        # Load employee data from session state if available
        self.employee_df = st.session_state.get('employee_df', pd.DataFrame())

    def _initialize_session_state(self):
        """Initialize all session state variables"""
        session_defaults = {
            'documents_processed': False,
            'employee_data_loaded': False,
            'chat_history': [],
            'generated_offer': None,
            'processing_status': None,
            'employee_df': pd.DataFrame(),
            'document_stats': {},
            'last_selected_employee': None
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def run(self):
        """Main application runner with modern UI"""
        # Add custom CSS first
        self._add_custom_css()
        
        # Modern header with gradient and animation
        st.markdown("""
        <div class="main-header fade-in">
            <h1>🚀 AI-Powered Offer Letter Generator</h1>
            <p>Generate professional offer letters with intelligent HR policy integration</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for configuration
        self.render_sidebar()
        
        # Main content area with modern styling
        if st.session_state.documents_processed and st.session_state.employee_data_loaded:
            self.render_main_interface()
        else:
            self.render_setup_interface()

    def _add_custom_css(self):
        """Add modern custom CSS styling with cool effects"""
        st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #2c3e50;
    }

    #MainMenu, footer, .stDeployButton {visibility: hidden;}

    .main-header {
        background: #2c3e50;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(44, 62, 80, 0.3);
        border: 1px solid #34495e;
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .main-header p {
        color: #ecf0f1;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    .modern-card, .employee-card, .chat-message {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    .stError {
        background: #fff5f5;
        border: 1px solid #f5c2c7;
        border-radius: 6px;
        padding: 1rem;
        color: #b02a37;
    }

    /* Info/Alert message styling */
    .stInfo, [data-testid="stAlert"] {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px !important;
        color: #495057 !important;
    }

    .stInfo > div, [data-testid="stAlert"] > div {
        background: #f8f9fa !important;
        color: #495057 !important;
    }

    /* Target all alert types */
    .stSuccess, .stWarning, .stInfo, .stError {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        color: #495057 !important;
    }

    /* Alert icons */
    .stInfo svg, .stSuccess svg, .stWarning svg {
        color: #6c757d !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #f8f9fa !important;
        border-bottom: 1px solid #dee2e6 !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
        border-bottom: none !important;
        border-radius: 6px 6px 0 0 !important;
        margin-right: 2px !important;
        padding: 0.5rem 1rem !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: #e9ecef !important;
        color: #343a40 !important;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #212529 !important;
        border-bottom: 1px solid #ffffff !important;
        font-weight: 600 !important;
    }

    /* Tab content area */
    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-top: none !important;
        padding: 1.5rem !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* Chat input styling */
    .stChatInput {
        background: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px !important;
    }

    .stChatInput > div {
        background: #ffffff !important;
        border: 1px solid #dee2e6 !important;
    }

    /* Chat input placeholder */
    .stChatInput input::placeholder {
        color: #6c757d !important;
    }

    /* Tooltip and popup styling */
    [data-baseweb="tooltip"] {
        background: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
    }

    /* Chat message containers */
    .stChatMessage {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px !important;
    }

    /* Any blue backgrounds in chat */
    .stChatMessage [style*="background-color: rgb(28, 131, 225)"],
    .stChatMessage [style*="background: rgb(28, 131, 225)"] {
        background: #f8f9fa !important;
        color: #495057 !important;
    }

    /* Override any remaining blue chat elements */
    [class*="chat"] [style*="blue"],
    [class*="Chat"] [style*="blue"] {
        background: #f8f9fa !important;
        color: #495057 !important;
    }

    /* Comprehensive chat message styling */
    .stChatMessage, [data-testid="stChatMessage"] {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }

    /* Chat message content */
    .stChatMessage > div, [data-testid="stChatMessage"] > div {
        background: #f8f9fa !important;
        color: #495057 !important;
    }

    /* User chat messages */
    .stChatMessage[data-testid="user-message"],
    [data-testid="stChatMessage"][data-testid="user-message"] {
        background: #e9ecef !important;
        border: 1px solid #adb5bd !important;
    }

    /* Assistant chat messages */
    .stChatMessage[data-testid="assistant-message"],
    [data-testid="stChatMessage"][data-testid="assistant-message"] {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }

    /* Chat input container */
    .stChatInput, [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }

    /* Chat input field */
    .stChatInput input, [data-testid="stChatInput"] input {
        background: #ffffff !important;
        color: #495057 !important;
        border: none !important;
    }

    /* Chat input bottom container */
    .stChatInput > div, [data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: none !important;
    }

    /* Force override all blue chat backgrounds */
    [style*="background-color: rgb(28, 131, 225)"],
    [style*="background: rgb(28, 131, 225)"],
    [style*="background-color: #1c83e1"],
    [style*="background: #1c83e1"],
    [style*="background-color: blue"],
    [style*="background: blue"] {
        background: #f8f9fa !important;
        background-color: #f8f9fa !important;
        color: #495057 !important;
    }

    /* Target specific chat message classes */
    .css-1c7y2kd, .css-1ec6rqw, .css-16huue1 {
        background: #f8f9fa !important;
        color: #495057 !important;
    }

    .status-success {
        background: #e6f4ea;
        color: #1e7e34;
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
    }

    .status-warning {
        background: #fff8e1;
        color: #856404;
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
    }

    .status-error {
        background: #fdecea;
        color: #b02a37;
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
    }

    .stButton > button {
        background: #2c3e50;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: background 0.3s ease;
    }

    .stButton > button:hover {
        background: #1a252f;
    }

    .css-1d391kg {
        background: #f8f9fa !important;
        border-right: 1px solid #dee2e6 !important;
    }

    /* Sidebar header styling */
    .css-1d391kg .css-1v0mbdj {
        background: #f8f9fa !important;
    }

    /* Main container and header */
    .css-18e3th9 {
        background: #f8f9fa !important;
    }

    /* Top navigation bar */
    .css-1rs6os {
        background: #f8f9fa !important;
        border-bottom: 1px solid #dee2e6 !important;
    }

    /* Streamlit header */
    .css-1avcm0n {
        background: #f8f9fa !important;
    }

    /* Remove blue header background */
    header[data-testid="stHeader"] {
        background: #f8f9fa !important;
        height: 3rem !important;
    }

    /* Navigation elements */
    .css-1v0mbdj {
        background: #f8f9fa !important;
    }

    /* Comprehensive header styling - target all possible header classes */
    .css-k1vhr4, .css-18e3th9, .css-1d391kg, .css-1v0mbdj,
    .css-1rs6os, .css-1avcm0n, .css-k1vhr4 {
        background: #f8f9fa !important;
    }

    /* Target Streamlit's main header container */
    .main .block-container {
        background: #f4f6f8 !important;
    }

    /* Target all header-related elements */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    .css-1kyxreq,
    .css-12ttj6m,
    .css-1v0mbdj,
    .css-18e3th9 {
        background: #f8f9fa !important;
        background-color: #f8f9fa !important;
    }

    /* Force override any blue backgrounds */
    * {
        --primary-color: #495057 !important;
        --background-color: #f4f6f8 !important;
        --secondary-background-color: #f8f9fa !important;
    }

    /* Target the main app container */
    .stApp > header {
        background: #f8f9fa !important;
        background-color: #f8f9fa !important;
    }

    /* Override any remaining blue elements */
    div[class*="css-"][style*="background"] {
        background: #f8f9fa !important;
    }

    .user-message, .assistant-message {
        background: #fafafa;
        border-left: 3px solid #d6d8db;
    }

    .stProgress > div > div > div > div {
        background: #adb5bd;
        border-radius: 10px;
    }

    .stFileUploader {
        background: #f8f9fa !important;
        border: 2px dashed #adb5bd !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        text-align: center !important;
    }

    .stFileUploader:hover {
        background: #e9ecef !important;
        border-color: #6c757d !important;
    }

    .stFileUploader > div {
        background: #f8f9fa !important;
        border: 2px dashed #adb5bd !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        color: #495057 !important;
    }

    .stFileUploader > div:hover {
        background: #e9ecef !important;
        border-color: #6c757d !important;
    }

    .stFileUploader label {
        color: #495057 !important;
    }

    .stFileUploader button {
        background: #495057 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }

    .stFileUploader button:hover {
        background: #343a40 !important;
    }

    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #34495e !important;
        background: #34495e !important;
        color: #ffffff !important;
    }

    .stSelectbox > div > div:focus {
        border-color: #5d6d7e !important;
        box-shadow: 0 0 0 2px rgba(93, 109, 126, 0.3) !important;
    }

    /* Selectbox text and options */
    .stSelectbox label {
        color: #ecf0f1 !important;
        font-weight: 500 !important;
    }

    .stSelectbox > div > div > div {
        color: #ffffff !important;
    }

    /* Dropdown arrow */
    .stSelectbox > div > div > div > div > svg {
        color: #ecf0f1 !important;
    }

    div[data-testid="stSpinner"],
    div[data-testid="stStatusWidget"] {
        background-color: transparent !important;
        backdrop-filter: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


    def render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.header("📋 Configuration")
        
        # Document upload section
        st.sidebar.subheader("1. Upload HR Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Upload HR Policy Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload HR policies, travel policies, leave policies, etc.",
            key="hr_documents"
        )
        
        # Employee data upload
        st.sidebar.subheader("2. Upload Employee Data")
        employee_file = st.sidebar.file_uploader(
            "Upload Employee CSV",
            type=['csv'],
            help="Upload CSV file with employee information (must include 'Employee Name' column)",
            key="employee_data"
        )
        
        # Process documents button
        col1, col2 = st.sidebar.columns(2)
        with col1:
            process_btn = st.button("🔄 Process", type="primary")
        with col2:
            clear_btn = st.button("🗑️ Clear", help="Clear all processed data")
        
        if process_btn:
            if uploaded_files and employee_file:
                self.process_documents(uploaded_files, employee_file)
            else:
                st.sidebar.error("Please upload both HR documents and employee data")
        
        if clear_btn:
            self._clear_all_data()
        
        # Status section
        st.sidebar.subheader("📊 Status")
        self._render_status_indicators()
        
        # Statistics section
        if st.session_state.documents_processed:
            self._render_document_stats()

    def _render_status_indicators(self):
        """Render status indicators in sidebar"""
        docs_status = "✅" if st.session_state.documents_processed else "❌"
        employee_status = "✅" if st.session_state.employee_data_loaded else "❌"
        
        st.sidebar.markdown(f"**Documents Processed:** {docs_status}")
        st.sidebar.markdown(f"**Employee Data Loaded:** {employee_status}")
        
        if st.session_state.processing_status:
            st.sidebar.info(st.session_state.processing_status)

    def _render_document_stats(self):
        """Render document processing statistics"""
        if st.session_state.document_stats:
            st.sidebar.subheader("📈 Document Stats")
            stats = st.session_state.document_stats
            st.sidebar.metric("Documents Processed", stats.get('total_documents', 0))
            st.sidebar.metric("Total Chunks", stats.get('total_chunks', 0))
            st.sidebar.metric("Employees", len(self.employee_df))

    def _clear_all_data(self):
        """Clear all processed data and reset session state"""
        for key in ['documents_processed', 'employee_data_loaded', 'chat_history', 
                   'generated_offer', 'processing_status', 'employee_df', 'document_stats']:
            st.session_state[key] = [] if key == 'chat_history' else (
                pd.DataFrame() if key == 'employee_df' else (
                    {} if key == 'document_stats' else (
                        False if key.endswith('_processed') or key.endswith('_loaded') else None
                    )
                )
            )
        
        # The vector store is cached, so we just clear its internal state
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            self.vector_store.clear()
        
        self.employee_df = pd.DataFrame()
        self.offer_generator = None
        st.sidebar.success("✅ All data cleared!")
        st.rerun()

    def process_documents(self, uploaded_files: List, employee_file) -> None:
        """Process uploaded documents and employee data"""
        progress_container = st.sidebar.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        try:
            documents = {}
            total_files = len(uploaded_files)
            
            # Process HR documents
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / (total_files + 2))
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(file.getbuffer())
                    tmp_path = tmp_file.name
                
                try:
                    if file.type == "application/pdf":
                        documents[file.name] = self.doc_processor.process_pdf(tmp_path)
                    elif file.type == "text/plain":
                        documents[file.name] = str(file.read(), "utf-8")
                    elif file.name.endswith('.docx'):
                        documents[file.name] = self.doc_processor.process_docx(tmp_path)
                    else:
                        st.warning(f"Unsupported file type: {file.name}")
                        continue
                finally:
                    os.unlink(tmp_path)
            
            # Process employee data
            status_text.text("Processing employee data...")
            progress_bar.progress((total_files + 1) / (total_files + 2))
            
            try:
                # Reset file pointer to beginning before reading
                employee_file.seek(0)
                self.employee_df = pd.read_csv(employee_file)
                
                if self.employee_df.empty:
                    st.error("The employee CSV file is empty. Please upload a file with employee data.")
                    return
                    
                self._validate_employee_data()
                
            except pd.errors.EmptyDataError:
                st.error("The employee CSV file is empty or contains no valid data.")
                return
            except pd.errors.ParserError as e:
                st.error(f"Error parsing employee CSV file: {str(e)}")
                st.info("Please ensure your CSV file has proper formatting with headers.")
                return
            except Exception as e:
                st.error(f"Error reading employee CSV: {str(e)}")
                st.info("Please check that your file is a valid CSV format.")
                return
            
            # Create document chunks
            status_text.text("Creating document chunks...")
            chunks = self.doc_processor.intelligent_chunk_documents(documents)
            
            if not chunks:
                st.error("No valid chunks created from documents")
                return
            
            # Create embeddings
            status_text.text("Creating embeddings...")
            self.vector_store.create_embeddings(chunks)
            
            # Initialize offer generator
            self.offer_generator = OfferGenerator(self.vector_store)
            
            # Update session state
            st.session_state.documents_processed = True
            st.session_state.employee_data_loaded = True
            st.session_state.employee_df = self.employee_df
            st.session_state.document_stats = {
                'total_documents': len(documents),
                'total_chunks': len(chunks),
                'total_employees': len(self.employee_df)
            }
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            st.sidebar.success(f"✅ Processed {len(chunks)} document chunks and {len(self.employee_df)} employee records")
            
            # Show success message and rerun to show main interface
            st.success("✨ Setup completed successfully! The main interface is now loading...")
            time.sleep(1)  # Brief pause to show success message
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            st.error("Stack trace:")
            st.code(traceback.format_exc())
        finally:
            progress_bar.empty()
            status_text.empty()

    def _validate_employee_data(self) -> None:
        """Validate employee data structure"""
        required_columns = ['Employee Name']
        missing_columns = [col for col in required_columns if col not in self.employee_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if self.employee_df.empty:
            raise ValueError("Employee data file is empty")
        
        # Log available columns for debugging
        st.info(f"Loaded {len(self.employee_df)} employee records with columns: {list(self.employee_df.columns)}")

    def render_sidebar(self):
        """Render modern sidebar with enhanced metrics and navigation"""
        with st.sidebar:
            # Modern sidebar header
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 8px; margin-bottom: 2rem; border: 1px solid #dee2e6;">
                <h2 style="color: #343a40; margin: 0; font-size: 1.5rem;">🎛️ Control Panel</h2>
                <p style="color: #6c757d; margin: 0.5rem 0 0 0; font-size: 0.9rem;">System Overview & Navigation</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Modern metrics cards
            self._render_sidebar_metrics()
            
            # System health indicator
            self._render_system_health()
            
            st.markdown("---")

    def _render_sidebar_metrics(self):
        """Render modern metrics in sidebar"""
        # Documents metric
        docs_count = len(st.session_state.get('uploaded_documents', []))
        chunks_count = st.session_state.get('document_stats', {}).get('total_chunks', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background: #ffffff; margin-bottom: 1rem; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 1.8rem; font-weight: 600; color: #343a40; margin: 0;">{docs_count}</div>
                <div style="font-size: 0.8rem; color: #6c757d; margin: 0.3rem 0 0 0;">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            employees_count = len(self.employee_df) if not self.employee_df.empty else 0
            st.markdown(f"""
            <div style="background: #ffffff; margin-bottom: 1rem; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 1.8rem; font-weight: 600; color: #343a40; margin: 0;">{employees_count}</div>
                <div style="font-size: 0.8rem; color: #6c757d; margin: 0.3rem 0 0 0;">Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Chunks metric (full width)
        if chunks_count > 0:
            st.markdown(f"""
            <div style="background: #ffffff; margin-bottom: 1rem; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 1.8rem; font-weight: 600; color: #343a40; margin: 0;">{chunks_count}</div>
                <div style="font-size: 0.8rem; color: #6c757d; margin: 0.3rem 0 0 0;">AI Knowledge Chunks</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_system_health(self):
        """Render system health indicator"""
        if st.session_state.documents_processed and st.session_state.employee_data_loaded:
            status = "🟢 Fully Operational"
            color = "#4CAF50"
        elif st.session_state.documents_processed or st.session_state.employee_data_loaded:
            status = "🟡 Partially Ready"
            color = "#ff9800"
        else:
            status = "🔴 Setup Required"
            color = "#f44336"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #ffffff; border-radius: 8px; margin: 1rem 0; border: 1px solid #dee2e6; border-left: 4px solid {color};">
            <div style="font-weight: 600; color: {color}; font-size: 1rem;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

    def render_main_interface(self):
        """Render main application interface"""
        if self.offer_generator is None:
            self.offer_generator = OfferGenerator(self.vector_store)
        
        # Modern setup tabs with enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📄 Document Processing", "👥 Employee Data Management"])
        
        with tab1:
            self.render_chat_interface()
        with tab1:
            self.render_document_setup()
        
        with tab2:
            self.render_employee_data_setup()
        
        # Add completion status
        if st.session_state.documents_processed and st.session_state.employee_data_loaded:
            st.markdown("""
            <div class="modern-card fade-in" style="background: linear-gradient(135deg, #e8f5e8, #f0f8f0); text-align: center; margin-top: 2rem;">
                <h2 style="color: #2e7d32; margin: 0 0 1rem 0;">🎉 Setup Complete!</h2>
                <p style="color: #1b5e20; margin: 0 0 1rem 0;">Your AI system is ready to generate offer letters and answer HR policy questions.</p>
                <div style="background: #4CAF50; color: white; padding: 0.7rem 1.5rem; border-radius: 25px; display: inline-block; font-weight: 600;">
                    ✨ All Systems Operational
                </div>
            </div>
            """, unsafe_allow_html=True)

    def render_setup_interface(self):
        """Render modern setup interface for initial configuration"""
        st.markdown("""
        <div class="modern-card fade-in">
            <h2 style="color: #333; margin: 0 0 1rem 0;">🚀 System Setup</h2>
            <p style="color: #666; margin: 0;">Welcome! Let's get your AI-powered offer letter system ready with a few simple steps.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Modern setup progress with visual indicators
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.documents_processed:
                st.markdown("""
                <div class="modern-card" style="background: linear-gradient(135deg, #e8f5e8, #f0f8f0); border-left: 4px solid #4CAF50;">
                    <h3 style="color: #2e7d32; margin: 0 0 0.5rem 0;">✅ Step 1: Documents</h3>
                    <p style="color: #1b5e20; margin: 0;">HR policies processed and ready!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="modern-card" style="background: linear-gradient(135deg, #fff3e0, #fafafa); border-left: 4px solid #ff9800;">
                    <h3 style="color: #f57c00; margin: 0 0 0.5rem 0;">📄 Step 1: Documents</h3>
                    <p style="color: #e65100; margin: 0;">Upload and process HR policy documents</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.employee_data_loaded:
                st.markdown("""
                <div class="modern-card" style="background: linear-gradient(135deg, #e8f5e8, #f0f8f0); border-left: 4px solid #4CAF50;">
                    <h3 style="color: #2e7d32; margin: 0 0 0.5rem 0;">✅ Step 2: Employee Data</h3>
                    <p style="color: #1b5e20; margin: 0;">Employee information loaded successfully!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="modern-card" style="background: linear-gradient(135deg, #fff3e0, #fafafa); border-left: 4px solid #ff9800;">
                    <h3 style="color: #f57c00; margin: 0 0 0.5rem 0;">👥 Step 2: Employee Data</h3>
                    <p style="color: #e65100; margin: 0;">Load employee data from CSV file</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Modern setup tabs with enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📄 Document Processing", "👥 Employee Data Management"])
        
        with tab1:
            self.render_document_setup()
        
        with tab2:
            self.render_employee_data_setup()
        
        # Add completion status
        if st.session_state.documents_processed and st.session_state.employee_data_loaded:
            st.markdown("""
            <div class="modern-card fade-in" style="background: linear-gradient(135deg, #e8f5e8, #f0f8f0); text-align: center; margin-top: 2rem;">
                <h2 style="color: #2e7d32; margin: 0 0 1rem 0;">🎉 Setup Complete!</h2>
                <p style="color: #1b5e20; margin: 0 0 1rem 0;">Your AI system is ready to generate offer letters and answer HR policy questions.</p>
                <div style="background: #4CAF50; color: white; padding: 0.7rem 1.5rem; border-radius: 25px; display: inline-block; font-weight: 600;">
                    ✨ All Systems Operational
                </div>
            </div>
            """, unsafe_allow_html=True)

    def render_document_setup(self):
        """Render document upload and processing interface"""
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #333; margin: 0 0 1rem 0;">📄 Document Processing</h3>
            <p style="color: #666; margin: 0 0 1rem 0;">Upload your HR policy documents to power the AI system.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader with modern styling
        uploaded_files = st.file_uploader(
            "Choose HR Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files containing HR policies, benefits, travel policies, etc."
        )
        
        if uploaded_files:
            st.markdown(f"""
            <div class="modern-card" style="background: #e9ecef; border: 1px solid #adb5bd;">
                <h4 style="color: #495057; margin: 0 0 0.5rem 0;">📁 Files Ready for Processing</h4>
                <p style="color: #6c757d; margin: 0;">Selected {len(uploaded_files)} document(s)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show file list
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size:,} bytes)")
        
        # Store documents in session state for later processing
        if uploaded_files:
            st.session_state['uploaded_documents'] = uploaded_files
            
            # Check if employee data is also available
            employee_file = st.session_state.get('employee_file')
            if employee_file:
                if st.button("✨ Process All Data", type="primary", use_container_width=True, key="process_docs_tab"):
                    self.process_documents(uploaded_files, employee_file)
            else:
                st.info("👥 Please also upload employee data in the Employee Data tab, then click 'Process All Data' there.")

    def render_employee_data_setup(self):
        """Render employee data upload interface"""
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #333; margin: 0 0 1rem 0;">👥 Employee Data Management</h3>
            <p style="color: #666; margin: 0 0 1rem 0;">Upload employee information to personalize offer letters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Employee data uploader
        employee_file = st.file_uploader(
            "Choose Employee CSV File",
            type=['csv'],
            help="Upload a CSV file with employee information including Name, Department, Band, etc."
        )
        
        if employee_file:
            st.markdown("""
            <div class="modern-card" style="background: #e9ecef; border: 1px solid #adb5bd;">
                <h4 style="color: #495057; margin: 0 0 0.5rem 0;">📈 Employee Data Ready</h4>
                <p style="color: #6c757d; margin: 0;">CSV file loaded successfully</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Preview employee data
            try:
                # Reset file pointer to beginning
                employee_file.seek(0)
                preview_df = pd.read_csv(employee_file)
                
                if preview_df.empty:
                    st.warning("The CSV file appears to be empty.")
                else:
                    st.write(f"**Preview:** {len(preview_df)} employees found")
                    st.write(f"**Columns:** {', '.join(preview_df.columns.tolist())}")
                    st.dataframe(preview_df.head(), use_container_width=True)
                    
                # Reset file pointer again for later processing
                employee_file.seek(0)
                
            except pd.errors.EmptyDataError:
                st.error("The CSV file is empty or has no valid data.")
            except pd.errors.ParserError as e:
                st.error(f"Error parsing CSV file: {str(e)}. Please check the file format.")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
                st.info("Please ensure your file is a valid CSV with proper headers.")
        
        # Sample CSV template
        with st.expander("📝 Sample CSV Format", expanded=False):
            st.markdown("""
            <div class="modern-card" style="background: linear-gradient(135deg, #fff3e0, #fafafa);">
                <h4 style="color: #f57c00; margin: 0 0 0.5rem 0;">📊 Required Columns</h4>
                <p style="color: #e65100; margin: 0;">Your CSV should include these columns:</p>
            </div>
            """, unsafe_allow_html=True)
            
            sample_data = {
                'Employee Name': ['John Doe', 'Jane Smith'],
                'Department': ['Engineering', 'Marketing'],
                'Band': ['L4', 'L3'],
                'Location': ['Bangalore', 'Mumbai'],
                'Total CTC (INR)': [1200000, 900000],
                'Email': ['john.doe@company.com', 'jane.smith@company.com']
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
            
            # Download sample CSV
            csv_data = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=csv_data,
                file_name="employee_template.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Combined processing when both are available
        uploaded_docs = st.session_state.get('uploaded_documents', [])
        if uploaded_docs and employee_file:
            if st.button("✨ Process All Data", type="primary", use_container_width=True, key="process_employee_tab"):
                self.process_documents(uploaded_docs, employee_file)
        elif not uploaded_docs and employee_file:
            st.info("📄 Please upload documents in the Document Processing tab first.")
        elif uploaded_docs and not employee_file:
            st.info("👥 Please upload employee CSV file to complete setup.")
        
        # Store files in session state
        if employee_file:
            st.session_state['employee_file'] = employee_file

    def render_chat_interface(self):
        """Render functional chat interface with memory"""
        st.markdown("""
        <div class="modern-card fade-in">
            <h2 style="color: #333; margin: 0 0 1rem 0;">💬 HR Policy Chat</h2>
            <p style="color: #666; margin: 0;">Ask questions about HR policies, benefits, travel policies, and more! I remember our conversation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize persistent chat memory if not exists
        if 'chat_memory' not in st.session_state:
            st.session_state.chat_memory = {
                'conversation_context': [],
                'topics_discussed': set(),
                'user_preferences': {},
                'session_summary': ""
            }
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Memory status removed for cleaner interface
        
        # Chat input
        if prompt := st.chat_input("Ask about HR policies, benefits, or employee information..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate context-aware response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Thinking with context..."):
                    try:
                        if self.offer_generator is None:
                            self.offer_generator = OfferGenerator(self.vector_store)
                        
                        # Get context-aware response with chat history
                        response = self._generate_contextual_response(prompt)
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Update memory
                        self._update_chat_memory(prompt, response)
                        
                    except Exception as e:
                        error_msg = f"😞 Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Chat controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🧠 Reset Memory", key="reset_memory"):
                st.session_state.chat_memory = {
                    'conversation_context': [],
                    'topics_discussed': set(),
                    'user_preferences': {},
                    'session_summary': ""
                }
                st.success("Memory reset!")
                st.rerun()

    def render_offer_letter_generator(self):
        """Render functional offer letter generator"""
        st.markdown("""
        <div class="modern-card fade-in">
            <h2 style="color: #333; margin: 0 0 1rem 0;">📝 Offer Letter Generator</h2>
            <p style="color: #666; margin: 0;">Generate personalized offer letters for your employees.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if self.employee_df.empty:
            st.warning("👥 No employee data available. Please upload employee CSV file first.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Select Employee:**")
            
            # Employee selection
            employee_names = sorted(self.employee_df['Employee Name'].tolist())
            selected_employee = st.selectbox(
                "Choose Employee",
                options=employee_names,
                help="Select an employee to generate offer letter",
                key="offer_employee_select"
            )
            
            if selected_employee:
                # Find employee data
                employee_data = self.employee_df[self.employee_df['Employee Name'] == selected_employee].iloc[0].to_dict()
                
                # Display employee info
                st.markdown("**Employee Details:**")
                st.write(f"👤 **Name:** {employee_data.get('Employee Name', 'N/A')}")
                st.write(f"🏢 **Department:** {employee_data.get('Department', 'N/A')}")
                st.write(f"📈 **Band:** {employee_data.get('Band', 'N/A')}")
                st.write(f"📍 **Location:** {employee_data.get('Location', 'N/A')}")
                
                if 'Total CTC (INR)' in employee_data:
                    ctc = employee_data.get('Total CTC (INR)', 0)
                    st.write(f"💰 **Total CTC:** ₹{ctc:,.2f}")
                
                # Generate button
                if st.button("🚀 Generate Offer Letter", type="primary", use_container_width=True, key="generate_offer"):
                    with st.spinner(f"Generating offer letter for {selected_employee}..."):
                        try:
                            if self.offer_generator is None:
                                self.offer_generator = OfferGenerator(self.vector_store)
                            
                            offer_letter = self.offer_generator.generate_offer_letter(
                                selected_employee, 
                                employee_data
                            )
                            
                            st.session_state.generated_offer = offer_letter
                            st.success(f"✅ Offer letter generated successfully for {selected_employee}!")
                            
                        except Exception as e:
                            st.error(f"Error generating offer letter: {str(e)}")
        
        with col2:
            if 'generated_offer' in st.session_state and st.session_state.generated_offer:
                st.markdown("### Generated Offer Letter")
                st.markdown("---")
                st.markdown(st.session_state.generated_offer)
                
                # Download buttons
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.download_button(
                        label="📥 Download as Markdown",
                        data=st.session_state.generated_offer,
                        file_name=f"offer_letter_{selected_employee.replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key="download_md"
                    )
                with col2b:
                    plain_text = st.session_state.generated_offer.replace('#', '').replace('*', '')
                    st.download_button(
                        label="📄 Download as Text",
                        data=plain_text,
                        file_name=f"offer_letter_{selected_employee.replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_txt"
                    )
                with col2c:
                    # Create PDF in memory
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elements = []
                    
                    # Add title style
                    title_style = ParagraphStyle(
                        'Title',
                        parent=styles['Heading1'],
                        fontSize=14,
                        spaceAfter=12,
                        alignment=TA_CENTER
                    )
                    
                    # Add normal text style
                    normal_style = ParagraphStyle(
                        'Normal',
                        parent=styles['Normal'],
                        fontSize=10,
                        leading=14,
                        spaceAfter=6
                    )
                    
                    # Process the offer letter text
                    lines = st.session_state.generated_offer.split('\n')
                    for line in lines:
                        if line.startswith('##'):
                            # Handle headers
                            elements.append(Paragraph(line.lstrip('#').strip(), title_style))
                        elif line.strip():
                            # Handle normal text
                            elements.append(Paragraph(line, normal_style))
                        elements.append(Spacer(1, 12))
                    
                    # Build the PDF
                    doc.build(elements)
                    pdf_buffer.seek(0)
                    
                    st.download_button(
                        label="📑 Download as PDF",
                        data=pdf_buffer,
                        file_name=f"offer_letter_{selected_employee.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf"
                    )
            else:
                st.info("👈 Select an employee and click 'Generate Offer Letter' to see the result here")

    def render_system_analytics(self):
        """Render system analytics placeholder"""
        st.markdown("""
        <div class="modern-card fade-in">
            <h2 style="color: #333; margin: 0 0 1rem 0;">📊 System Analytics</h2>
            <p style="color: #666; margin: 0;">View system performance metrics and usage statistics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show basic metrics if available
        if st.session_state.document_stats:
            stats = st.session_state.document_stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("AI Chunks", stats.get('total_chunks', 0))
            with col3:
                st.metric("Employees", stats.get('total_employees', 0))
        else:
            st.info("📈 Analytics will be available once documents are processed.")

    def render_main_interface(self):
        """Render modern main interface when system is ready"""
        st.markdown("""
        <div class="modern-card fade-in" style="text-align: center; background: #f8f9fa; border: 1px solid #dee2e6;">
            <h2 style="margin: 0 0 1rem 0; color: #343a40;">🎆 System Ready!</h2>
            <p style="margin: 0; color: #6c757d;">All systems operational. Choose your next action below.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Modern navigation cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 HR Policy Chat", use_container_width=True, help="Chat with AI about HR policies", key="nav_chat"):
                st.session_state.main_tab = "chat"
        
        with col2:
            if st.button("📝 Generate Offer Letter", use_container_width=True, help="Create personalized offer letters", key="nav_offer"):
                st.session_state.main_tab = "offer"
        
        with col3:
            if st.button("📊 System Analytics", use_container_width=True, help="View system performance metrics", key="nav_analytics"):
                st.session_state.main_tab = "analytics"
        
        # Render selected tab content with modern styling
        if hasattr(st.session_state, 'main_tab'):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.session_state.main_tab == "chat":
                self.render_chat_interface()
            elif st.session_state.main_tab == "offer":
                self.render_offer_letter_generator()
            elif st.session_state.main_tab == "analytics":
                self.render_system_analytics()
        else:
            # Default welcome message
            st.markdown("""
            <div class="modern-card slide-in" style="text-align: center; margin-top: 2rem;">
                <h3 style="color: #333; margin: 0 0 1rem 0;">🚀 Ready to Get Started?</h3>
                <p style="color: #666; margin: 0;">Click on any of the options above to begin using your AI-powered HR system.</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _generate_contextual_response(self, prompt: str) -> str:
        """Generate response with chat history context"""
        try:
            # Check if we have meaningful conversation history
            recent_history = st.session_state.chat_history[-4:]  # Last 4 messages for context
            
            # If no significant history, use regular response
            if len(recent_history) < 2:
                return self.offer_generator.answer_policy_question(prompt)
            
            # Build minimal context for continuation
            context_topics = []
            for msg in recent_history:
                if msg['role'] == 'user':
                    # Extract key topics from user messages
                    topics = self._extract_topics(msg['content'])
                    context_topics.extend(topics)
            
            # Only add context if there are related topics
            if context_topics and any(topic in prompt.lower() for topic in context_topics):
                # Create subtle context hint
                context_hint = f"Context: Previous discussion included {', '.join(set(context_topics)[:2])}."
                enhanced_prompt = f"{context_hint}\n\nQuestion: {prompt}\n\nProvide a direct, helpful answer without repeating introductory phrases."
                
                response = self.offer_generator.answer_policy_question(enhanced_prompt)
            else:
                # Use regular response for unrelated questions
                response = self.offer_generator.answer_policy_question(prompt)
            
            return response
            
        except Exception as e:
            # Fallback to regular response if context fails
            return self.offer_generator.answer_policy_question(prompt)
    
    def _update_chat_memory(self, user_message: str, assistant_response: str):
        """Update persistent chat memory"""
        try:
            memory = st.session_state.chat_memory
            
            # Add to conversation context (keep last 10 exchanges)
            memory['conversation_context'].append({
                'user': user_message,
                'assistant': assistant_response,
                'timestamp': datetime.now().isoformat()
            })
            
            if len(memory['conversation_context']) > 10:
                memory['conversation_context'] = memory['conversation_context'][-10:]
            
            # Extract and store topics
            topics = self._extract_topics(user_message + " " + assistant_response)
            memory['topics_discussed'].update(topics)
            
            # Update session summary
            memory['session_summary'] = self._generate_session_summary()
            
        except Exception as e:
            pass  # Fail silently if memory update fails
    
    def _extract_topics(self, text: str) -> set:
        """Extract key topics from conversation"""
        hr_keywords = {
            'leave', 'vacation', 'sick', 'travel', 'policy', 'benefits', 'salary', 
            'insurance', 'medical', 'dental', 'retirement', '401k', 'pto', 'holiday',
            'maternity', 'paternity', 'remote', 'work', 'overtime', 'bonus', 'promotion',
            'performance', 'review', 'training', 'development', 'harassment', 'diversity'
        }
        
        words = set(word.lower() for word in text.split())
        return words.intersection(hr_keywords)
    
    def _generate_session_summary(self) -> str:
        """Generate summary of current session"""
        memory = st.session_state.chat_memory
        topics = list(memory['topics_discussed'])[:5]  # Top 5 topics
        
        if topics:
            return f"Discussed: {', '.join(topics)}"
        return "New conversation started"
    
    def _get_memory_summary(self) -> str:
        """Get formatted memory summary for sidebar"""
        memory = st.session_state.chat_memory
        
        summary_parts = []
        
        # Conversation length
        conv_count = len(memory['conversation_context'])
        summary_parts.append(f"• {conv_count} exchanges remembered")
        
        # Topics discussed
        topics = list(memory['topics_discussed'])[:3]
        if topics:
            summary_parts.append(f"• Topics: {', '.join(topics)}")
        
        # Session summary
        if memory['session_summary']:
            summary_parts.append(f"• {memory['session_summary']}")
        
        return "\n".join(summary_parts) if summary_parts else "• No memory yet"

if __name__ == "__main__":
    try:
        app = OfferLetterApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
        st.error("Stack trace:")
        st.code(traceback.format_exc())