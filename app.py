import streamlit as st
import pandas as pd
import os
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
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
        self.vector_store = VectorStore()
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
        """Main application runner"""
        st.title("🤖 AI-Powered Offer Letter Generator")
        st.markdown("Generate personalized offer letters using HR policies and employee data")
        
        # Add custom CSS for better styling
        self._add_custom_css()
        
        # Sidebar for configuration
        self.render_sidebar()
        
        # Main content area
        if st.session_state.documents_processed and st.session_state.employee_data_loaded:
            self.render_main_interface()
        else:
            self.render_setup_interface()

    def _add_custom_css(self):
        """Add custom CSS styling"""
        st.markdown("""
        <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #4CAF50;
        }
        .status-success {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-error {
            color: #f44336;
            font-weight: bold;
        }
        .employee-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
            margin-bottom: 1rem;
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
            process_btn = st.button("🔄 Process Documents", type="primary")
        with col2:
            clear_btn = st.button("🗑️ Clear Data", help="Clear all processed data")
        
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
        
        # Clear vector store
        if hasattr(self, 'vector_store'):
            self.vector_store = VectorStore()
        
        self.employee_df = pd.DataFrame()
        self.offer_generator = None
        st.sidebar.success("✅ All data cleared!")
        st.rerun()

    def process_documents(self, uploaded_files: List, employee_file) -> None:
        """Process uploaded documents and employee data"""
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            with st.spinner("Processing documents..."):
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
                    self.employee_df = pd.read_csv(employee_file)
                    self._validate_employee_data()
                except Exception as e:
                    st.error(f"Error reading employee CSV: {str(e)}")
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
                
                st.success(f"✅ Processed {len(chunks)} document chunks and {len(self.employee_df)} employee records")
                
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

    def render_setup_interface(self):
        """Render setup interface for first-time users"""
        st.header("🚀 Getting Started")
        
        # Introduction
        st.markdown("""
        Welcome to the AI-Powered Offer Letter Generator! This application helps HR teams create 
        personalized offer letters by leveraging company policies and employee data.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Step 1: Upload HR Documents")
            st.markdown("""
            Upload your HR policy documents including:
            - 📋 Leave Policy
            - ✈️ Travel Policy  
            - 📄 Sample Offer Letters
            - 🏢 Company Policies
            - 💼 Benefits Documentation
            
            **Supported formats:** PDF, TXT, DOCX
            """)
        
        with col2:
            st.subheader("👥 Step 2: Upload Employee Data")
            st.markdown("""
            Upload a CSV file with employee information:
            - 👤 Employee Name *(required)*
            - 🏢 Department
            - 📊 Band Level
            - 💰 Salary Information
            - 📍 Location
            - 📧 Email, etc.
            """)
        
        # Sample CSV template
        st.subheader("📝 Sample Employee CSV Format")
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
            mime="text/csv"
        )
        
        st.info("👆 Please use the sidebar to upload your documents and get started!")

    def render_main_interface(self):
        """Render main application interface"""
        if self.offer_generator is None:
            self.offer_generator = OfferGenerator(self.vector_store)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Chat Interface", 
            "📄 Generate Offer Letter", 
            "📊 Employee Data",
            "⚙️ System Info"
        ])
        
        with tab1:
            self.render_chat_interface()
        with tab2:
            self.render_offer_letter_generator()
        with tab3:
            self.render_employee_data_viewer()
        with tab4:
            self.render_system_info()

    def render_chat_interface(self):
        """Render chat interface for HR policy questions"""
        st.subheader("💬 Ask HR Policy Questions")
        st.markdown("Ask questions about HR policies, benefits, or employee information using natural language.")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about HR policies, benefits, or employee information..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = self.offer_generator.answer_policy_question(prompt)
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    def render_offer_letter_generator(self):
        """Render offer letter generation interface"""
        st.subheader("📄 Generate Offer Letter")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Select Employee:**")
            
            # Check if employee data is loaded
            if self.employee_df.empty:
                st.error("No employee data loaded. Please upload employee CSV file first.")
                return
            
            # Employee selection
            employee_names = sorted(self.employee_df['Employee Name'].tolist())
            selected_employee = st.selectbox(
                "Choose Employee",
                options=employee_names,
                help="Select an employee to generate offer letter",
                index=0 if not st.session_state.last_selected_employee else (
                    employee_names.index(st.session_state.last_selected_employee) 
                    if st.session_state.last_selected_employee in employee_names else 0
                )
            )
            
            if selected_employee:
                st.session_state.last_selected_employee = selected_employee
                employee_data = Utils.find_employee_by_name(self.employee_df, selected_employee)
                
                if not employee_data:
                    st.error(f"Employee data not found for: {selected_employee}")
                    return
                
                # Display employee info in a card-like format
                self._render_employee_card(employee_data)
                
                # Additional options
                st.markdown("**Generation Options:**")
                include_benefits = st.checkbox("Include detailed benefits", value=True)
                include_policies = st.checkbox("Include relevant policies", value=True)
                
                # Generate button
                if st.button("🚀 Generate Offer Letter", type="primary", use_container_width=True):
                    self.generate_and_display_offer_letter(
                        selected_employee, 
                        employee_data,
                        include_benefits=include_benefits,
                        include_policies=include_policies
                    )
        
        with col2:
            if 'generated_offer' not in st.session_state or st.session_state.generated_offer is None:
                st.info("👈 Select an employee and click 'Generate Offer Letter' to see the result here")
            else:
                st.markdown("### Generated Offer Letter")
                st.markdown("---")
                st.markdown(st.session_state.generated_offer)
                
                # Action buttons
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.download_button(
                        label="📥 Download as Markdown",
                        data=st.session_state.generated_offer,
                        file_name=f"offer_letter_{selected_employee.replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2b:
                    # Convert to plain text for download
                    plain_text = st.session_state.generated_offer.replace('#', '').replace('*', '')
                    st.download_button(
                        label="📄 Download as Text",
                        data=plain_text,
                        file_name=f"offer_letter_{selected_employee.replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2c:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        self.generate_and_display_offer_letter(selected_employee, employee_data)

    def _render_employee_card(self, employee_data: Dict[str, Any]):
        """Render employee information card"""
        st.markdown('<div class="employee-card">', unsafe_allow_html=True)
        st.markdown("**Employee Details:**")
        
        # Basic info
        st.write(f"**👤 Name:** {employee_data.get('Employee Name', 'N/A')}")
        st.write(f"**🏢 Department:** {employee_data.get('Department', 'N/A')}")
        st.write(f"**📊 Band:** {employee_data.get('Band', 'N/A')}")
        st.write(f"**📍 Location:** {employee_data.get('Location', 'N/A')}")
        
        # Salary info
        if 'Total CTC (INR)' in employee_data:
            ctc = employee_data.get('Total CTC (INR)', 0)
            formatted_ctc = Utils.format_currency(ctc) if hasattr(Utils, 'format_currency') else f"₹{ctc:,.2f}"
            st.write(f"**💰 Total CTC:** {formatted_ctc}")
        
        # Additional fields
        for key, value in employee_data.items():
            if key not in ['Employee Name', 'Department', 'Band', 'Location', 'Total CTC (INR)'] and value:
                st.write(f"**{key}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def generate_and_display_offer_letter(self, employee_name: str, employee_data: Dict[str, Any], 
                                        include_benefits: bool = True, include_policies: bool = True):
        """Generate and display offer letter"""
        with st.spinner(f"Generating offer letter for {employee_name}..."):
            try:
                # Add generation options to employee data
                generation_options = {
                    'include_benefits': include_benefits,
                    'include_policies': include_policies
                }
                
                offer_letter = self.offer_generator.generate_offer_letter(
                    employee_name, 
                    employee_data,
                    **generation_options
                )
                
                # Store in session state
                st.session_state.generated_offer = offer_letter
                st.success(f"✅ Offer letter generated successfully for {employee_name}!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating offer letter: {str(e)}")
                st.error("Stack trace:")
                st.code(traceback.format_exc())

    def render_employee_data_viewer(self):
        """Render employee data overview and management"""
        st.subheader("📊 Employee Data Overview")
        
        if not self.employee_df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Employees", len(self.employee_df))
            with col2:
                if 'Department' in self.employee_df.columns:
                    st.metric("Departments", self.employee_df['Department'].nunique())
                else:
                    st.metric("Departments", "N/A")
            with col3:
                if 'Band' in self.employee_df.columns:
                    st.metric("Band Levels", self.employee_df['Band'].nunique())
                else:
                    st.metric("Band Levels", "N/A")
            with col4:
                if 'Total CTC (INR)' in self.employee_df.columns:
                    avg_ctc = self.employee_df['Total CTC (INR)'].mean()
                    formatted_avg = Utils.format_currency(avg_ctc) if hasattr(Utils, 'format_currency') else f"₹{avg_ctc:,.0f}"
                    st.metric("Avg CTC", formatted_avg)
                else:
                    st.metric("Avg CTC", "N/A")
            
            # Filters
            st.markdown("**Filter Data:**")
            col1, col2 = st.columns(2)
            
            filtered_df = self.employee_df.copy()
            
            with col1:
                if 'Department' in self.employee_df.columns:
                    dept_options = sorted(self.employee_df['Department'].unique())
                    dept_filter = st.multiselect(
                        "Department",
                        options=dept_options,
                        default=dept_options
                    )
                    filtered_df = filtered_df[filtered_df['Department'].isin(dept_filter)]
            
            with col2:
                if 'Band' in self.employee_df.columns:
                    band_options = sorted(self.employee_df['Band'].unique())
                    band_filter = st.multiselect(
                        "Band Level",
                        options=band_options,
                        default=band_options
                    )
                    filtered_df = filtered_df[filtered_df['Band'].isin(band_filter)]
            
            # Search functionality
            search_term = st.text_input("🔍 Search employees", placeholder="Enter name, department, or any field...")
            if search_term:
                mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                filtered_df = filtered_df[mask]
            
            # Display filtered data
            st.markdown(f"**Showing {len(filtered_df)} employees:**")
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            if st.button("📥 Export Filtered Data"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="filtered_employees.csv",
                    mime="text/csv"
                )
            
        else:
            st.warning("No employee data loaded. Please upload employee CSV file first.")

    def render_system_info(self):
        """Render system information and diagnostics"""
        st.subheader("⚙️ System Information")
        
        # Vector store statistics
        if hasattr(self.vector_store, 'get_statistics'):
            st.markdown("**Vector Store Statistics:**")
            try:
                stats = self.vector_store.get_statistics()
                col1, col2 = st.columns(2)
                
                with col1:
                    for key, value in list(stats.items())[:len(stats)//2]:
                        st.metric(key.replace('_', ' ').title(), value)
                
                with col2:
                    for key, value in list(stats.items())[len(stats)//2:]:
                        st.metric(key.replace('_', ' ').title(), value)
                        
            except Exception as e:
                st.error(f"Error retrieving vector store statistics: {str(e)}")
        
        # Health check
        if st.button("🏥 Run Health Check"):
            with st.spinner("Running health check..."):
                try:
                    if hasattr(self.vector_store, 'health_check'):
                        health_status = self.vector_store.health_check()
                        
                        # Display health status
                        if health_status['status'] == 'healthy':
                            st.success(f"✅ System Status: {health_status['status'].upper()}")
                        elif health_status['status'] == 'degraded':
                            st.warning(f"⚠️ System Status: {health_status['status'].upper()}")
                        else:
                            st.error(f"❌ System Status: {health_status['status'].upper()}")
                        
                        # Display checks
                        if health_status.get('checks'):
                            st.markdown("**Health Checks:**")
                            for check_name, check_result in health_status['checks'].items():
                                status_icon = "✅" if check_result == 'ok' else "❌"
                                st.write(f"{status_icon} {check_name}: {check_result}")
                        
                        # Display warnings and errors
                        if health_status.get('warnings'):
                            st.markdown("**Warnings:**")
                            for warning in health_status['warnings']:
                                st.warning(warning)
                        
                        if health_status.get('errors'):
                            st.markdown("**Errors:**")
                            for error in health_status['errors']:
                                st.error(error)
                    else:
                        st.info("Health check not available for current vector store")
                        
                except Exception as e:
                    st.error(f"Health check failed: {str(e)}")

if __name__ == "__main__":
    try:
        app = OfferLetterApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {str(e)}")
        st.error("Stack trace:")
        st.code(traceback.format_exc())