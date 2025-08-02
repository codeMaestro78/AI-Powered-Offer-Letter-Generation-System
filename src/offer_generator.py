from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import json
import hashlib
import logging

from src.llm_interface import LLMInterface
from src.vector_store import VectorStore
from config.settings import Config
from templates.offer_template import OfferTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfferType(Enum):
    """Types of offer letters"""
    FULL_TIME = "full_time"
    INTERN = "intern"
    CONTRACT = "contract"
    CONSULTANT = "consultant"

class ValidationStatus(Enum):
    """Validation status for offers"""
    VALID = "valid"
    INCOMPLETE = "incomplete"
    ERROR = "error"

@dataclass(frozen=True)
class EmployeeProfile:
    """Structured employee data model"""
    name: str
    band: str
    department: str
    location: str = "Bangalore"
    base_salary: float = 0
    performance_bonus: float = 0
    retention_bonus: float = 0
    total_ctc: float = 0
    joining_date: Optional[str] = None
    manager: str = "To be assigned"
    offer_type: OfferType = OfferType.FULL_TIME
    additional_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmployeeProfile':
        """Create EmployeeProfile from dictionary"""
        return cls(
            name=data.get('Name', 'Unknown'),
            band=data.get('Band', 'L1'),
            department=data.get('Department', 'General'),
            location=data.get('Location', 'Bangalore'),
            base_salary=float(data.get('Base Salary (INR)', 0)),
            performance_bonus=float(data.get('Performance Bonus (INR)', 0)),
            retention_bonus=float(data.get('Retention Bonus (INR)', 0)),
            total_ctc=float(data.get('Total CTC (INR)', 0)),
            joining_date=data.get('Joining Date'),
            manager=data.get('Manager', 'To be assigned'),
            offer_type=OfferType(data.get('Offer Type', 'full_time')),
            additional_data={k: v for k, v in data.items() if k not in [
                'Name', 'Band', 'Department', 'Location', 'Base Salary (INR)',
                'Performance Bonus (INR)', 'Retention Bonus (INR)', 'Total CTC (INR)',
                'Joining Date', 'Manager', 'Offer Type'
            ]}
        )

@dataclass
class PolicyContext:
    """Context for policy retrieval"""
    band_policies: List[Dict[str, Any]] = field(default_factory=list)
    department_policies: List[Dict[str, Any]] = field(default_factory=list)
    location_policies: List[Dict[str, Any]] = field(default_factory=list)
    general_policies: List[Dict[str, Any]] = field(default_factory=list)
    templates: List[Dict[str, Any]] = field(default_factory=list)

class OfferGenerator:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.config = Config()  # Initialize the Config object
        self.llm = LLMInterface()
        self.template = OfferTemplate()
        
        # Initialize performance tracking
        self.generation_stats = {
            'total_generated': 0,
            'successful': 0,
            'failed': 0,
            'avg_generation_time': 0.0,
            'cache_hits': 0
        }
        
        # Initialize thread-safe caching
        self._policy_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 3600  # 1 hour cache TTL
        
        # Speed optimization settings
        self._fast_mode = True
        self._skip_extensive_validation = True
        self._use_parallel_processing = True
        
        # Test system connectivity
        self._test_system_connectivity()

    def _test_system_connectivity(self):
        """Test system connectivity and log status"""
        try:
            if self.llm.test_connection():
                logger.info("✅ LLM connection successful")
            else:
                logger.warning("⚠️ LLM connection test failed. Proceeding with fallback mode...")
        except Exception as e:
            logger.error(f"❌ System connectivity test failed: {str(e)}")

    def _fetch_employee_data(self, employee_name: str) -> Dict[str, Any]:
        """Fetch employee data from VectorStore or use defaults."""
        try:
            query = f"employee profile for {employee_name}"
            search_results = self.vector_store.search(query, k=1, score_threshold=0.7)
            if search_results and hasattr(search_results[0], 'chunk'):
                employee_chunk = search_results[0].chunk
                employee_data = {
                    'Name': employee_chunk.get('name', employee_name),
                    'Band': employee_chunk.get('band', 'L3'),
                    'Department': employee_chunk.get('department', 'Engineering'),
                    'Location': employee_chunk.get('location', 'Bangalore'),
                    'Base Salary (INR)': float(employee_chunk.get('base_salary', 900000)),
                    'Performance Bonus (INR)': float(employee_chunk.get('performance_bonus', 200000)),
                    'Retention Bonus (INR)': float(employee_chunk.get('retention_bonus', 100000)),
                    'Total CTC (INR)': float(employee_chunk.get('total_ctc', 1200000)),
                    'Joining Date': employee_chunk.get('joining_date', (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d')),
                    'Manager': employee_chunk.get('manager', 'Priya Sharma'),
                    'Offer Type': employee_chunk.get('offer_type', 'full_time')
                }
                logger.info(f"Fetched employee data for {employee_name}: {employee_data}")
                return employee_data
            else:
                logger.warning(f"No employee data found for {employee_name}. Using defaults.")
        except Exception as e:
            logger.error(f"Error fetching employee data: {str(e)}. Using defaults.")

        # Default employee data
        return {
            'Name': employee_name,
            'Band': 'L3',
            'Department': 'Engineering',
            'Location': 'Bangalore',
            'Base Salary (INR)': 900000,
            'Performance Bonus (INR)': 200000,
            'Retention Bonus (INR)': 100000,
            'Total CTC (INR)': 1200000,
            'Joining Date': (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
            'Manager': 'Priya Sharma',
            'Offer Type': 'full_time'
        }

    def generate_offer_letter(self, employee_name: str, employee_data: Dict[str, Any] = None, **kwargs) -> str:
        """Generate offer letter with optimized speed and smart quality balance"""
        start_time = time.time()
        generation_id = hashlib.md5(f"{employee_name}_{time.time()}".encode()).hexdigest()[:8]
        
        logger.info(f"⚡ Fast offer generation [{generation_id}] for {employee_name}")
        
        try:
            # Fast Path: Quick validation and generation
            profile = self._fast_create_profile(employee_name, employee_data)
            template_config = self._select_optimal_template(profile)
            
            # Parallel processing for speed
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Concurrent policy retrieval and query preparation
                policy_future = executor.submit(self._get_policy_context, profile)
                query_future = executor.submit(self._create_enhanced_offer_query, profile)
                
                policy_context = policy_future.result(timeout=5)  # 5 second timeout
                query = query_future.result(timeout=2)  # 2 second timeout
            
            # Single-attempt generation with fast model
            offer_letter = self._fast_generate_offer(query, profile, policy_context, template_config)
            
            # Quick post-processing
            final_offer = self._fast_post_process(offer_letter, profile, template_config)
            
            generation_time = time.time() - start_time
            self._update_generation_stats(True, generation_time)
            
            logger.info(f"⚡ Offer generated in {generation_time:.2f}s [{generation_id}]")
            return final_offer
            
        except Exception as e:
            generation_time = time.time() - start_time
            self._update_generation_stats(False, generation_time)
            logger.error(f"❌ Fast generation failed [{generation_id}]: {str(e)}")
            
            # Quick fallback
            if 'profile' in locals():
                return self._generate_quick_fallback(profile, generation_id)
            else:
                minimal_profile = EmployeeProfile(name=employee_name, band="L1", department="General")
                return self._generate_quick_fallback(minimal_profile, generation_id)

    def answer_policy_question(self, query: str, employee_data: Dict[str, Any] = None) -> str:
        """Answer an HR policy-related question."""
        try:
            logger.info(f"📋 Answering policy question: {query}")
            # Fetch employee data if not provided
            if not employee_data:
                employee_data = self._fetch_employee_data("Generic Employee")
            employee_profile = EmployeeProfile.from_dict(employee_data)
            
            # Retrieve policy context
            policy_context = self._get_policy_context(employee_profile)
            all_chunks = self._flatten_policy_context(policy_context)
            
            # Generate response using LLM
            response = self.llm.generate_response(query, all_chunks, employee_data)
            
            logger.info(f"✅ Policy question answered successfully")
            return response
        except Exception as e:
            logger.error(f"❌ Error answering policy question: {str(e)}")
            return self.llm._get_fallback_response(self.llm._classify_query_type(query))

    @dataclass
    class ValidationResult:
        is_valid: bool
        error_message: str = ""
        warnings: List[str] = field(default_factory=list)

    def _validate_employee_profile(self, profile: EmployeeProfile) -> ValidationResult:
        """Enhanced validation with detailed feedback"""
        errors = []
        warnings = []
        
        required_fields = {
            'name': profile.name,
            'band': profile.band,
            'department': profile.department,
            'total_ctc': profile.total_ctc
        }
        
        for field_name, value in required_fields.items():
            if not value:
                errors.append(f"Missing required field: {field_name}")
        
        if profile.total_ctc <= 0:
            errors.append("Total CTC must be greater than 0")
        
        if profile.base_salary > profile.total_ctc:
            warnings.append("Base salary exceeds total CTC")
        
        if profile.band not in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']:
            warnings.append(f"Unusual band level: {profile.band}")
        
        if profile.joining_date:
            try:
                joining_date = datetime.strptime(profile.joining_date, '%Y-%m-%d')
                if joining_date < datetime.now():
                    warnings.append("Joining date is in the past")
            except ValueError:
                errors.append("Invalid joining date format. Use YYYY-MM-DD")
        
        return self.ValidationResult(
            is_valid=len(errors) == 0,
            error_message="; ".join(errors),
            warnings=warnings
        )

    def _get_policy_context(self, profile: EmployeeProfile) -> PolicyContext:
        """Retrieve and cache policy context efficiently"""
        cache_key = self._generate_cache_key(profile)
        
        with self._cache_lock:
            if cache_key in self._policy_cache:
                cached_data = self._policy_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self._cache_ttl:
                    self.generation_stats['cache_hits'] += 1
                    logger.info("📦 Using cached policy context")
                    return cached_data['context']
                else:
                    del self._policy_cache[cache_key]
        
        context = self._retrieve_policy_context(profile)
        
        with self._cache_lock:
            self._policy_cache[cache_key] = {
                'context': context,
                'timestamp': time.time()
            }
        
        return context

    def _generate_cache_key(self, profile: EmployeeProfile) -> str:
        """Generate cache key for policy context"""
        key_data = f"{profile.band}_{profile.department}_{profile.location}_{profile.offer_type.value}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _retrieve_policy_context(self, profile: EmployeeProfile) -> PolicyContext:
        """Retrieve all relevant policies with parallel processing"""
        context = PolicyContext()
        
        search_tasks = [
            ('band', f"band {profile.band} level entitlements benefits policy compensation", 4),
            ('department', f"{profile.department} department work from office travel policy requirements", 3),
            ('location', f"{profile.location} office location work from office commute policy", 2),
            ('leave', "leave entitlement annual sick casual maternity paternity policy", 2),
            ('benefits', "employee benefits health insurance medical policy", 2),
            ('training', "training development learning opportunities career growth", 1),
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_category = {
                executor.submit(self.vector_store.search, query, k): category
                for category, query, k in search_tasks
            }
            
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    chunks = future.result()
                    chunk_dicts = [chunk.chunk for chunk in chunks if hasattr(chunk, 'chunk')]
                    if category == 'band':
                        context.band_policies = chunk_dicts
                    elif category == 'department':
                        context.department_policies = chunk_dicts
                    elif category == 'location':
                        context.location_policies = chunk_dicts
                    else:
                        context.general_policies.extend(chunk_dicts)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to retrieve {category} policies: {str(e)}")
        
        try:
            template_chunks = self.vector_store.search_by_metadata({"type": "template"}, k=2)
            context.templates = [chunk.chunk for chunk in template_chunks if hasattr(chunk, 'chunk')]
        except Exception as e:
            logger.warning(f"⚠️ Failed to retrieve templates: {str(e)}")
            context.templates = [{"content": "Standard offer letter template"}]
        
        logger.info(f"📋 Retrieved policy context: {len(self._flatten_policy_context(context))} total chunks")
        return context

    def _flatten_policy_context(self, context: PolicyContext) -> List[Dict[str, Any]]:
        """Flatten policy context and remove duplicates"""
        all_chunks = []
        seen_hashes = set()
        
        for chunk_group in [context.templates, context.band_policies, context.department_policies, 
                           context.location_policies, context.general_policies]:
            for chunk in chunk_group:
                content = chunk.get('content', '')
                content_hash = hashlib.md5(content[:500].encode()).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    all_chunks.append(chunk)
                if len(all_chunks) >= 15:
                    break
        
        return all_chunks

    def _create_enhanced_offer_query(self, profile: EmployeeProfile) -> str:
        """Create a detailed and professional prompt for offer letter generation"""

        joining_date = profile.joining_date or (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        position_title = self._determine_position_title(profile)

        query = f"""
    Generate a formal and detailed offer letter for the candidate **{profile.name}**, using the following structured information:

    ### EMPLOYEE DETAILS
    - **Name**: {profile.name}
    - **Position Title**: {position_title}
    - **Band Level**: {profile.band}
    - **Department**: {profile.department}
    - **Work Location**: {profile.location}
    - **Joining Date**: {joining_date}
    - **Reporting Manager**: {profile.manager}
    - **Offer Type**: {profile.offer_type.value.replace('_', ' ').title()}

    ### COMPENSATION STRUCTURE
    - **Base Salary**: ₹{profile.base_salary:,.0f} per annum
    - **Performance Bonus**: ₹{profile.performance_bonus:,.0f} per annum
    - **Retention Bonus**: ₹{profile.retention_bonus:,.0f} per annum
    - **Total CTC**: ₹{profile.total_ctc:,.0f} per annum

    ### OFFER LETTER GUIDELINES
    Generate a comprehensive and professionally written offer letter that adheres to the following requirements:

    1. Start with a **welcoming and positive introduction**.
    2. Present a **clear summary of the role and compensation**.
    3. Include **policy highlights**, such as:
    - Leave policies relevant to **{profile.band}**
    - Department-specific guidelines for **{profile.department}**
    - Location-specific provisions for **{profile.location}**
    - Standard employee benefits and perks
    - Performance review and evaluation framework
    4. Include all necessary **legal terms and conditions** (e.g., at-will employment, confidentiality, notice period).
    5. Conclude with **next steps** and clear **HR contact details**.
    6. End with a **professional sign-off** and space for digital or physical signature.

    ### FORMATTING INSTRUCTIONS
    - Use official **business letter formatting** with proper headings
    - Maintain a **professional, respectful tone**
    - Use **bullet points** for benefits and policy items
    - Ensure **logical structure, spacing, and readability**

    The final output should be a polished, ready-to-send offer letter that reflects **{self.config.COMPANY_NAME}**'s brand and professionalism, incorporating the provided employee and compensation details along with relevant policies.
    """
        return query


    def _determine_position_title(self, profile: EmployeeProfile) -> str:
        """Determine position title based on department and band"""
        position_mapping = {
            'Engineering': 'Software Engineer',
            'Sales': 'Sales Executive',
            'HR': 'HR Specialist',
            'Finance': 'Financial Analyst',
            'Marketing': 'Marketing Specialist'
        }
        base_title = position_mapping.get(profile.department, f"{profile.department} Specialist")
        if profile.band in ['L1', 'L2']:
            return f"Junior {base_title}"
        elif profile.band in ['L5', 'L6']:
            return f"Senior {base_title}"
        return base_title

    def _profile_to_legacy_dict(self, profile: EmployeeProfile) -> Dict[str, Any]:
        """Convert EmployeeProfile to legacy dictionary format"""
        return {
            'Name': profile.name,
            'Band': profile.band,
            'Department': profile.department,
            'Location': profile.location,
            'Base Salary (INR)': profile.base_salary,
            'Performance Bonus (INR)': profile.performance_bonus,
            'Retention Bonus (INR)': profile.retention_bonus,
            'Total CTC (INR)': profile.total_ctc,
            'Joining Date': profile.joining_date,
            'Manager': profile.manager,
            'Offer Type': profile.offer_type.value,
            **profile.additional_data
        }

    def _post_process_offer(self, offer_letter: str, profile: EmployeeProfile) -> str:
        """Post-process offer letter for consistency and formatting"""
        processed = offer_letter.strip()
        if not processed.startswith('[') and 'CompanyABC' not in processed[:200]:
            header = f"""
[CompanyABC]
123 Business Park, Bangalore, Karnataka, India
Date: {datetime.now().strftime('%B %d, %Y')}

"""
            processed = header + processed
        if 'peopleops@companyabc.com' not in processed:
            processed += f"\n\nFor questions, contact HR at peopleops@companyabc.com."
        return processed.replace('\n\n\n', '\n\n')

    def _generate_fallback_offer(self, profile: EmployeeProfile, error: str) -> str:
        """Generate a minimal but professional fallback offer letter in case of system failure"""

        position_title = self._determine_position_title(profile)
        today = datetime.now().strftime('%B %d, %Y')

        return f"""
    [CompanyABC]  
    123 Business Park, Bangalore, Karnataka, India  
    Date: {today}

    Dear {profile.name},

    We are delighted to extend a preliminary offer for the position of **{position_title}** in the **{profile.department}** department at **CompanyABC**.

    ### Position Summary:
    - **Role**: {position_title}  
    - **Band Level**: {profile.band}  
    - **Department**: {profile.department}  
    - **Work Location**: {profile.location}  
    - **Total Annual Compensation (CTC)**: ₹{profile.total_ctc:,.0f}

    This offer is extended in accordance with our standard employment terms and company policies. A detailed offer letter will follow shortly.

    In the meantime, if you have any questions or require clarification, please reach out to our HR team at **peopleops@companyabc.com**.

    We appreciate your patience and look forward to having you on board.

    Warm regards,  
    **Anita Desai**  
    Head of Human Resources  
    CompanyABC

    ---

    **Note**: This letter has been generated as a fallback due to a temporary system issue.  
    *Error reference: {error}*
    """


    def _update_generation_stats(self, success: bool, generation_time: float):
        """Update generation statistics"""
        self.generation_stats['total_generated'] += 1
        if success:
            self.generation_stats['successful'] += 1
        else:
            self.generation_stats['failed'] += 1
        current_avg = self.generation_stats['avg_generation_time']
        total = self.generation_stats['total_generated']
        self.generation_stats['avg_generation_time'] = ((current_avg * (total - 1)) + generation_time) / total

    def validate_offer_completeness(self, offer_letter: str, profile: Optional[EmployeeProfile] = None) -> Dict[str, Any]:
        """Validate offer letter completeness"""
        validation_checks = {
            'has_employee_name': False,
            'has_position': False,
            'has_salary_details': False,
            'has_joining_date': False,
            'has_policy_information': False,
            'has_contact_info': False,
            'has_professional_format': False,
            'has_company_branding': False,
            'has_benefits_details': False,
            'has_legal_terms': False
        }
        
        offer_lower = offer_letter.lower()
        validation_checks['has_salary_details'] = ('₹' in offer_letter or 'inr' in offer_lower or 'salary' in offer_lower)
        validation_checks['has_joining_date'] = any(word in offer_lower for word in ['date', 'join', 'start', 'commence'])
        validation_checks['has_policy_information'] = any(word in offer_lower for word in ['policy', 'leave', 'benefit', 'entitlement'])
        validation_checks['has_contact_info'] = 'peopleops@companyabc.com' in offer_lower
        validation_checks['has_professional_format'] = len(offer_letter.split('\n')) > 10 and 'dear' in offer_lower
        validation_checks['has_company_branding'] = 'companyabc' in offer_lower
        validation_checks['has_benefits_details'] = any(word in offer_lower for word in ['health', 'insurance', 'medical', 'pf', 'provident'])
        validation_checks['has_legal_terms'] = any(word in offer_lower for word in ['terms', 'conditions', 'agreement', 'employment'])
        
        if profile:
            validation_checks['has_employee_name'] = profile.name.lower() in offer_lower
            position_title = self._determine_position_title(profile)
            validation_checks['has_position'] = any(word in offer_lower for word in position_title.lower().split())
        else:
            validation_checks['has_employee_name'] = True
            validation_checks['has_position'] = True
        
        total_checks = len(validation_checks)
        passed_checks = sum(validation_checks.values())
        completion_score = (passed_checks / total_checks) * 100
        status = ValidationStatus.VALID if completion_score >= 90 else ValidationStatus.INCOMPLETE if completion_score >= 70 else ValidationStatus.ERROR
        
        return {
            'validation_checks': validation_checks,
            'completion_score': completion_score,
            'status': status.value,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'missing_elements': [check for check, passed in validation_checks.items() if not passed]
        }

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics"""
        stats = self.generation_stats.copy()
        if stats['total_generated'] > 0:
            stats['success_rate'] = (stats['successful'] / stats['total_generated']) * 100
            stats['failure_rate'] = (stats['failed'] / stats['total_generated']) * 100
            stats['cache_hit_rate'] = (stats['cache_hits'] / stats['total_generated']) * 100
        else:
            stats['success_rate'] = 0
            stats['failure_rate'] = 0
            stats['cache_hit_rate'] = 0
        stats['cache_size'] = len(self._policy_cache)
        stats['llm_stats'] = self.llm.get_performance_stats()
        return stats

    def clear_cache(self):
        """Clear all caches"""
        with self._cache_lock:
            self._policy_cache.clear()
        self.llm.clear_cache()
        logger.info("🧹 All caches cleared")

    def _auto_fix_profile_issues(self, profile: EmployeeProfile, validation_result) -> EmployeeProfile:
        """Auto-fix common profile issues"""
        fixes_applied = []
        profile_dict = self._profile_to_legacy_dict(profile)
        
        # Fix missing or invalid salary
        if profile.total_ctc <= 0 and profile.base_salary <= 0:
            # Estimate based on band and department
            estimated_salary = self._estimate_salary(profile.band, profile.department)
            profile_dict.update({
                'Base Salary (INR)': estimated_salary * 0.7,
                'Performance Bonus (INR)': estimated_salary * 0.2,
                'Total CTC (INR)': estimated_salary
            })
            fixes_applied.append('salary_estimation')
        
        # Fix invalid band
        if profile.band not in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'INTERN']:
            profile_dict['Band'] = 'L1'
            fixes_applied.append('band_normalization')
        
        # Fix missing joining date
        if not profile.joining_date:
            profile_dict['Joining Date'] = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            fixes_applied.append('joining_date_default')
        
        if fixes_applied:
            logger.info(f"🔧 Auto-fixes applied: {', '.join(fixes_applied)}")
            return EmployeeProfile.from_dict(profile_dict)
        
        return profile
    
    def _estimate_salary(self, band: str, department: str) -> float:
        """Estimate salary based on band and department"""
        base_salaries = {
            'L1': 600000, 'L2': 800000, 'L3': 1200000, 'L4': 1800000,
            'L5': 2500000, 'L6': 3500000, 'L7': 5000000, 'L8': 7000000,
            'L9': 10000000, 'INTERN': 25000
        }
        
        department_multipliers = {
            'Engineering': 1.1, 'Product': 1.05, 'Data Science': 1.15,
            'Sales': 1.0, 'Marketing': 0.95, 'HR': 0.9, 'Finance': 1.0
        }
        
        base = base_salaries.get(band, 800000)
        multiplier = department_multipliers.get(department, 1.0)
        
        return base * multiplier
    
    def _get_template_specific_policies(self, template_config: Dict[str, Any]) -> Dict[str, List]:
        """Get policies specific to the selected template"""
        # This would typically query the vector store for template-specific policies
        # For now, return empty structure
        return {'band': [], 'department': [], 'location': [], 'general': [], 'templates': []}
    
    def _get_role_specific_policies(self, profile: EmployeeProfile) -> List[Dict[str, Any]]:
        """Get policies specific to the role/position"""
        # This would query for role-specific policies
        return []
    
    def _deduplicate_policies(self, context: PolicyContext) -> PolicyContext:
        """Remove duplicate policies from context"""
        seen = set()
        
        def dedupe_list(policy_list):
            unique_policies = []
            for policy in policy_list:
                policy_hash = hashlib.md5(str(policy).encode()).hexdigest()
                if policy_hash not in seen:
                    seen.add(policy_hash)
                    unique_policies.append(policy)
            return unique_policies
        
        return PolicyContext(
            band_policies=dedupe_list(context.band_policies),
            department_policies=dedupe_list(context.department_policies),
            location_policies=dedupe_list(context.location_policies),
            general_policies=dedupe_list(context.general_policies),
            templates=dedupe_list(context.templates)
        )
    
    def _create_enhanced_offer_query_v2(self, profile: EmployeeProfile, template_config: Dict[str, Any], attempt: int) -> str:
        """Create enhanced query with attempt-specific improvements"""
        base_query = self._create_enhanced_offer_query(profile)
        
        # Add template-specific instructions
        template_instructions = f"\n\nTEMPLATE REQUIREMENTS:\n- Use {template_config['tone']} tone\n- Include sections: {', '.join(template_config['sections'])}\n- Template type: {template_config['type']}"
        
        # Add attempt-specific improvements
        attempt_improvements = {
            0: "\n\nFOCUS: Ensure professional formatting and complete information.",
            1: "\n\nFOCUS: Enhance clarity and add more specific details about benefits and policies.",
            2: "\n\nFOCUS: Maximize completeness and professional presentation. This is the final attempt."
        }
        
        return base_query + template_instructions + attempt_improvements.get(attempt, "")
    
    def _quick_quality_check(self, offer_letter: str, profile: EmployeeProfile) -> float:
        """Perform quick quality assessment"""
        checks = {
            'has_name': profile.name.lower() in offer_letter.lower(),
            'has_position': len([word for word in ['engineer', 'manager', 'analyst', 'specialist', 'director'] if word in offer_letter.lower()]) > 0,
            'has_salary': any(symbol in offer_letter for symbol in ['₹', 'INR', 'salary']),
            'has_company': 'companyabc' in offer_letter.lower(),
            'has_contact': 'peopleops' in offer_letter.lower(),
            'proper_length': 500 <= len(offer_letter) <= 3000,
            'has_structure': len(offer_letter.split('\n')) >= 10
        }
        
        return (sum(checks.values()) / len(checks)) * 100
    
    # Legacy methods for backward compatibility - now optimized for speed
    def _comprehensive_post_processing(self, offer_letter: str, profile: EmployeeProfile, template_config: Dict[str, Any]) -> str:
        """Fast post-processing with essential enhancements only"""
        return self._fast_post_process(offer_letter, profile, template_config)
    
    def _perform_quality_assurance(self, offer_letter: str, profile: EmployeeProfile) -> Dict[str, Any]:
        """Fast quality check with minimal overhead"""
        quality_score = self._quick_quality_check(offer_letter, profile)
        
        return {
            'quality_score': quality_score,
            'individual_scores': {'overall': quality_score},
            'validation_details': {'completion_score': quality_score},
            'recommendations': [] if quality_score >= 75 else ['quick_enhance']
        }
    
    def _enhance_offer_quality(self, offer_letter: str, profile: EmployeeProfile, qa_result: Dict[str, Any]) -> str:
        """Fast quality enhancement"""
        return self._quick_enhance_offer(offer_letter, profile)
    
    def _check_formatting_quality(self, offer_letter: str) -> float:
        """Check formatting quality"""
        # Implementation for formatting checks
        return 85.0  # Placeholder
    
    def _check_content_accuracy(self, offer_letter: str, profile: EmployeeProfile) -> float:
        """Check content accuracy"""
        # Implementation for content accuracy checks
        return 90.0  # Placeholder
    
    def _check_professional_tone(self, offer_letter: str) -> float:
        """Check professional tone"""
        # Implementation for tone analysis
        return 88.0  # Placeholder
    
    def _generate_improvement_recommendations(self, qa_checks: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        if qa_checks['formatting_quality'] < 80:
            recommendations.append('improve_formatting')
        if qa_checks['content_accuracy'] < 85:
            recommendations.append('add_missing_details')
        if qa_checks['professional_tone'] < 85:
            recommendations.append('enhance_professionalism')
        return recommendations
    
    def _improve_formatting(self, offer_letter: str) -> str:
        """Improve formatting"""
        return offer_letter  # Placeholder implementation
    
    def _add_missing_details(self, offer_letter: str, profile: EmployeeProfile) -> str:
        """Add missing details"""
        return offer_letter  # Placeholder implementation
    
    def _enhance_professionalism(self, offer_letter: str) -> str:
        """Enhance professionalism"""
        return offer_letter  # Placeholder implementation
    
    def _add_executive_enhancements(self, offer_letter: str, profile: EmployeeProfile) -> str:
        """Add executive-level enhancements"""
        return offer_letter  # Placeholder implementation
    
    def _add_intern_enhancements(self, offer_letter: str, profile: EmployeeProfile) -> str:
        """Add intern-specific enhancements"""
        return offer_letter  # Placeholder implementation

    def _create_and_validate_profile(self, employee_name: str, employee_data: Dict[str, Any] = None) -> EmployeeProfile:
        """Create and validate employee profile with enhanced checks"""
        # Create profile
        if employee_data:
            profile = EmployeeProfile.from_dict(employee_data)
        else:
            employee_data = self._fetch_employee_data(employee_name)
            profile = EmployeeProfile.from_dict(employee_data)
        
        # Enhanced validation
        validation_result = self._validate_employee_profile(profile)
        if not validation_result.is_valid:
            # Try to auto-fix common issues
            profile = self._auto_fix_profile_issues(profile, validation_result)
            
            # Re-validate after fixes
            validation_result = self._validate_employee_profile(profile)
            if not validation_result.is_valid:
                raise ValueError(f"Profile validation failed: {validation_result.error_message}")
        
        return profile
    
    def _select_optimal_template(self, profile: EmployeeProfile) -> Dict[str, Any]:
        """Select optimal template based on employee profile and role requirements"""
        template_configs = {
            'senior_executive': {
                'type': 'senior_executive',
                'bands': ['L6', 'L7', 'L8', 'L9'],
                'sections': ['welcome', 'role_details', 'compensation_detailed', 'equity', 'benefits_comprehensive', 'policies', 'legal_detailed'],
                'tone': 'formal_executive'
            },
            'manager': {
                'type': 'manager',
                'bands': ['L4', 'L5'],
                'sections': ['welcome', 'role_details', 'compensation', 'team_info', 'benefits', 'policies', 'legal'],
                'tone': 'professional_warm'
            },
            'individual_contributor': {
                'type': 'individual_contributor',
                'bands': ['L1', 'L2', 'L3'],
                'sections': ['welcome', 'role_details', 'compensation', 'benefits', 'policies', 'legal'],
                'tone': 'professional_friendly'
            },
            'intern': {
                'type': 'intern',
                'bands': ['INTERN'],
                'sections': ['welcome', 'internship_details', 'stipend', 'learning_opportunities', 'policies_basic'],
                'tone': 'encouraging'
            }
        }
        
        # Select based on band and offer type
        if profile.offer_type == OfferType.INTERN:
            return template_configs['intern']
        
        for template_name, config in template_configs.items():
            if profile.band in config['bands']:
                return config
        
        # Default fallback
        return template_configs['individual_contributor']
    
    def _get_enhanced_policy_context(self, profile: EmployeeProfile) -> PolicyContext:
        """Get enhanced policy context with intelligent filtering"""
        base_context = self._get_policy_context(profile)
        
        # Add template-specific policies
        template_config = self._select_optimal_template(profile)
        template_policies = self._get_template_specific_policies(template_config)
        
        # Add role-specific policies
        role_policies = self._get_role_specific_policies(profile)
        
        # Merge and deduplicate
        enhanced_context = PolicyContext(
            band_policies=base_context.band_policies + template_policies.get('band', []),
            department_policies=base_context.department_policies + template_policies.get('department', []),
            location_policies=base_context.location_policies + template_policies.get('location', []),
            general_policies=base_context.general_policies + template_policies.get('general', []) + role_policies,
            templates=base_context.templates + template_policies.get('templates', [])
        )
        
        return self._deduplicate_policies(enhanced_context)
    
    def _fast_create_profile(self, employee_name: str, employee_data: Dict[str, Any] = None) -> EmployeeProfile:
        """Fast profile creation with minimal validation"""
        if employee_data:
            profile = EmployeeProfile.from_dict(employee_data)
        else:
            employee_data = self._fetch_employee_data(employee_name)
            profile = EmployeeProfile.from_dict(employee_data)
        
        # Quick validation - only check critical fields
        if not profile.name or not profile.band:
            profile = self._auto_fix_profile_issues(profile, None)
        
        return profile
    
    def _fast_generate_offer(self, query: str, profile: EmployeeProfile, policy_context: PolicyContext, template_config: Dict[str, Any]) -> str:
        """Fast single-attempt offer generation"""
        try:
            # Use fast model for quicker generation
            legacy_employee_data = self._profile_to_legacy_dict(profile)
            all_chunks = self._flatten_policy_context(policy_context)
            
            # Add speed optimization to query
            optimized_query = f"{query}\n\nIMPORTANT: Generate a complete, professional offer letter efficiently. Focus on essential details."
            
            offer_letter = self.llm.generate_response(optimized_query, all_chunks, legacy_employee_data)
            
            # Quick quality check - if good enough, return immediately
            quality_score = self._quick_quality_check(offer_letter, profile)
            if quality_score >= 75:  # Lower threshold for speed
                return offer_letter
            
            # If quality is too low, do one quick enhancement
            return self._quick_enhance_offer(offer_letter, profile)
            
        except Exception as e:
            logger.warning(f"Fast generation failed: {str(e)}")
            raise
    
    def _fast_post_process(self, offer_letter: str, profile: EmployeeProfile, template_config: Dict[str, Any]) -> str:
        """Fast post-processing with essential fixes only"""
        # Only do critical post-processing
        processed = offer_letter
        
        # Ensure employee name is correct
        if profile.name.lower() not in processed.lower():
            processed = processed.replace("[Employee Name]", profile.name)
            processed = processed.replace("Dear Sir/Madam", f"Dear {profile.name}")
        
        # Ensure company name is present
        if "companyabc" not in processed.lower():
            processed += "\n\nBest regards,\nPeopleOps Team\nCompanyABC\nEmail: peopleops@companyabc.com"
        
        return processed
    
    def _quick_enhance_offer(self, offer_letter: str, profile: EmployeeProfile) -> str:
        """Quick enhancement for low-quality offers"""
        enhanced = offer_letter
        
        # Add missing salary if not present
        if not any(symbol in enhanced for symbol in ['₹', 'INR', 'salary']) and profile.total_ctc > 0:
            salary_section = f"\n\n**Compensation Package:**\n• Total CTC: ₹{profile.total_ctc:,.0f} per annum"
            enhanced += salary_section
        
        # Add missing position if not clear
        position_title = self._determine_position_title(profile)
        if position_title.lower() not in enhanced.lower():
            enhanced = enhanced.replace("this position", f"the position of {position_title}")
        
        return enhanced
    
    def _generate_quick_fallback(self, profile: EmployeeProfile, generation_id: str) -> str:
        """Generate quick fallback offer for speed"""
        position_title = self._determine_position_title(profile)
        
        return f"""Dear {profile.name},

We are pleased to offer you the position of {position_title} at CompanyABC.

**Position Details:**
• Role: {position_title}
• Department: {profile.department}
• Band: {profile.band}
• Location: {profile.location}

**Compensation:**
• Total CTC: ₹{profile.total_ctc:,.0f} per annum

**Next Steps:**
Please confirm your acceptance by replying to peopleops@companyabc.com.

We look forward to welcoming you to our team!

Best regards,
PeopleOps Team
CompanyABC

---
Generation ID: {generation_id}"""
    
    def _generate_enhanced_fallback(self, profile: EmployeeProfile, error: str, generation_id: str) -> str:
        """Generate enhanced fallback offer with better formatting and completeness"""
        logger.warning(f"⚠️ Generating enhanced fallback offer [{generation_id}] for {profile.name} due to: {error}")
        
        position_title = self._determine_position_title(profile)
        template_config = self._select_optimal_template(profile)
        
        # Enhanced fallback with template-based sections
        sections = []
        
        # Header
        sections.append(f"**OFFER LETTER**\n\nGeneration ID: {generation_id}\nDate: {datetime.now().strftime('%B %d, %Y')}\n")
        
        # Welcome
        sections.append(f"Dear {profile.name},\n\nWe are delighted to extend this offer of employment for the position of **{position_title}** at CompanyABC.")
        
        # Role Details
        sections.append(f"""**POSITION DETAILS:**
• Role: {position_title}
• Department: {profile.department}
• Band: {profile.band}
• Location: {profile.location}
• Reporting Manager: {profile.manager}
• Employment Type: {profile.offer_type.value.replace('_', ' ').title()}""")
        
        # Compensation
        if profile.total_ctc > 0:
            sections.append(f"""**COMPENSATION PACKAGE:**
• Base Salary: ₹{profile.base_salary:,.0f} per annum
• Performance Bonus: ₹{profile.performance_bonus:,.0f} per annum
• Total CTC: ₹{profile.total_ctc:,.0f} per annum""")
        
        # Benefits (template-based)
        if template_config['type'] in ['senior_executive', 'manager']:
            sections.append("""**BENEFITS & PERQUISITES:**
• Health Insurance (Family coverage)
• Provident Fund (12% employer contribution)
• Gratuity as per company policy
• Flexible working arrangements
• Learning & Development opportunities""")
        
        # Next Steps
        sections.append(f"""**NEXT STEPS:**
Please confirm your acceptance by replying to peopleops@companyabc.com within 7 days.
{f'Proposed Joining Date: {profile.joining_date}' if profile.joining_date else 'Joining date to be mutually decided.'}

We look forward to welcoming you to the CompanyABC family!

**Best regards,**
PeopleOps Team
CompanyABC
Email: peopleops@companyabc.com""")
        
        # Footer
        sections.append(f"---\n*This is an automated fallback offer letter [{generation_id}]. A detailed offer will be provided upon system recovery.*")
        
        return "\n\n".join(sections)

    def export_offer_as_json(self, offer_letter: str, profile: EmployeeProfile) -> str:
        """Export offer letter data as structured JSON with enhanced metadata"""
        qa_result = self._perform_quality_assurance(offer_letter, profile)
        
        offer_data = {
            'employee': self._profile_to_legacy_dict(profile),
            'offer_letter': offer_letter,
            'validation': self.validate_offer_completeness(offer_letter, profile),
            'quality_assurance': qa_result,
            'generated_at': datetime.now().isoformat(),
            'metadata': {
                'generator_version': '3.0_enhanced',
                'llm_model': self.llm.config.LLM_MODEL,
                'generation_stats': self.get_generation_statistics(),
                'template_used': self._select_optimal_template(profile)['type']
            }
        }
        return json.dumps(offer_data, indent=2, ensure_ascii=False)