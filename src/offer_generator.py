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
        """Generate offer letter with enhanced performance and error handling"""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Generating offer letter for {employee_name}")
            
            # Fetch employee data if not provided
            if not employee_data:
                employee_data = self._fetch_employee_data(employee_name)
            
            # Create structured employee profile
            employee_profile = EmployeeProfile.from_dict(employee_data)
            
            # Validate employee data
            validation_result = self._validate_employee_profile(employee_profile)
            if not validation_result.is_valid:
                return f"❌ Validation Error: {validation_result.error_message}"
            
            # Retrieve policies with caching
            policy_context = self._get_policy_context(employee_profile)
            
            # Generate optimized query
            query = self._create_enhanced_offer_query(employee_profile)
            
            # Convert to legacy format for LLM compatibility
            legacy_employee_data = self._profile_to_legacy_dict(employee_profile)
            all_chunks = self._flatten_policy_context(policy_context)
            
            # Generate offer letter
            offer_letter = self.llm.generate_response(query, all_chunks, legacy_employee_data)
            
            # Post-process and validate
            processed_offer = self._post_process_offer(offer_letter, employee_profile)
            
            # Update statistics
            generation_time = time.time() - start_time
            self._update_generation_stats(True, generation_time)
            
            logger.info(f"✅ Offer letter generated successfully in {generation_time:.2f}s")
            return processed_offer
            
        except Exception as e:
            generation_time = time.time() - start_time
            self._update_generation_stats(False, generation_time)
            logger.error(f"❌ Error generating offer letter for {employee_name}: {str(e)}")
            return self._generate_fallback_offer(employee_profile, str(e))

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
        """Create optimized query for offer letter generation"""
        joining_date = profile.joining_date or (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        position_title = self._determine_position_title(profile)
        
        query = f"""
Generate a comprehensive, professional offer letter for {profile.name} using the following specifications:

EMPLOYEE INFORMATION:
• Name: {profile.name}
• Position: {position_title}
• Band Level: {profile.band}
• Department: {profile.department}
• Location: {profile.location}
• Joining Date: {joining_date}
• Reporting Manager: {profile.manager}
• Offer Type: {profile.offer_type.value.replace('_', ' ').title()}

COMPENSATION BREAKDOWN:
• Base Salary: ₹{profile.base_salary:,.0f} per annum
• Performance Bonus: ₹{profile.performance_bonus:,.0f} per annum
• Retention Bonus: ₹{profile.retention_bonus:,.0f} per annum
• Total CTC: ₹{profile.total_ctc:,.0f} per annum

OFFER LETTER REQUIREMENTS:
1. Professional business format with company letterhead
2. Warm, welcoming opening paragraph
3. Clear position and compensation details
4. Comprehensive policy integration:
   - Leave policies specific to {profile.band} level
   - Work arrangements for {profile.department}
   - Location-specific policies for {profile.location}
   - Benefits and perks details
   - Performance evaluation framework
5. Legal terms and conditions
6. Clear next steps and contact information
7. Professional closing with signature block

FORMATTING STANDARDS:
• Use proper business letter structure
• Include clear section headings
• Bullet points for benefits and policies
• Professional tone throughout
• Proper spacing and readability

Generate a complete, ready-to-send offer letter that reflects CompanyABC's professionalism and includes all relevant policy details from the provided context.
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
        """Generate basic fallback offer letter"""
        position_title = self._determine_position_title(profile)
        return f"""
[CompanyABC]
123 Business Park, Bangalore, Karnataka, India
Date: {datetime.now().strftime('%B %d, %Y')}

Dear {profile.name},

We are pleased to offer you the position of {position_title} in our {profile.department} department.

Position Details:
• Role: {position_title}
• Band Level: {profile.band}
• Department: {profile.department}
• Location: {profile.location}
• Total CTC: ₹{profile.total_ctc:,.0f} per annum

This offer is subject to standard company policies and terms of employment.

Please contact HR at peopleops@companyabc.com for complete details and to proceed with the joining process.

Best regards,
Anita Desai
Head of Human Resources
CompanyABC

Note: This is a basic offer template generated due to system limitations. 
Error details: {error}
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

    def export_offer_as_json(self, offer_letter: str, profile: EmployeeProfile) -> str:
        """Export offer letter data as structured JSON"""
        offer_data = {
            'employee': self._profile_to_legacy_dict(profile),
            'offer_letter': offer_letter,
            'validation': self.validate_offer_completeness(offer_letter, profile),
            'generated_at': datetime.now().isoformat(),
            'metadata': {
                'generator_version': '2.0',
                'llm_model': self.llm.config.LLM_MODEL,
                'generation_stats': self.get_generation_statistics()
            }
        }
        return json.dumps(offer_data, indent=2, ensure_ascii=False)