from multiprocessing import context
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
from config.settings import Config
import time
import logging
import asyncio
from functools import wraps
import json
import re
from enum import Enum
from dataclasses import dataclass

# Setup for logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseType(Enum):
    OFFER_LETTER = "offer_letter"
    POLICY_QUERY = "policy_query"
    GENERAL = "general"
    SIMPLE = "simple"

@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_output_tokens: int = 3000
    top_p: float = 0.9
    top_k: int = 40
    candidate_count: int = 1
    stop_sequences: Optional[List[str]] = None

class LLMInterface:
    def __init__(self):
        self.config = Config()
        self._initialize_api()
        self._setup_models()
        self._setup_rate_limiting()
        self._setup_caching()

    def _initialize_api(self):
        try:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise

    def _setup_models(self):
        self.fast_generation_config = GenerationConfig(temperature=0.1, max_output_tokens=1000, top_p=0.8)
        self.comprehensive_generation_config = GenerationConfig(temperature=0.2, max_output_tokens=4000, top_p=0.9)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
        self.model_fast = self._create_model(self.fast_generation_config)
        self.model_comprehensive = self._create_model(self.comprehensive_generation_config)

    def _create_model(self, gen_config: GenerationConfig):
        return genai.GenerativeModel(
            model_name=self.config.LLM_MODEL or 'gemini-1.5-flash',
            generation_config={
                'temperature': gen_config.temperature,
                'max_output_tokens': gen_config.max_output_tokens,
                'top_p': gen_config.top_p,
                'top_k': gen_config.top_k,
                'candidate_count': gen_config.candidate_count,
            },
            safety_settings=self.safety_settings
        )

    def _setup_rate_limiting(self):
        self.last_request_time = 0
        self.min_request_interval = 0.5
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_minute = 60

    def _setup_caching(self):
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour to match OfferGenerator
        self.max_cache_size = 100
        self._cache_hits = 0
        self._total_requests = 0

    def rate_limit_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._advanced_rate_limit()
            return func(self, *args, **kwargs)
        return wrapper

    def _advanced_rate_limit(self):
        current_time = time.time()
        if current_time - self.request_window_start > 60:
            self.request_count = 0
            self.request_window_start = current_time
        if self.request_count >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
        self.request_count += 1
        self._total_requests += 1

    def _get_cache_key(self, query: str, context_hash: str = "") -> str:
        return f"{hash(query + context_hash)}"

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        if cache_key in self.response_cache:
            response_data = self.response_cache[cache_key]
            if time.time() - response_data['timestamp'] < self.cache_ttl:
                logger.info("Returning cached response")
                self._cache_hits += 1
                return response_data['response']
            else:
                del self.response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        if len(self.response_cache) >= self.max_cache_size:
            oldest_key = min(self.response_cache.keys(), key=lambda k: self.response_cache[k]['timestamp'])
            del self.response_cache[oldest_key]
        self.response_cache[cache_key] = {'response': response, 'timestamp': time.time()}

    def _classify_query_type(self, query: str) -> ResponseType:
        """Classify query type to align with OfferGenerator requirements."""
        query_lower = query.lower().strip()
        keywords = {
            ResponseType.OFFER_LETTER: ['offer letter', 'job offer', 'employment offer', 'compensation package', 'joining date'],
            ResponseType.POLICY_QUERY: ['policy', 'leave', 'benefit', 'travel', 'wfo', 'hr policy', 'vacation', 'insurance', 'entitlement'],
            ResponseType.SIMPLE: ['who', 'what', 'when', 'where', 'how many', 'list', 'define'],
        }

        match_counts = {rt: sum(1 for kw in kws if kw in query_lower) for rt, kws in keywords.items()}
        max_matches = max(match_counts.values(), default=0)

        if max_matches > 0:
            for rt, count in match_counts.items():
                if count == max_matches:
                    return rt
        return ResponseType.SIMPLE if len(query) < 50 else ResponseType.GENERAL

    def _rank_context_chunks(self, query: str, chunks: List[Dict[str, Any]], employee_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Rank context chunks by relevance to query and employee profile."""
        query_words = set(re.findall(r'\w+', query.lower()))
        band = employee_data.get('Band', '') if employee_data else ''
        department = employee_data.get('Department', '') if employee_data else ''
        location = employee_data.get('Location', '') if employee_data else ''
        offer_type = employee_data.get('Offer Type', '') if employee_data else ''

        def score_chunk(chunk: Dict[str, Any]) -> float:
            content = chunk.get('content', '').lower()
            section = chunk.get('section', '').lower()
            source = chunk.get('source', '').lower()
            content_words = set(re.findall(r'\w+', content + section + source))
            overlap = len(query_words.intersection(content_words))
            # Boost scores for relevant metadata
            boost = 1.0
            if 'policy' in source.lower():
                boost *= 1.5
            if band and band.lower() in content + section + source:
                boost *= 1.3
            if department and department.lower() in content + section + source:
                boost *= 1.2
            if location and location.lower() in content + section + source:
                boost *= 1.2
            if offer_type and offer_type.lower() in content + section + source:
                boost *= 1.1
            return overlap * boost

        ranked_chunks = sorted(chunks, key=score_chunk, reverse=True)
        return ranked_chunks[:5]  # Limit to top 5 to match OfferGenerator

    def _prepare_context(self, chunks: List[Dict[str, Any]], employee_data: Dict[str, Any] = None, 
                        simplified: bool = False) -> str:
        """Prepare context aligned with OfferGenerator's PolicyContext structure."""
        context_parts = []

        # Add employee data
        if employee_data:
            context_parts.append("### Employee Information")
            for key, value in employee_data.items():
                if key not in ['additional_data']:  # Exclude redundant fields
                    context_parts.append(f"- {key}: {value}")
            context_parts.append("")

        # Organize chunks by type to match PolicyContext
        band_policies = [c for c in chunks if 'band' in c.get('source', '').lower() or c.get('section', '').lower()]
        dept_policies = [c for c in chunks if employee_data.get('Department', '').lower() in c.get('source', '').lower() or c.get('section', '').lower()]
        loc_policies = [c for c in chunks if employee_data.get('Location', '').lower() in c.get('source', '').lower() or c.get('section', '').lower()]
        templates = [c for c in chunks if 'template' in c.get('source', '').lower()]
        general_policies = [c for c in chunks if c not in band_policies and c not in dept_policies and c not in loc_policies and c not in templates]

        # Rank chunks within each category
        ranked_chunks = []
        for chunk_group in [templates, band_policies, dept_policies, loc_policies, general_policies]:
            ranked_chunks.extend(self._rank_context_chunks("", chunk_group, employee_data))

        # Limit context for simple queries
        if simplified:
            ranked_chunks = ranked_chunks[:2]

        if ranked_chunks:
            context_parts.append("### Policy Context")
            for i, chunk in enumerate(ranked_chunks, 1):
                source = chunk.get('source', 'Policy Document')
                section = chunk.get('section', 'General')
                content = chunk.get('content', '').strip()
                
                if simplified:
                    context_parts.append(f"- From {source}: {content[:200]}...")
                else:
                    context_parts.append(f"\n#### Reference {i}")
                    context_parts.append(f"Source: {source}")
                    context_parts.append(f"Section: {section}")
                    context_parts.append(f"Content: {content}")
                    context_parts.append("-" * 40)

        if not ranked_chunks:
            context_parts.append("No specific policy context available. Use general company knowledge.")

        return "\n".join(context_parts)

    def _create_simple_prompt(self, query: str, context_chunks: List[Dict[str, Any]], 
                            employee_data: Dict[str, Any] = None) -> str:
        """Create optimized prompt for simple queries."""
        context = self._prepare_context(context_chunks, employee_data, simplified=True)
        
        return f"""### System Instructions
You are a concise HR assistant for {self.config.COMPANY_NAME}. Provide brief, accurate, and professional responses. If context is missing, provide a general answer and suggest contacting HR at hr@{self.config.COMPANY_NAME.lower()}.com.

### Context
{context}

### Query
{query}

### Task
- Answer in 2-3 sentences.
- Use bullet points or plain text for clarity.
- If unsure, say: "Please contact HR at hr@{self.config.COMPANY_NAME.lower()}.com."

### Constraints
- Keep response under 200 words.
- Avoid speculation.
- Use professional tone.
"""

    def _create_comprehensive_prompt(self, query: str, context_chunks: List[Dict[str, Any]], 
                                   employee_data: Dict[str, Any], query_type: ResponseType) -> str:
        """Create comprehensive prompt aligned with OfferGenerator requirements."""
        context = self._prepare_context(context_chunks, employee_data)
        system_prompt = self._get_system_prompt(query_type)
        task_instructions = self._get_task_instructions(query_type)

        # Add example for offer letters
        example = ""
        if query_type == ResponseType.OFFER_LETTER:
            example = f"""### Official Offer Letter
[{self.config.COMPANY_NAME}]

Date: {{date}}

Dear {{employee_name}},

We are excited to extend to you an offer for the position of **{{position}}** in our **{{department}}** team at **{self.config.COMPANY_NAME}**. We were impressed with your qualifications and believe you will be a valuable addition to our organization.

Please find the key details of the offer below:

- **Start Date**: {{joining_date}}
- **Annual Compensation**: ₹{{total_ctc:,.0f}}, which includes:
  - **Base Salary**: ₹{{base_salary:,.0f}}
  - **Performance Bonus**: ₹{{performance_bonus:,.0f}}
  - **Retention Bonus**: ₹{{retention_bonus:,.0f}}

- **Benefits**:
  - Comprehensive health insurance coverage
  - Provident Fund contributions
  - Leave policies aligned with the {{band}} band

- **Work Location**: {{location}}

- **Employment Terms**: Your employment will be on an *at-will* basis, governed by the policies and procedures of {self.config.COMPANY_NAME}.

- **Next Steps**: Please sign and return this letter by **{{deadline}}** to confirm your acceptance. Should you have any questions, feel free to contact our HR team at **hr@{self.config.COMPANY_NAME.lower()}.com**.

We look forward to welcoming you to the team and working together toward shared success.

Warm regards,  
[HR Name]  
Human Resources  
{self.config.COMPANY_NAME}
"""


        return f"""### System Instructions
{system_prompt}

### Context
{context}

### Query
{query}

### Task
{task_instructions}

{example}

### Constraints
- Use markdown for formatting (headers, bullets, tables).
- Ensure response is professional, complete, and actionable.
- If context is missing, provide a general response and suggest contacting HR at hr@{self.config.COMPANY_NAME.lower()}.com.
- For offer letters, include:
  - Company letterhead
  - Compensation breakdown
  - Benefits and policy details
  - Legal terms
  - Next steps and HR contact
- For policy queries, cite specific policy sections and provide actionable steps.
- Ensure response aligns with employee data (band, department, location).
"""

    def _get_system_prompt(self, query_type: ResponseType) -> str:
        if query_type == ResponseType.OFFER_LETTER:
            return f"""You are an expert HR writing assistant for {self.config.COMPANY_NAME}. Your sole task is to generate a professional, legally compliant offer letter based on the provided context and query. Do not provide guidance or summaries; write the complete letter.

Key requirements for the offer letter:
- Professional formatting (letterhead, salutation, closing)
- Personalized details (name, band, department, location, compensation)
- Comprehensive policy integration (leave, benefits, work arrangements)
- Clear terms, conditions, and next steps"""
        
        base_prompt = f"You are an expert HR assistant for {self.config.COMPANY_NAME}, specializing in policy interpretation and professional offer letter generation."
        
        if query_type == ResponseType.POLICY_QUERY:
            return f"""{base_prompt}
Provide accurate policy interpretations with:
- Specific policy section references
- Band/department/location-specific guidance
- Clear explanations and actionable steps
- Compliance with {self.config.COMPANY_NAME} standards"""
        
        else:
            return f"""{base_prompt}
Deliver accurate, helpful HR guidance with:
- Relevant policy references
- Practical solutions and next steps
- Professional and concise communication"""

    def _get_task_instructions(self, query_type: ResponseType) -> str:
        if query_type == ResponseType.OFFER_LETTER:
            return f"""Generate a complete offer letter in markdown format with:
- Company letterhead for {self.config.COMPANY_NAME}
- Personalized welcome message with employee name 
- Detailed compensation breakdown (base salary, bonuses, total CTC)
- Comprehensive benefits list (health, provident fund, leave policies)
- Work arrangements specific to department and location
- Legal terms (e.g., at-will employment)
- Clear next steps and HR contact (hr@{self.config.COMPANY_NAME.lower()}.com)
- Ensure alignment with {self.config.COMPANY_NAME} policies"""
        
        elif query_type == ResponseType.POLICY_QUERY:
            return f"""Answer the following policy question in markdown format. Provide a direct and concise answer with:
- Specific policy section references
- Band/department/location-specific details
- Step-by-step implementation guidance
- Contact information: hr@{self.config.COMPANY_NAME.lower()}.com"""
        
        else:
            return f"""Provide a helpful response in markdown format with:
- Relevant context and policy references
- Clear, actionable information
- Professional tone and structure
- Contact: hr@{self.config.COMPANY_NAME.lower()}.com"""

    @rate_limit_decorator
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], 
                         employee_data: Dict[str, Any] = None) -> str:
        """Generate response with validation and OfferGenerator compatibility."""
        context_hash = str(hash(str(context_chunks) + str(employee_data)))
        cache_key = self._get_cache_key(query, context_hash)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        query_type = self._classify_query_type(query)
        model = self.model_fast if query_type == ResponseType.SIMPLE else self.model_comprehensive
        prompt = self._create_simple_prompt(query, context_chunks, employee_data) if query_type == ResponseType.SIMPLE else self._create_comprehensive_prompt(query, context_chunks, employee_data, query_type)

        try:
            logger.info(f"Generating {query_type.value} response...")
            start_time = time.time()
            response = model.generate_content(prompt)
            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f} seconds")

            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name == "SAFETY":
                    logger.warning("Response blocked by safety filters")
                    return self._get_fallback_response(query_type)
                if hasattr(candidate, 'content') and candidate.content.parts:
                    result = response.text
                    # Post-process for offer letters
                    if query_type == ResponseType.OFFER_LETTER:
                        result = self._post_process_offer(result, employee_data)
                    self._cache_response(cache_key, result)
                    return result
                else:
                    logger.error("No content generated")
                    return self._get_fallback_response(query_type)
            else:
                logger.error("No candidates in response")
                return self._get_fallback_response(query_type)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._handle_api_error(e, query_type)

    def _post_process_offer(self, offer_letter: str, employee_data: Dict[str, Any] = None) -> str:
        """Post-process offer letter to ensure completeness and formatting."""
        processed = offer_letter.strip()
        
        # Ensure company branding
        if not processed.startswith('[') and self.config.COMPANY_NAME not in processed[:200]:
            from datetime import datetime
            header = f"""
[{self.config.COMPANY_NAME}]

Date: {datetime.now().strftime('%B %d, %Y')}

"""
            processed = header + processed

        # Ensure HR contact
        if f"hr@{self.config.COMPANY_NAME.lower()}.com" not in processed:
            processed += f"\n\nFor questions, contact HR at hr@{self.config.COMPANY_NAME.lower()}.com."

        # Validate completeness
        validation = self._validate_offer_completeness(processed, employee_data)
        if validation['status'] != 'valid':
            logger.warning(f"Offer letter incomplete: {validation['missing_elements']}")
            processed += f"\n\n**Note**: This offer letter may be incomplete. Missing: {', '.join(validation['missing_elements'])}. Please verify with HR."

        return processed

    def _validate_offer_completeness(self, offer_letter: str, employee_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate offer letter completeness to align with OfferGenerator."""
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
        validation_checks['has_contact_info'] = any(word in offer_lower for word in ['contact', 'email', 'phone', 'hr'])
        validation_checks['has_professional_format'] = len(offer_letter.split('\n')) > 10 and 'dear' in offer_lower
        validation_checks['has_company_branding'] = self.config.COMPANY_NAME.lower() in offer_lower
        validation_checks['has_benefits_details'] = any(word in offer_lower for word in ['health', 'insurance', 'medical', 'pf', 'provident'])
        validation_checks['has_legal_terms'] = any(word in offer_lower for word in ['terms', 'conditions', 'agreement', 'employment'])

        if employee_data:
            validation_checks['has_employee_name'] = employee_data.get('Name', '').lower() in offer_lower
            position = employee_data.get('Department', '').lower()  # Simplified; could use OfferGenerator's _determine_position_title
            validation_checks['has_position'] = position in offer_lower
        else:
            validation_checks['has_employee_name'] = True
            validation_checks['has_position'] = True

        total_checks = len(validation_checks)
        passed_checks = sum(validation_checks.values())
        completion_score = (passed_checks / total_checks) * 100
        status = 'valid' if completion_score >= 90 else 'incomplete' if completion_score >= 70 else 'error'

        return {
            'validation_checks': validation_checks,
            'completion_score': completion_score,
            'status': status,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'missing_elements': [check for check, passed in validation_checks.items() if not passed]
        }

    def _get_fallback_response(self, query_type: ResponseType) -> str:
        fallbacks = {
            ResponseType.OFFER_LETTER: f"Unable to generate offer letter. Please contact HR at hr@{self.config.COMPANY_NAME.lower()}.com for assistance.",
            ResponseType.POLICY_QUERY: f"Could not retrieve policy details. Please refer to the employee handbook or contact HR at hr@{self.config.COMPANY_NAME.lower()}.com.",
            ResponseType.GENERAL: f"Unable to provide a complete response. Please rephrase or contact HR at hr@{self.config.COMPANY_NAME.lower()}.com.",
            ResponseType.SIMPLE: "Could not process request. Please try rephrasing."
        }
        return fallbacks.get(query_type, "Unable to generate response. Please try again.")

    def _handle_api_error(self, error: Exception, query_type: ResponseType) -> str:
        error_str = str(error).lower()
        if "429" in error_str or "quota" in error_str:
            return "⚠️ Service temporarily busy. Please wait and try again."
        elif "timeout" in error_str:
            return "⚠️ Request timed out. Please simplify your query and try again."
        elif "connection" in error_str:
            return "⚠️ Connection issue. Please check your internet and try again."
        else:
            logger.error(f"Unexpected API error: {str(error)}")
            return self._get_fallback_response(query_type)

    @rate_limit_decorator
    def generate_simple_response(self, prompt: str) -> str:
        cache_key = self._get_cache_key(prompt)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        try:
            logger.info("Generating simple response...")
            response = self.model_fast.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                result = response.text
                self._cache_response(cache_key, result)
                return result
            else:
                return "Unable to generate response. Please try again."
        except Exception as e:
            logger.error(f"Error in simple response generation: {str(e)}")
            return self._handle_api_error(e, ResponseType.SIMPLE)

    def test_connection(self) -> bool:
        try:
            test_response = self.generate_simple_response(
                "Please respond with exactly: 'Connection successful'"
            )
            return "connection successful" in test_response.lower()
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.response_cache),
            "requests_this_minute": self.request_count,
            "cache_hit_ratio": self._cache_hits / max(self._total_requests, 1),
            "last_request_time": self.last_request_time
        }

    def clear_cache(self):
        self.response_cache.clear()
        logger.info("Response cache cleared")

class LLMMonitor:
    def __init__(self):
        self.error_count = 0
        self.success_count = 0
        self.response_times = []
        
    def log_success(self, response_time: float):
        self.success_count += 1
        self.response_times.append(response_time)
        
    def log_error(self):
        self.error_count += 1
        
    def get_stats(self) -> Dict[str, Any]:
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        total_requests = self.success_count + self.error_count
        success_rate = self.success_count / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "error_count": self.error_count
        }