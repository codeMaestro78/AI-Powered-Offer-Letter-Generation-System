from datetime import datetime
from typing import Dict, List, Optional,Any

from config.settings import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OfferTemplate:
    def __init__(self):
        """Initialize OfferTemplate with configuration."""
        try:
            self.config = Config()
            self.company_name = getattr(self.config, 'COMPANY_NAME', 'CompanyABC')
            self.company_email = getattr(self.config, 'COMPANY_EMAIL', 'peopleops@companyabc.com')
            self.company_website = getattr(self.config, 'COMPANY_WEBSITE', 'https://www.companyabc.com')
        except AttributeError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise ValueError("Invalid configuration: Missing required attributes in Config")

    def _validate_employee_data(self, employee_data: Dict[str, str]) -> Dict[str, str]:
        """Validate employee data and provide defaults for optional fields."""
        required_fields = ['candidate_name', 'position', 'joining_date']
        optional_fields = {
            'band_level': 'Not Specified',
            'location': 'Not Specified',
            'department': 'Not Specified',
            'base_salary': 'TBD',
            'performance_bonus': 'TBD',
            'retention_bonus': 'TBD',
            'total_ctc': 'TBD'
        }
        validated_data = employee_data.copy() if employee_data else {}
        for field in required_fields:
            if not validated_data.get(field):
                logger.warning(f"Missing required field: {field}. Using placeholder.")
                validated_data[field] = f'[{field.capitalize()}]'
        for field, default in optional_fields.items():
            validated_data.setdefault(field, default)
        return validated_data

    def _extract_policy_content(self, context_chunks: List[Dict[str, str]], policy_type: str, employee_data: Dict[str, str]) -> str:
        """Extract relevant policy content from context chunks, using general policies if no employee data."""
        if not context_chunks:
            logger.warning(f"No context chunks provided for {policy_type}. Using default.")
            return self._get_default_policy(policy_type, employee_data)
        
        band = employee_data.get('band_level', '').lower() if employee_data else ''
        department = employee_data.get('department', '').lower() if employee_data else ''
        location = employee_data.get('location', '').lower() if employee_data else ''
        
        relevant_chunks = [
            chunk for chunk in context_chunks 
            if policy_type.lower() in chunk.get('section', '').lower() or 
               policy_type.lower() in chunk.get('content', '').lower()
        ]
        if not relevant_chunks and not employee_data:
            return self._get_default_policy(policy_type, employee_data)
        
        # Prioritize chunks matching employee data
        scored_chunks = []
        for chunk in relevant_chunks:
            content = chunk.get('content', '').lower()
            section = chunk.get('section', '').lower()
            score = 1.0
            if band and band in content + section:
                score *= 1.5
            if department and department in content + section:
                score *= 1.3
            if location and location in content + section:
                score *= 1.2
            scored_chunks.append((chunk, score))
        
        # Sort by score and take the top chunk
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        if scored_chunks:
            return scored_chunks[0][0].get('content', self._get_default_policy(policy_type, employee_data))
        return self._get_default_policy(policy_type, employee_data)

    def _get_default_policy(self, policy_type: str, employee_data: Dict[str, str]) -> str:
        """Return default policy content when specific data is unavailable."""
        band = employee_data.get('band_level', 'Not Specified') if employee_data else 'Not Specified'
        department = employee_data.get('department', 'Not Specified') if employee_data else 'Not Specified'
        location = employee_data.get('location', 'Not Specified') if employee_data else 'Not Specified'
        
        if policy_type == 'leave_policy':
            return f"- Annual Leave: 18 days total (10 earned, 6 sick, 2 casual)\n- Application: Submit via HR portal"
        elif policy_type == 'wfo_policy':
            return f"- {location}: Minimum 3 days work-from-office per week"
        elif policy_type == 'travel_policy':
            return f"- Eligible for business travel as per {self.company_name} policy\n- Contact HR for details"
        return "Contact HR at {} for details.".format(self.company_email)

    def get_base_template(self, employee_data: Dict[str, str] = None, context_chunks: List[Dict[str, str]] = None) -> str:
        """Return a formatted offer letter template with dynamic policy integration."""
        employee_data = self._validate_employee_data(employee_data or {})
        context_chunks = context_chunks or []
        
        # Extract policies dynamically
        leave_policy = self._extract_policy_content(context_chunks, 'leave_policy', employee_data)
        wfo_policy = self._extract_policy_content(context_chunks, 'wfo_policy', employee_data)
        travel_policy = self._extract_policy_content(context_chunks, 'travel_policy', employee_data)
        
        # Format financials with proper INR formatting
        def format_currency(value: str) -> str:
            try:
                return f"₹{float(value):,.0f}" if value != 'TBD' else 'TBD'
            except ValueError:
                logger.warning(f"Invalid currency value: {value}. Using TBD.")
                return 'TBD'

        template = f"""# Offer Letter – {self.company_name}

**Date**: {datetime.now().strftime('%B %d, %Y')}

**Dear {employee_data['candidate_name']},**

## 1. Appointment Details
We are delighted to offer you the position of **{employee_data['position']}** in the **{employee_data['department']}** team at {self.company_name}. This is a full-time role based out of our **{employee_data['location']}** office, starting on **{employee_data['joining_date']}**. Your employment will be governed by the terms outlined in this letter and the {self.company_name} Employee Handbook.

## 2. Compensation Structure
| Component            | Annual (INR) |
|---------------------|--------------|
| Fixed Salary        | {format_currency(employee_data['base_salary'])} |
| Performance Bonus   | {format_currency(employee_data['performance_bonus'])} |
| Retention Bonus     | {format_currency(employee_data['retention_bonus'])} |
| **Total CTC**       | **{format_currency(employee_data['total_ctc'])}** |

*Performance bonuses are disbursed quarterly, subject to performance evaluation.*

## 3. Leave Entitlements (Band {employee_data['band_level']})
{leave_policy}

## 4. Work From Office Policy ({employee_data['department']} Team)
{wfo_policy}

## 5. Travel Policy (Band {employee_data['band_level']})
{travel_policy}

## 6. Confidentiality & Intellectual Property
You are required to maintain strict confidentiality of all proprietary data, financials, codebases, and client information. All work products created during your employment shall remain the intellectual property of {self.company_name}. A separate Non-Disclosure Agreement (NDA) and Intellectual Property Agreement will be provided upon acceptance.

## 7. Termination & Exit
- Either party may terminate employment with **60 days' notice**.
- During the probation period (first 3 months), a **15-day notice period** applies.
- All company property and access must be returned on your final working day.

## 8. Next Steps
Please confirm your acceptance by signing and returning this letter via DocuSign within **5 working days**. Upon acceptance, your onboarding buddy and People Ops partner will contact you for pre-joining formalities.

For any questions, please contact HR at {self.company_email}.

**Warm regards,**  
HR Business Partner  
{self.company_name}  
📧 {self.company_email}  
🌐 {self.company_website}
"""
        # Validate template completeness
        validation = self._validate_template(template, employee_data)
        if validation['status'] != 'valid':
            logger.warning(f"Offer letter incomplete: {validation['missing_elements']}")
            template += f"\n\n**Note**: This offer letter may be incomplete. Missing: {', '.join(validation['missing_elements'])}. Please verify with HR."
        
        return template.strip()

    def _validate_template(self, template: str, employee_data: Dict[str, str]) -> Dict[str, Any]:
        """Validate the completeness of the generated offer letter."""
        validation_checks = {
            'has_candidate_name': employee_data['candidate_name'] != '[Candidate Name]' and employee_data['candidate_name'].lower() in template.lower(),
            'has_position': employee_data['position'] != '[Position]' and employee_data['position'].lower() in template.lower(),
            'has_joining_date': employee_data['joining_date'] != '[Joining Date]' and employee_data['joining_date'].lower() in template.lower(),
            'has_company_branding': self.company_name.lower() in template.lower(),
            'has_contact_info': self.company_email.lower() in template.lower(),
            'has_policy_information': any(word in template.lower() for word in ['leave', 'wfo', 'travel', 'policy']),
            'has_compensation_details': any(word in template.lower() for word in ['salary', 'bonus', 'ctc', '₹']),
            'has_legal_terms': any(word in template.lower() for word in ['confidentiality', 'termination', 'intellectual property']),
            'has_professional_format': len(template.split('\n')) > 20 and 'dear' in template.lower()
        }
        passed_checks = sum(validation_checks.values())
        total_checks = len(validation_checks)
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