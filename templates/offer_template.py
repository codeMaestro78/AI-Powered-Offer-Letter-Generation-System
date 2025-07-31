from datetime import datetime
from typing import Dict, List, Optional, Any

from config.settings import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OfferTemplate:
    def __init__(self):
        """Initialize OfferTemplate with company configuration."""
        try:
            self.config = Config()
            self.company_name = getattr(self.config, 'COMPANY_NAME', 'CompanyABC')
            self.company_email = getattr(self.config, 'COMPANY_EMAIL', 'peopleops@companyabc.com')
            self.company_website = getattr(self.config, 'COMPANY_WEBSITE', 'https://www.companyabc.com')
        except AttributeError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise ValueError("Invalid configuration: Missing required attributes in Config")

    def _validate_employee_data(self, employee_data: Dict[str, str]) -> Dict[str, str]:
        """Ensure required fields are present and fill optional defaults."""
        required = ['candidate_name', 'position', 'joining_date']
        optional_defaults = {
            'band_level': 'Not Specified',
            'location': 'Not Specified',
            'department': 'Not Specified',
            'base_salary': 'TBD',
            'performance_bonus': 'TBD',
            'retention_bonus': 'TBD',
            'total_ctc': 'TBD'
        }
        validated = employee_data.copy() if employee_data else {}

        for field in required:
            if not validated.get(field):
                logger.warning(f"Missing required field: {field}. Using placeholder.")
                validated[field] = f'[{field.replace("_", " ").title()}]'

        for field, default in optional_defaults.items():
            validated.setdefault(field, default)

        return validated

    def _extract_policy_content(self, context_chunks: List[Dict[str, str]], policy_type: str, employee_data: Dict[str, str]) -> str:
        """Fetch the most relevant policy content or fallback to default."""
        if not context_chunks:
            logger.warning(f"No context provided for {policy_type}. Using default policy.")
            return self._get_default_policy(policy_type, employee_data)

        def matches_chunk(chunk: Dict[str, str]) -> bool:
            return policy_type.lower() in chunk.get('section', '').lower() or policy_type.lower() in chunk.get('content', '').lower()

        filtered = [chunk for chunk in context_chunks if matches_chunk(chunk)]
        if not filtered:
            return self._get_default_policy(policy_type, employee_data)

        # Score relevance
        band = employee_data.get('band_level', '').lower()
        dept = employee_data.get('department', '').lower()
        loc = employee_data.get('location', '').lower()

        def score_chunk(chunk: Dict[str, str]) -> float:
            content = (chunk.get('section', '') + chunk.get('content', '')).lower()
            score = 1.0
            if band and band in content: score *= 1.5
            if dept and dept in content: score *= 1.3
            if loc and loc in content: score *= 1.2
            return score

        scored = sorted(filtered, key=score_chunk, reverse=True)
        return scored[0].get('content', self._get_default_policy(policy_type, employee_data))

    def _get_default_policy(self, policy_type: str, employee_data: Dict[str, str]) -> str:
        """Provide fallback/default policy text."""
        loc = employee_data.get('location', 'Not Specified')
        if policy_type == 'leave_policy':
            return "- Annual Leave: 18 days total (10 earned, 6 sick, 2 casual)\n- Application: Submit via HR portal"
        elif policy_type == 'wfo_policy':
            return f"- {loc}: Minimum 3 days work-from-office per week"
        elif policy_type == 'travel_policy':
            return f"- Eligible for business travel as per {self.company_name} policy\n- Contact HR for details"
        return f"- For {policy_type}, please contact HR at {self.company_email}"

    def _format_currency(self, value: str) -> str:
        """Safely format currency string into INR format."""
        try:
            return f"₹{float(value):,.0f}" if value != 'TBD' else 'TBD'
        except (ValueError, TypeError):
            logger.warning(f"Invalid currency input: {value}. Using 'TBD'.")
            return 'TBD'

    def _validate_template(self, template: str, employee_data: Dict[str, str]) -> Dict[str, Any]:
        """Check if the generated template is sufficiently complete."""
        checks = {
            'has_candidate_name': '[Candidate Name]' not in employee_data['candidate_name'] and employee_data['candidate_name'].lower() in template.lower(),
            'has_position': '[Position]' not in employee_data['position'] and employee_data['position'].lower() in template.lower(),
            'has_joining_date': '[Joining Date]' not in employee_data['joining_date'] and employee_data['joining_date'].lower() in template.lower(),
            'has_company_branding': self.company_name.lower() in template.lower(),
            'has_contact_info': self.company_email.lower() in template.lower(),
            'has_policy_information': any(x in template.lower() for x in ['leave', 'wfo', 'travel', 'policy']),
            'has_compensation_details': any(x in template.lower() for x in ['salary', 'bonus', 'ctc', '₹']),
            'has_legal_terms': any(x in template.lower() for x in ['confidentiality', 'termination', 'intellectual property']),
            'has_professional_format': len(template.splitlines()) > 20 and 'dear' in template.lower()
        }

        passed = sum(checks.values())
        total = len(checks)
        score = (passed / total) * 100
        status = 'valid' if score >= 90 else 'incomplete' if score >= 70 else 'error'

        return {
            'validation_checks': checks,
            'completion_score': score,
            'status': status,
            'passed_checks': passed,
            'total_checks': total,
            'missing_elements': [k for k, v in checks.items() if not v]
        }

    def get_base_template(self, employee_data: Dict[str, str] = None, context_chunks: List[Dict[str, str]] = None) -> str:
        """Generate a complete offer letter based on input and policy context."""
        employee_data = self._validate_employee_data(employee_data or {})
        context_chunks = context_chunks or []

        # Extract policy content
        leave_policy = self._extract_policy_content(context_chunks, 'leave_policy', employee_data)
        wfo_policy = self._extract_policy_content(context_chunks, 'wfo_policy', employee_data)
        travel_policy = self._extract_policy_content(context_chunks, 'travel_policy', employee_data)

        # Format output letter
        template = f"""# Offer Letter – {self.company_name}

**Date**: {datetime.now().strftime('%B %d, %Y')}

**Dear {employee_data['candidate_name']},**

## 1. Appointment Details
We are pleased to offer you the role of **{employee_data['position']}** in our **{employee_data['department']}** team at {self.company_name}. This is a full-time position based at our **{employee_data['location']}** office, with your expected joining date being **{employee_data['joining_date']}**.

## 2. Compensation Structure
| Component            | Annual (INR) |
|---------------------|--------------|
| Fixed Salary        | {self._format_currency(employee_data['base_salary'])} |
| Performance Bonus   | {self._format_currency(employee_data['performance_bonus'])} |
| Retention Bonus     | {self._format_currency(employee_data['retention_bonus'])} |
| **Total CTC**       | **{self._format_currency(employee_data['total_ctc'])}** |

*Bonuses are performance-linked and disbursed as per company policy.*

## 3. Leave Policy (Band: {employee_data['band_level']})
{leave_policy}

## 4. Work From Office Guidelines ({employee_data['department']} Department)
{wfo_policy}

## 5. Travel Policy (Band: {employee_data['band_level']})
{travel_policy}

## 6. Confidentiality & Intellectual Property
You are expected to maintain strict confidentiality of all company-related information. Any intellectual property created during your tenure shall remain the property of {self.company_name}.

## 7. Termination Terms
- 60 days' notice by either party
- 15 days' notice during the 3-month probation period
- Final clearance required for exit formalities

## 8. Next Steps
Please confirm your acceptance via DocuSign within 5 working days. Our HR team will follow up with onboarding details.

**For queries, contact us at**: {self.company_email}

**Warm regards,**  
HR Business Partner  
{self.company_name}  
📧 {self.company_email}  
🌐 {self.company_website}
"""

        # Final validation
        validation = self._validate_template(template, employee_data)
        if validation['status'] != 'valid':
            logger.warning(f"Incomplete offer template. Missing: {', '.join(validation['missing_elements'])}")
            template += f"\n\n---\n**Note**: This letter may be incomplete. Please verify manually.\nMissing elements: {', '.join(validation['missing_elements'])}"

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