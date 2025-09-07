import os
import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class RiskFactor:
    category: str  # "operational", "financial", "regulatory", "competitive", "market"
    description: str
    severity: str  # "high", "medium", "low"
    likelihood: str  # "high", "medium", "low"
    potential_impact: str
    mitigation_factors: List[str]

class AIRiskAnalyzer:
    """AI-powered analysis of business risks from SEC filings and financial data."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze_filing_risks(self, filing_text: str, symbol: str) -> List[RiskFactor]:
        """Extract and analyze risk factors from SEC filings using AI."""
        
        # Extract risk factors section
        risk_section = self._extract_risk_section(filing_text)
        
        if not risk_section:
            return []
        
        # Use AI to analyze and categorize risks
        prompt = f"""
        Analyze the following risk factors from {symbol}'s SEC filing and identify the 5 most significant risks.
        
        Risk Factors Text:
        {risk_section[:4000]}  # Limit to avoid token limits
        
        For each risk, provide:
        1. Category (operational/financial/regulatory/competitive/market)
        2. Clear description (1-2 sentences)
        3. Severity assessment (high/medium/low)
        4. Likelihood assessment (high/medium/low)
        5. Potential business impact
        6. Any mitigation factors mentioned
        
        Return as JSON:
        {{
            "risks": [
                {{
                    "category": "operational",
                    "description": "Clear risk description",
                    "severity": "high",
                    "likelihood": "medium", 
                    "potential_impact": "Impact on business",
                    "mitigation_factors": ["factor1", "factor2"]
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            risks = []
            for risk_data in result["risks"]:
                risks.append(RiskFactor(
                    category=risk_data["category"],
                    description=risk_data["description"],
                    severity=risk_data["severity"],
                    likelihood=risk_data["likelihood"],
                    potential_impact=risk_data["potential_impact"],
                    mitigation_factors=risk_data.get("mitigation_factors", [])
                ))
            
            return risks
            
        except Exception as e:
            logger.error(f"Error analyzing risks for {symbol}: {e}")
            return []
    
    def _extract_risk_section(self, filing_text: str) -> str:
        """Extract risk factors section from SEC filing."""
        
        # Common patterns for risk factor sections
        risk_patterns = [
            r"item\s+1a\.?\s*risk\s+factors(.*?)item\s+1b",
            r"risk\s+factors(.*?)(?:item\s+\d|unresolved|legal)",
            r"principal\s+risks(.*?)(?:item\s+\d|forward.looking)"
        ]
        
        filing_lower = filing_text.lower()
        
        for pattern in risk_patterns:
            match = re.search(pattern, filing_lower, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)[:10000]  # Return first 10k chars
        
        # Fallback: look for any mention of risks
        risk_keywords = ["risk", "uncertainty", "challenge", "threat"]
        for keyword in risk_keywords:
            if keyword in filing_lower:
                start_idx = filing_lower.find(keyword)
                return filing_text[start_idx:start_idx + 5000]
        
        return ""
    
    def analyze_financial_risks(self, financial_data: Dict[str, Any]) -> List[RiskFactor]:
        """Identify financial risks from quantitative data."""
        
        risks = []
        fundamentals = financial_data.get("fundamentals", {})
        
        # High debt risk
        de_ratio = fundamentals.get("de_ratio")
        if de_ratio and de_ratio > 2.0:
            risks.append(RiskFactor(
                category="financial",
                description=f"High debt-to-equity ratio of {de_ratio:.1f}x indicates elevated financial leverage",
                severity="high" if de_ratio > 3.0 else "medium",
                likelihood="high",
                potential_impact="Increased interest expense and refinancing risk during market stress",
                mitigation_factors=["Strong cash generation", "Diversified funding sources"]
            ))
        
        # Low profitability risk
        net_margin = fundamentals.get("net_margin")
        if net_margin and net_margin < 0.05:
            risks.append(RiskFactor(
                category="operational",
                description=f"Low net margin of {net_margin:.1%} suggests operational inefficiency",
                severity="medium" if net_margin > 0 else "high",
                likelihood="high",
                potential_impact="Vulnerability to cost inflation and competitive pressure",
                mitigation_factors=["Cost reduction initiatives", "Pricing power"]
            ))
        
        # Cash flow risk
        fcf = fundamentals.get("fcf")
        if fcf and fcf < 0:
            risks.append(RiskFactor(
                category="financial",
                description="Negative free cash flow indicates cash consumption",
                severity="high",
                likelihood="high", 
                potential_impact="Potential need for external financing or asset sales",
                mitigation_factors=["Access to credit facilities", "Asset monetization options"]
            ))
        
        return risks

# Global instance
ai_risk_analyzer = AIRiskAnalyzer()
