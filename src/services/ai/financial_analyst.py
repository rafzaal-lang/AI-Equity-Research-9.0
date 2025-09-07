import os
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from functools import wraps
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class FinancialInsight:
    category: str  # "strength", "weakness", "opportunity", "threat"
    insight: str
    supporting_data: Dict[str, Any]
    confidence: float  # 0-1

@dataclass
class CompanyAnalysis:
    symbol: str
    executive_summary: str
    key_insights: List[FinancialInsight]
    investment_thesis: Dict[str, str]  # bull_case, bear_case
    analyst_rating: str  # "Buy", "Hold", "Sell"
    price_target_range: Dict[str, float]  # low, base, high

def rate_limit_openai(calls_per_minute=15):
    """Decorator to rate limit OpenAI API calls"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                logger.info(f"Rate limiting: waiting {left_to_wait:.2f} seconds")
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

class AIFinancialAnalyst:
    """AI-powered financial analysis that generates insights, not just calculations."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set - AI analysis will be disabled")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
    
    @rate_limit_openai(calls_per_minute=12)  # Conservative rate limiting
    def _make_openai_request(self, messages, max_tokens=1000, temperature=0.3):
        """Centralized OpenAI request with rate limiting and error handling"""
        if not self.client:
            logger.warning("OpenAI client not available - returning fallback response")
            return None
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Updated to use available model
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {type(e).__name__}: {e}")
            return None
        
    def analyze_company(self, symbol: str, financial_data: Dict[str, Any], 
                       filing_excerpts: List[str] = None) -> CompanyAnalysis:
        """Generate comprehensive AI analysis of a company."""
        
        if not self.client:
            logger.info(f"AI analysis disabled for {symbol} - returning basic analysis")
            return self._create_fallback_analysis(symbol, financial_data)
        
        try:
            # Extract key financial metrics
            fundamentals = financial_data.get("fundamentals", {})
            dcf = financial_data.get("dcf_valuation", {})
            comps = financial_data.get("comps", {})
            
            # Generate insights using AI with error handling
            insights = self._generate_key_insights(symbol, fundamentals, dcf, comps)
            executive_summary = self._generate_executive_summary(symbol, insights, fundamentals)
            investment_thesis = self._generate_investment_thesis(symbol, insights, financial_data)
            rating, price_targets = self._generate_rating_and_targets(symbol, dcf, insights)
            
            return CompanyAnalysis(
                symbol=symbol,
                executive_summary=executive_summary,
                key_insights=insights,
                investment_thesis=investment_thesis,
                analyst_rating=rating,
                price_target_range=price_targets
            )
        except Exception as e:
            logger.error(f"AI analysis failed for {symbol}: {e}")
            return self._create_fallback_analysis(symbol, financial_data)
    
    def _create_fallback_analysis(self, symbol: str, financial_data: Dict[str, Any]) -> CompanyAnalysis:
        """Create a basic analysis when AI is not available"""
        fundamentals = financial_data.get("fundamentals", {})
        dcf = financial_data.get("dcf_valuation", {})
        
        # Create basic insights based on financial metrics
        insights = []
        
        # Revenue insight
        revenue = fundamentals.get("revenue")
        if revenue and revenue > 1e9:
            insights.append(FinancialInsight(
                category="strength",
                insight=f"Large-scale operations with revenue of ${revenue/1e9:.1f}B",
                supporting_data={"revenue": revenue},
                confidence=0.9
            ))
        
        # Profitability insight
        net_margin = fundamentals.get("net_margin")
        if net_margin:
            if net_margin > 0.15:
                insights.append(FinancialInsight(
                    category="strength",
                    insight=f"Strong profitability with {net_margin:.1%} net margin",
                    supporting_data={"net_margin": net_margin},
                    confidence=0.8
                ))
            elif net_margin < 0:
                insights.append(FinancialInsight(
                    category="weakness",
                    insight=f"Negative profitability with {net_margin:.1%} net margin",
                    supporting_data={"net_margin": net_margin},
                    confidence=0.9
                ))
        
        # Determine basic rating
        if net_margin and net_margin > 0.1:
            rating = "Buy"
        elif net_margin and net_margin > 0:
            rating = "Hold"
        else:
            rating = "Hold"
        
        # Basic price targets based on DCF if available
        dcf_value = dcf.get("fair_value_per_share") if dcf else None
        if dcf_value:
            price_targets = {
                "low": dcf_value * 0.85,
                "base": dcf_value,
                "high": dcf_value * 1.15
            }
        else:
            price_targets = {"low": 0.0, "base": 0.0, "high": 0.0}
        
        return CompanyAnalysis(
            symbol=symbol,
            executive_summary=f"Analysis completed for {symbol}. Review fundamental metrics and valuation below.",
            key_insights=insights,
            investment_thesis={
                "bull_case": "Potential upside based on fundamental strength and market position.",
                "bear_case": "Consider market risks, competitive pressures, and economic headwinds."
            },
            analyst_rating=rating,
            price_target_range=price_targets
        )
    
    def _generate_key_insights(self, symbol: str, fundamentals: Dict, 
                              dcf: Dict, comps: Dict) -> List[FinancialInsight]:
        """AI identifies key financial insights from the data."""
        
        # Prepare financial summary for AI
        financial_summary = self._prepare_financial_summary(fundamentals, dcf, comps)
        
        prompt = f"""
        As a senior equity research analyst, analyze {symbol}'s financial data and identify 4-6 key insights.
        For each insight, categorize as strength/weakness/opportunity/threat and provide specific supporting evidence.
        
        Financial Data:
        {financial_summary}
        
        Provide insights in this JSON format:
        {{
            "insights": [
                {{
                    "category": "strength|weakness|opportunity|threat",
                    "insight": "Clear, specific insight in 1-2 sentences",
                    "supporting_data": "Specific metrics that support this insight",
                    "confidence": 0.85
                }}
            ]
        }}
        
        Focus on:
        - Profitability trends and margin analysis
        - Cash generation and capital efficiency  
        - Balance sheet strength and debt levels
        - Competitive positioning vs peers
        - Growth sustainability
        """
        
        response = self._make_openai_request([{"role": "user", "content": prompt}], max_tokens=1000)
        
        if not response:
            return []  # Return empty if AI fails
        
        try:
            import json
            result = json.loads(response)
            
            insights = []
            for insight_data in result.get("insights", []):
                insights.append(FinancialInsight(
                    category=insight_data.get("category", "neutral"),
                    insight=insight_data.get("insight", ""),
                    supporting_data=insight_data.get("supporting_data", ""),
                    confidence=insight_data.get("confidence", 0.7)
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error parsing insights for {symbol}: {e}")
            return []
    
    def _generate_executive_summary(self, symbol: str, insights: List[FinancialInsight], 
                                   fundamentals: Dict) -> str:
        """AI generates executive summary based on insights."""
        
        if not insights:
            return f"{symbol} analysis completed. Review detailed financial metrics below."
        
        insight_text = "\n".join([f"- {i.insight}" for i in insights])
        
        prompt = f"""
        Write a 3-4 sentence executive summary for {symbol} based on these key insights:
        
        {insight_text}
        
        Key metrics: Revenue ${fundamentals.get('revenue', 'N/A'):,}, 
        Net Margin {fundamentals.get('net_margin', 'N/A'):.1%}, 
        ROE {fundamentals.get('roe', 'N/A'):.1%}
        
        Style: Professional, concise, investment-focused. Lead with the investment conclusion.
        """
        
        response = self._make_openai_request([{"role": "user", "content": prompt}], max_tokens=200)
        
        if response:
            return response
        else:
            return f"{symbol} analysis completed. See detailed insights below."
    
    def _generate_investment_thesis(self, symbol: str, insights: List[FinancialInsight], 
                                   financial_data: Dict) -> Dict[str, str]:
        """AI generates bull and bear investment cases."""
        
        strengths = [i.insight for i in insights if i.category == "strength"]
        weaknesses = [i.insight for i in insights if i.category == "weakness"]
        opportunities = [i.insight for i in insights if i.category == "opportunity"]
        threats = [i.insight for i in insights if i.category == "threat"]
        
        dcf_value = financial_data.get("dcf_valuation", {}).get("enterprise_value", 0)
        
        bull_prompt = f"""
        Create a bull case for {symbol} based on:
        Strengths: {'; '.join(strengths)}
        Opportunities: {'; '.join(opportunities)}
        DCF Value: ${dcf_value:,.0f}
        
        Write 2-3 sentences focusing on upside catalysts and competitive advantages.
        """
        
        bear_prompt = f"""
        Create a bear case for {symbol} based on:
        Weaknesses: {'; '.join(weaknesses)}
        Threats: {'; '.join(threats)}
        
        Write 2-3 sentences focusing on downside risks and challenges.
        """
        
        bull_response = self._make_openai_request([{"role": "user", "content": bull_prompt}], max_tokens=150)
        bear_response = self._make_openai_request([{"role": "user", "content": bear_prompt}], max_tokens=150)
        
        return {
            "bull_case": bull_response or "Upside potential based on financial strengths.",
            "bear_case": bear_response or "Downside risks from market challenges."
        }
    
    def _generate_rating_and_targets(self, symbol: str, dcf: Dict, 
                                    insights: List[FinancialInsight]) -> tuple:
        """AI generates analyst rating and price targets."""
        
        # Simple scoring based on insights
        strength_count = len([i for i in insights if i.category in ["strength", "opportunity"]])
        weakness_count = len([i for i in insights if i.category in ["weakness", "threat"]])
        
        net_positive = strength_count - weakness_count
        
        if net_positive >= 2:
            rating = "Buy"
        elif net_positive >= 0:
            rating = "Hold"
        else:
            rating = "Sell"
        
        # Price targets based on DCF with adjustments
        base_value = dcf.get("enterprise_value", 100000000)
        
        price_targets = {
            "low": base_value * 0.85,
            "base": base_value,
            "high": base_value * 1.15
        }
        
        return rating, price_targets
    
    def _prepare_financial_summary(self, fundamentals: Dict, dcf: Dict, comps: Dict) -> str:
        """Prepare financial data summary for AI analysis."""
        
        def fmt_num(x, is_pct=False):
            if x is None:
                return "N/A"
            if is_pct:
                return f"{x:.1%}"
            return f"${x:,.0f}" if x > 1000 else f"{x:.2f}"
        
        summary = f"""
        Revenue: {fmt_num(fundamentals.get('revenue'))}
        Net Income: {fmt_num(fundamentals.get('net_income'))}
        Free Cash Flow: {fmt_num(fundamentals.get('fcf'))}
        
        Margins:
        - Gross: {fmt_num(fundamentals.get('gross_margin'), True)}
        - Operating: {fmt_num(fundamentals.get('op_margin'), True)}
        - Net: {fmt_num(fundamentals.get('net_margin'), True)}
        
        Returns:
        - ROE: {fmt_num(fundamentals.get('roe'), True)}
        - ROIC: {fmt_num(fundamentals.get('roic'), True)}
        
        Valuation:
        - DCF Enterprise Value: {fmt_num(dcf.get('enterprise_value'))}
        - Peer Average P/E: {fmt_num(self._get_peer_average_pe(comps))}
        """
        
        return summary
    
    def _get_peer_average_pe(self, comps: Dict) -> Optional[float]:
        """Calculate average P/E from peer data."""
        peers = comps.get("peers", [])
        pe_values = [p.get("pe") for p in peers if p.get("pe") is not None]
        return sum(pe_values) / len(pe_values) if pe_values else None

# Global instance
ai_financial_analyst = AIFinancialAnalyst()
