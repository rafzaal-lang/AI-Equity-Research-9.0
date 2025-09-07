import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
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

class AIFinancialAnalyst:
    """AI-powered financial analysis that generates insights, not just calculations."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def analyze_company(self, symbol: str, financial_data: Dict[str, Any], 
                       filing_excerpts: List[str] = None) -> CompanyAnalysis:
        """Generate comprehensive AI analysis of a company."""
        
        # Extract key financial metrics
        fundamentals = financial_data.get("fundamentals", {})
        dcf = financial_data.get("dcf_valuation", {})
        comps = financial_data.get("comps", {})
        
        # Generate insights using AI
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
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            insights = []
            for insight_data in result["insights"]:
                insights.append(FinancialInsight(
                    category=insight_data["category"],
                    insight=insight_data["insight"],
                    supporting_data=insight_data["supporting_data"],
                    confidence=insight_data.get("confidence", 0.7)
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights for {symbol}: {e}")
            return []
    
    def _generate_executive_summary(self, symbol: str, insights: List[FinancialInsight], 
                                   fundamentals: Dict) -> str:
        """AI generates executive summary based on insights."""
        
        insight_text = "\n".join([f"- {i.insight}" for i in insights])
        
        prompt = f"""
        Write a 3-4 sentence executive summary for {symbol} based on these key insights:
        
        {insight_text}
        
        Key metrics: Revenue ${fundamentals.get('revenue', 'N/A'):,}, 
        Net Margin {fundamentals.get('net_margin', 'N/A'):.1%}, 
        ROE {fundamentals.get('roe', 'N/A'):.1%}
        
        Style: Professional, concise, investment-focused. Lead with the investment conclusion.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating executive summary for {symbol}: {e}")
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
        
        try:
            bull_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": bull_prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            bear_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": bear_prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            return {
                "bull_case": bull_response.choices[0].message.content.strip(),
                "bear_case": bear_response.choices[0].message.content.strip()
            }
        except Exception as e:
            logger.error(f"Error generating investment thesis for {symbol}: {e}")
            return {
                "bull_case": "Upside potential based on financial strengths.",
                "bear_case": "Downside risks from market challenges."
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
