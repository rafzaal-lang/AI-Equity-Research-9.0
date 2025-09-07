# src/services/ai/financial_analyst.py
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from functools import wraps
from openai import OpenAI

from src.services.llm.json_client import chat_json
from src.services.data.fundamentals_agg import fetch_fundamentals, robust_peer_pe

logger = logging.getLogger(__name__)

@dataclass
class FinancialInsight:
    category: str  # "strength", "weakness", "opportunity", "threat"
    insight: str
    supporting_data: Dict[str, Any] | str
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
        """Centralized OpenAI request with rate limiting and error handling (text responses)"""
        if not self.client:
            logger.warning("OpenAI client not available - returning fallback response")
            return None
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {type(e).__name__}: {e}")
            return None

    @rate_limit_openai(calls_per_minute=12)
    def _make_openai_request_json(self, messages, max_tokens=1200, temperature=0.2) -> Dict[str, Any]:
        """JSON-enforced responses using the json_client wrapper."""
        if not self.client:
            logger.warning("OpenAI client not available - returning fallback JSON")
            return {}
        try:
            return chat_json(
                messages,
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                temperature=temperature,
                enforce_json_mode=True
            )
        except Exception as e:
            logger.error(f"OpenAI JSON call failed: {type(e).__name__}: {e}")
            return {}

    def analyze_company(self, symbol: str, financial_data: Dict[str, Any], 
                       filing_excerpts: List[str] = None) -> CompanyAnalysis:
        """Generate comprehensive AI analysis of a company."""
        
        if not self.client:
            logger.info(f"AI analysis disabled for {symbol} - returning basic analysis")
            return self._create_fallback_analysis(symbol, financial_data)
        
        try:
            # pull fields from pipeline
            fundamentals = (financial_data.get("fundamentals") or {}).copy()
            dcf = financial_data.get("dcf_valuation") or financial_data.get("dcf") or {}
            comps = (financial_data.get("comps") or {}).copy()

            # --- auto-enrich fundamentals if thin/empty ---
            core_keys = ["revenue", "net_income", "net_margin"]
            if not any(k in fundamentals and isinstance(fundamentals.get(k), (int, float)) for k in core_keys):
                logger.info("Enriching fundamentals from FMP for %s", symbol)
                fetched = fetch_fundamentals(symbol)
                fundamentals.update({k: v for k, v in fetched.items() if v is not None})

            # robust peer P/E if comps exist
            if "peers" in comps and isinstance(comps["peers"], list):
                rpe = robust_peer_pe(comps["peers"])
                if rpe is not None:
                    comps["peer_pe_robust"] = rpe
                    # prefer robust value as the peer average used in prompts
                    fundamentals.setdefault("pe", rpe)

            # write back enriched data for composer to show clean KPIs
            financial_data["fundamentals"] = fundamentals
            financial_data["comps"] = comps

            # --- AI pipeline ---
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
        fundamentals = financial_data.get("fundamentals", {}) or {}
        dcf = financial_data.get("dcf_valuation", {}) or {}
        
        insights: List[FinancialInsight] = []
        
        # Revenue insight
        revenue = fundamentals.get("revenue")
        if isinstance(revenue, (int, float)) and revenue > 1e9:
            insights.append(FinancialInsight(
                category="strength",
                insight=f"Large-scale operations with revenue of ${revenue/1e9:.1f}B",
                supporting_data={"revenue": revenue},
                confidence=0.9
            ))
        
        # Profitability insight
        net_margin = fundamentals.get("net_margin")
        if isinstance(net_margin, (int, float)):
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
        if isinstance(net_margin, (int, float)) and net_margin > 0.1:
            rating = "Buy"
        elif isinstance(net_margin, (int, float)) and net_margin > 0:
            rating = "Hold"
        else:
            rating = "Hold"
        
        # Basic price targets based on DCF if available
        dcf_value = dcf.get("fair_value_per_share") if dcf else None
        if isinstance(dcf_value, (int, float)):
            price_targets = {
                "low": float(dcf_value) * 0.85,
                "base": float(dcf_value),
                "high": float(dcf_value) * 1.15
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
        """AI identifies key financial insights from the data, robust JSON."""
        
        financial_summary = self._prepare_financial_summary(fundamentals, dcf, comps)
        
        schema_hint = (
            "Return a single JSON object with this shape:\n"
            "{\n"
            "  \"insights\": [\n"
            "    {\n"
            "      \"category\": \"strength|weakness|opportunity|threat\",\n"
            "      \"insight\": \"Clear, specific insight in 1-2 sentences\",\n"
            "      \"supporting_data\": { \"metric\": \"value or note\" } ,\n"
            "      \"confidence\": 0.0\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not include any extra keys."
        )
        
        messages = [
            {"role": "system", "content": "You are a precise equity research co-pilot. Always return strict JSON."},
            {"role": "user", "content": (
                f"Analyze {symbol}'s financial data and identify 4-6 key insights.\n"
                f"Focus on profitability trends, cash generation, balance sheet, competitive stance, and growth sustainability.\n\n"
                f"Financial Data:\n{financial_summary}\n\n"
                f"{schema_hint}"
            )}
        ]

        result = self._make_openai_request_json(messages, max_tokens=1200, temperature=0.2) or {}
        raw_insights = result.get("insights") if isinstance(result, dict) else None
        if not isinstance(raw_insights, list):
            logger.warning(f"No structured insights parsed for {symbol}. Got keys: {list(result.keys()) if isinstance(result, dict) else 'n/a'}")
            return []

        insights: List[FinancialInsight] = []
        for item in raw_insights:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "neutral"))
            insight_text = str(item.get("insight", "")).strip()
            supporting = item.get("supporting_data", {})
            if not isinstance(supporting, (dict, str)):
                supporting = {}
            try:
                confidence = float(item.get("confidence", 0.7))
            except Exception:
                confidence = 0.7
            insights.append(FinancialInsight(
                category=category,
                insight=insight_text,
                supporting_data=supporting,
                confidence=confidence
            ))
        return insights
    
    def _generate_executive_summary(self, symbol: str, insights: List[FinancialInsight], 
                                   fundamentals: Dict) -> str:
        """AI generates executive summary based on insights."""
        
        if not insights:
            return f"{symbol} analysis completed. Review detailed financial metrics below."
        
        insight_text = "\n".join([f"- {i.insight}" for i in insights if i.insight])
        
        def fmt_pct(x):
            return f"{x:.1%}" if isinstance(x, (int, float)) else "N/A"
        def fmt_num(x):
            if isinstance(x, (int, float)):
                return f"${x:,.0f}"
            return "N/A"
        
        prompt = (
            f"Write a 3-4 sentence executive summary for {symbol} based on these key insights:\n\n"
            f"{insight_text}\n\n"
            f"Key metrics: Revenue {fmt_num(fundamentals.get('revenue'))}, "
            f"Net Margin {fmt_pct(fundamentals.get('net_margin'))}, "
            f"ROE {fmt_pct(fundamentals.get('roe'))}.\n"
            f"Style: Professional, concise, investment-focused. Lead with the investment conclusion."
        )
        
        response = self._make_openai_request([{"role": "user", "content": prompt}], max_tokens=220)
        if response:
            return response
        return f"{symbol} analysis completed. See detailed insights below."
    
    def _generate_investment_thesis(self, symbol: str, insights: List[FinancialInsight], 
                                   financial_data: Dict) -> Dict[str, str]:
        """AI generates bull and bear investment cases."""
        
        strengths = [i.insight for i in insights if i.category == "strength"]
        weaknesses = [i.insight for i in insights if i.category == "weakness"]
        opportunities = [i.insight for i in insights if i.category == "opportunity"]
        threats = [i.insight for i in insights if i.category == "threat"]
        
        dcf_value = financial_data.get("dcf_valuation", {}).get("enterprise_value", 0)
        if not isinstance(dcf_value, (int, float)):
            dcf_value = 0
        
        bull_prompt = (
            f"Create a bull case for {symbol} based on:\n"
            f"Strengths: {'; '.join(strengths)}\n"
            f"Opportunities: {'; '.join(opportunities)}\n"
            f"DCF Enterprise Value: ${dcf_value:,.0f}\n"
            f"Write 2-3 sentences focusing on upside catalysts and competitive advantages."
        )
        
        bear_prompt = (
            f"Create a bear case for {symbol} based on:\n"
            f"Weaknesses: {'; '.join(weaknesses)}\n"
            f"Threats: {'; '.join(threats)}\n"
            f"Write 2-3 sentences focusing on downside risks and challenges."
        )
        
        bull_response = self._make_openai_request([{"role": "user", "content": bull_prompt}], max_tokens=150)
        bear_response = self._make_openai_request([{"role": "user", "content": bear_prompt}], max_tokens=150)
        
        return {
            "bull_case": bull_response or "Upside potential based on financial strengths.",
            "bear_case": bear_response or "Downside risks from market challenges."
        }
    
    def _generate_rating_and_targets(self, symbol: str, dcf: Dict, 
                                    insights: List[FinancialInsight]) -> Tuple[str, Dict[str, float]]:
        """AI generates analyst rating and price targets."""
        
        strength_count = len([i for i in insights if i.category in ["strength", "opportunity"]])
        weakness_count = len([i for i in insights if i.category in ["weakness", "threat"]])
        net_positive = strength_count - weakness_count
        
        if net_positive >= 2:
            rating = "Buy"
        elif net_positive >= 0:
            rating = "Hold"
        else:
            rating = "Sell"
        
        base_value = dcf.get("enterprise_value", 100000000)
        if not isinstance(base_value, (int, float)):
            base_value = 100000000
        
        price_targets = {
            "low": float(base_value) * 0.85,
            "base": float(base_value),
            "high": float(base_value) * 1.15
        }
        
        return rating, price_targets
    
    def _prepare_financial_summary(self, fundamentals: Dict, dcf: Dict, comps: Dict) -> str:
        """Prepare financial data summary for AI analysis."""
        
        def fmt_num(x, is_pct=False):
            if not isinstance(x, (int, float)):
                return "N/A"
            if is_pct:
                return f"{x:.1%}"
            return f"${x:,.0f}" if x >= 1000 else f"{x:.2f}"
        
        peer_pe = comps.get("peer_pe_robust")
        if not isinstance(peer_pe, (int, float)):
            peer_pe = None

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
        - Enterprise Value: {fmt_num((dcf or {}).get('enterprise_value'))}
        - Peer Average P/E (robust): {peer_pe if peer_pe is not None else 'N/A'}
        """
        return summary
    
    def _get_peer_average_pe(self, comps: Dict) -> Optional[float]:
        """ kept for backward compatibility (not used anymore) """
        peers = comps.get("peers", []) or []
        pe_values = [p.get("pe") for p in peers if isinstance(p, dict) and isinstance(p.get("pe"), (int, float))]
        return (sum(pe_values) / len(pe_values)) if pe_values else None

# Global instance
ai_financial_analyst = AIFinancialAnalyst()
