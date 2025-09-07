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
            "      \"insight\": \"Clear, specific insight in 1-2 s
