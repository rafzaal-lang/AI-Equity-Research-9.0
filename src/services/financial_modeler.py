from __future__ import annotations
from typing import Dict, Any, Optional, List
import pandas as pd
from src.services.providers import fmp_provider as fmp
from src.services.wacc.peer_beta import peer_beta_wacc
from src.core.pti import calculate_cagr, get_latest_value
import logging

logger = logging.getLogger(__name__)

def _num(x):
    try: return float(x) if x is not None else None
    except: return None

def _to_df(rows: List[dict]) -> pd.DataFrame:
    """Convert list of dicts to DataFrame."""
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def ttm_snapshot(symbol: str) -> Dict[str, Any]:
    inc = (fmp.income_statement(symbol, period="annual", limit=1) or [{}])[0]
    bs = (fmp.balance_sheet(symbol, period="annual", limit=1) or [{}])[0]
    cf = (fmp.cash_flow(symbol, period="annual", limit=1) or [{}])[0]
    km = (fmp.key_metrics_ttm(symbol) or [{}])[0]
    
    revenue = _num(inc.get("revenue"))
    gross_profit = _num(inc.get("grossProfit"))
    operating_income = _num(inc.get("operatingIncome"))
    net_income = _num(inc.get("netIncome"))
    total_assets = _num(bs.get("totalAssets"))
    total_debt = _num(bs.get("totalDebt"))
    shareholders_equity = _num(bs.get("totalStockholdersEquity"))
    operating_cf = _num(cf.get("operatingCashFlow"))
    capex = _num(cf.get("capitalExpenditure"))
    
    return {
        "symbol": symbol.upper(),
        "reported": {
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_debt": total_debt,
            "shareholders_equity": shareholders_equity,
            "operating_cash_flow": operating_cf,
            "capex": capex,
            "free_cash_flow": (operating_cf + capex) if operating_cf and capex else None,
        },
        "margins": {
            "gross_margin": (gross_profit / revenue) if revenue and gross_profit else None,
            "operating_margin": (operating_income / revenue) if revenue and operating_income else None,
            "net_margin": (net_income / revenue) if revenue and net_income else None,
        },
        "ratios": {
            "debt_to_equity": (total_debt / shareholders_equity) if total_debt and shareholders_equity else None,
            "roa": (net_income / total_assets) if net_income and total_assets else None,
            "roe": (net_income / shareholders_equity) if net_income and shareholders_equity else None,
        }
    }

def simple_dcf(symbol: str, years: int = 5) -> Dict[str, Any]:
    """Simple DCF model with equity value calculation."""
    inc = fmp.income_statement(symbol, period="annual", limit=3)
    if not inc: return {"error": "No income statements"}
    
    # Simple growth rate from last 2 years
    if len(inc) >= 2:
        rev_growth = (inc[0]["revenue"] / inc[1]["revenue"] - 1) if inc[0].get("revenue") and inc[1].get("revenue") else 0.05
    else:
        rev_growth = 0.05
    
    # Cap growth rates
    rev_growth = max(min(rev_growth, 0.25), -0.10)
    
    base_fcf = _num(inc[0].get("freeCashFlow")) or 0
    if base_fcf <= 0:
        # Estimate FCF from operating income
        base_fcf = _num(inc[0].get("operatingIncome")) or 0
        base_fcf *= 0.7  # rough tax adjustment
    
    # Simple DCF projection
    discount_rate = 0.10
    terminal_growth = 0.025
    
    projections = []
    for year in range(1, years + 1):
        fcf = base_fcf * ((1 + rev_growth) ** year)
        pv = fcf / ((1 + discount_rate) ** year)
        projections.append({"year": year, "fcf": fcf, "pv": pv})
    
    pv_sum = sum(p["pv"] for p in projections)
    terminal_fcf = projections[-1]["fcf"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / ((1 + discount_rate) ** years)
    
    enterprise_value = pv_sum + terminal_pv
    
    # Calculate equity value and fair value per share
    equity_value = enterprise_value
    fair_value_per_share = None
    
    try:
        # Get balance sheet for net debt calculation
        bs = fmp.balance_sheet(symbol, period="annual", limit=1)
        if bs:
            total_debt = _num(bs[0].get("totalDebt")) or 0
            cash = _num(bs[0].get("cashAndCashEquivalents")) or 0
            net_debt = total_debt - cash
            equity_value = enterprise_value - net_debt
        
        # Get shares outstanding for per-share calculation
        key_metrics = fmp.key_metrics_ttm(symbol)
        if key_metrics:
            shares_outstanding = _num(key_metrics[0].get("sharesOutstanding")) or _num(key_metrics[0].get("weightedAverageShsOutDil"))
            if shares_outstanding and shares_outstanding > 0:
                fair_value_per_share = equity_value / shares_outstanding
    except Exception:
        pass  # Use enterprise value if equity calculation fails
    
    return {
        "symbol": symbol.upper(),
        "assumptions": {
            "revenue_growth": rev_growth,
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "projection_years": years,
        },
        "projections": projections,
        "terminal_value": terminal_value,
        "terminal_value_pv": terminal_pv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value_per_share,
        "terminal_value_pct": terminal_pv / enterprise_value if enterprise_value else None,
    }

def build_model(symbol: str, period: str = "annual", force_refresh: bool = False) -> Dict[str, Any]:
    try:
        snapshot = ttm_snapshot(symbol)
        dcf = simple_dcf(symbol)
        
        return {
            "symbol": symbol.upper(),
            "model_type": "basic_dcf",
            "core_financials": snapshot,
            "ttm_snapshot": snapshot["reported"],
            "dcf_valuation": dcf,
        }
    except Exception as e:
        return {"error": str(e)}



def enhanced_ttm_snapshot(symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Enhanced TTM snapshot with more comprehensive metrics."""
    km = (fmp.key_metrics_ttm(symbol) or [{}])[0]
    evs = (fmp.enterprise_values(symbol, period="quarter", limit=8) or [{}])
    ev0 = evs[0] if evs else {}
    
    return {
        "symbol": symbol.upper(),
        "ttm": {
            "revenue_per_share_ttm": _num(km.get("revenuePerShareTTM")),
            "net_income_per_share_ttm": _num(km.get("netIncomePerShareTTM")),
            "free_cash_flow_ttm": _num(km.get("freeCashFlowTTM")),
            "operating_cash_flow_ttm": _num(km.get("operatingCashFlowTTM")),
            "roe_ttm": _num(km.get("roeTTM")), 
            "roa_ttm": _num(km.get("roaTTM")),
            "gross_margin_ttm": _num(km.get("grossProfitMarginTTM")),
            "op_margin_ttm": _num(km.get("operatingProfitMarginTTM")),
            "net_margin_ttm": _num(km.get("netProfitMarginTTM")),
            "market_cap_latest": _num(ev0.get("marketCapitalization")),
            "enterprise_value_latest": _num(ev0.get("enterpriseValue")),
        },
    }

def core_financials(symbol: str, period: str = "annual", limit: int = 8) -> Dict[str, Any]:
    """Get core financial statements and calculate key metrics with historical trends."""
    inc = _to_df(fmp.income_statement(symbol, period=period, limit=limit))
    bs = _to_df(fmp.balance_sheet(symbol, period=period, limit=limit))
    cf = _to_df(fmp.cash_flow(symbol, period=period, limit=limit))
    
    if inc.empty or bs.empty:
        return {"error": "no data"}

    # Latest period data
    i0, b0 = inc.iloc[0], bs.iloc[0]
    c0 = cf.iloc[0] if not cf.empty else {}

    # Key financial metrics
    revenue = _num(i0.get("revenue"))
    net_income = _num(i0.get("netIncome"))
    gross = _num(i0.get("grossProfit"))
    op_income = _num(i0.get("operatingIncome"))
    fcf = _num((c0 or {}).get("freeCashFlow"))
    op_cf = _num((c0 or {}).get("operatingCashFlow"))
    total_equity = _num(b0.get("totalStockholdersEquity"))
    total_assets = _num(b0.get("totalAssets"))
    total_debt = _num(b0.get("totalDebt"))
    shares = _num(i0.get("weightedAverageShsOut"))

    # Calculate margins
    margins = {
        "gross_margin": (gross / revenue) if revenue and gross is not None else None,
        "operating_margin": (op_income / revenue) if revenue and op_income is not None else None,
        "net_margin": (net_income / revenue) if revenue and net_income is not None else None,
    }
    
    # Calculate ratios
    ratios = {
        "roe": (net_income / total_equity) if total_equity else None,
        "roa": (net_income / total_assets) if total_assets else None,
        "debt_to_equity": (total_debt / total_equity) if total_equity else None,
    }

    # Calculate growth rates using enhanced CAGR function
    rev_data = [{"date": row.get("date"), "value": _num(row.get("revenue"))} 
                for _, row in inc.iterrows()]
    ni_data = [{"date": row.get("date"), "value": _num(row.get("netIncome"))} 
               for _, row in inc.iterrows()]
    
    growth = {
        "revenue_cagr_3y": calculate_cagr(rev_data[:3], "value"),
        "net_income_cagr_3y": calculate_cagr(ni_data[:3], "value"),
    }

    return {
        "reported": {
            "revenue": revenue, 
            "net_income": net_income, 
            "free_cash_flow": fcf, 
            "operating_cash_flow": op_cf, 
            "shares_out": shares
        },
        "margins": margins, 
        "ratios": ratios, 
        "growth_rates": growth,
        "historical_data": {
            "revenue_series": [_num(row.get("revenue")) for _, row in inc.iterrows()],
            "net_income_series": [_num(row.get("netIncome")) for _, row in inc.iterrows()],
            "fcf_series": [_num(row.get("freeCashFlow")) for _, row in cf.iterrows()] if not cf.empty else []
        }
    }

def multi_stage_dcf(base_fcf: float, years_stage1: int = 5, g1: float = 0.06, 
                   years_stage2: int = 5, g2: float = 0.04, g_terminal: float = 0.025, 
                   wacc: float = 0.09) -> Dict[str, Any]:
    """Enhanced multi-stage DCF with sensitivity analysis."""
    
    # Stage 1: High growth
    stage1_flows = []
    fcf = base_fcf
    for t in range(1, years_stage1 + 1):
        fcf *= (1 + g1)
        pv = fcf / ((1 + wacc) ** t)
        stage1_flows.append({"year": t, "fcf": fcf, "pv": pv})
    
    # Stage 2: Moderate growth
    stage2_flows = []
    for t in range(years_stage1 + 1, years_stage1 + years_stage2 + 1):
        fcf *= (1 + g2)
        pv = fcf / ((1 + wacc) ** t)
        stage2_flows.append({"year": t, "fcf": fcf, "pv": pv})
    
    # Terminal value
    terminal_cf = fcf * (1 + g_terminal)
    terminal_val = terminal_cf / (wacc - g_terminal)
    terminal_pv = terminal_val / ((1 + wacc) ** (years_stage1 + years_stage2))
    
    # Total enterprise value
    stage1_pv = sum(f["pv"] for f in stage1_flows)
    stage2_pv = sum(f["pv"] for f in stage2_flows)
    ev = stage1_pv + stage2_pv + terminal_pv
    
    # Sensitivity analysis
    wacc_range = [wacc - 0.01, wacc, wacc + 0.01]
    tg_range = [g_terminal - 0.005, g_terminal, g_terminal + 0.005]
    
    sensitivity = {}
    for w in wacc_range:
        for tg in tg_range:
            if w <= tg:
                continue
            
            # Recompute terminal CF for the tested terminal growth (tg)
            tv_cf = fcf * (1 + tg)  # fcf is last Stage-2 FCF
            s1_pv = sum(f["fcf"] / ((1 + w) ** f["year"]) for f in stage1_flows)
            s2_pv = sum(f["fcf"] / ((1 + w) ** f["year"]) for f in stage2_flows)
            tv = tv_cf / (w - tg)
            tv_pv = tv / ((1 + w) ** (years_stage1 + years_stage2))
            
            sensitivity[f"wacc_{round(w*100,1)}%_tg_{round(tg*100,1)}%"] = float(s1_pv + s2_pv + tv_pv)
    
    return {
        "assumptions": {
            "wacc": wacc, 
            "g1": g1, 
            "g2": g2, 
            "g_terminal": g_terminal, 
            "years_stage1": years_stage1, 
            "years_stage2": years_stage2
        },
        "stage1_pv": float(stage1_pv),
        "stage2_pv": float(stage2_pv),
        "terminal_pv": float(terminal_pv),
        "enterprise_value": float(ev),
        "sensitivity_analysis": sensitivity,
        "terminal_value_pct": float(terminal_pv / ev) if ev else None,
        "cash_flows": {
            "stage1": stage1_flows,
            "stage2": stage2_flows,
            "terminal": {"fcf": terminal_cf, "value": terminal_val, "pv": terminal_pv}
        }
    }

def enhanced_build_model(symbol: str, period: str = "annual", force_refresh: bool = False, 
                        peers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build comprehensive financial model with peer-beta WACC."""
    
    # Get core financials
    core = core_financials(symbol, period=period, limit=8)
    if "error" in core:
        return {"symbol": symbol.upper(), "error": "no data"}
    
    # Get TTM snapshot
    snap = enhanced_ttm_snapshot(symbol, force_refresh=force_refresh)
    
    # Determine base FCF
    base_fcf = core["reported"].get("free_cash_flow") or snap["ttm"].get("free_cash_flow_ttm")
    if base_fcf is None:
        return {"symbol": symbol.upper(), "error": "no base FCF"}
    
    # Get peer list if not provided
    if peers is None:
        try:
            prof = fmp.profile(symbol) or {}
            sector, industry = prof.get("sector"), prof.get("industry")
            if sector and industry:
                peers = fmp.peers_by_screener(sector, industry, limit=20)[:10]
            else:
                peers = []
        except Exception:
            peers = []
    
    # Calculate peer-beta WACC
    wacc_analysis = peer_beta_wacc(symbol, peers or [])
    wacc = wacc_analysis.get("wacc", 0.09)
    
    # Use historical growth rate or default
    growth_rate = core["growth_rates"].get("revenue_cagr_3y") or 0.06
    growth_rate = max(min(growth_rate, 0.15), -0.05)  # Cap between -5% and 15%
    
    # Build DCF model
    dcf = multi_stage_dcf(
        base_fcf=base_fcf,
        years_stage1=5,
        g1=growth_rate,
        years_stage2=5,
        g2=0.04,
        g_terminal=0.025,
        wacc=wacc
    )
    
    return {
        "symbol": symbol.upper(),
        "model_type": "comprehensive_enhanced",
        "core_financials": core,
        "ttm_snapshot": snap,
        "wacc_analysis": wacc_analysis,
        "dcf_valuation": dcf,
        "peer_list": peers or []
    }



def calculate_comprehensive_ratios(income_data: List[Dict], balance_data: List[Dict], 
                                 cashflow_data: List[Dict], market_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Calculate comprehensive financial ratios across multiple categories."""
    
    if not income_data or not balance_data:
        return {"error": "Insufficient data for ratio calculation"}
    
    # Get latest period data
    latest_income = income_data[0]
    latest_balance = balance_data[0]
    latest_cashflow = cashflow_data[0] if cashflow_data else {}
    
    # Extract key values
    revenue = _num(latest_income.get("revenue"))
    net_income = _num(latest_income.get("netIncome"))
    gross_profit = _num(latest_income.get("grossProfit"))
    operating_income = _num(latest_income.get("operatingIncome"))
    ebit = _num(latest_income.get("ebitda")) or operating_income
    interest_expense = _num(latest_income.get("interestExpense"))
    
    total_assets = _num(latest_balance.get("totalAssets"))
    current_assets = _num(latest_balance.get("totalCurrentAssets"))
    cash = _num(latest_balance.get("cashAndCashEquivalents"))
    inventory = _num(latest_balance.get("inventory"))
    receivables = _num(latest_balance.get("netReceivables"))
    
    total_liabilities = _num(latest_balance.get("totalLiabilities"))
    current_liabilities = _num(latest_balance.get("totalCurrentLiabilities"))
    total_debt = _num(latest_balance.get("totalDebt"))
    long_term_debt = _num(latest_balance.get("longTermDebt"))
    shareholders_equity = _num(latest_balance.get("totalStockholdersEquity"))
    
    operating_cf = _num(latest_cashflow.get("operatingCashFlow"))
    free_cf = _num(latest_cashflow.get("freeCashFlow"))
    capex = _num(latest_cashflow.get("capitalExpenditure"))
    
    # Market data (if available)
    market_cap = _num(market_data.get("market_cap")) if market_data else None
    share_price = _num(market_data.get("share_price")) if market_data else None
    shares_outstanding = _num(latest_income.get("weightedAverageShsOut"))
    
    # Calculate ratios by category
    
    # 1. Profitability Ratios
    profitability = {
        "gross_margin": (gross_profit / revenue) if revenue and gross_profit else None,
        "operating_margin": (operating_income / revenue) if revenue and operating_income else None,
        "net_margin": (net_income / revenue) if revenue and net_income else None,
        "ebit_margin": (ebit / revenue) if revenue and ebit else None,
        "roa": (net_income / total_assets) if net_income and total_assets else None,
        "roe": (net_income / shareholders_equity) if net_income and shareholders_equity else None,
        "roic": None,  # Will calculate below
        "asset_turnover": (revenue / total_assets) if revenue and total_assets else None,
        "equity_multiplier": (total_assets / shareholders_equity) if total_assets and shareholders_equity else None
    }
    
    # Calculate ROIC (Return on Invested Capital)
    if net_income and total_assets and current_liabilities:
        invested_capital = total_assets - current_liabilities
        if invested_capital > 0:
            profitability["roic"] = net_income / invested_capital
    
    # 2. Liquidity Ratios
    liquidity = {
        "current_ratio": (current_assets / current_liabilities) if current_assets and current_liabilities else None,
        "quick_ratio": ((current_assets - inventory) / current_liabilities) if current_assets and inventory and current_liabilities else None,
        "cash_ratio": (cash / current_liabilities) if cash and current_liabilities else None,
        "working_capital": (current_assets - current_liabilities) if current_assets and current_liabilities else None,
        "working_capital_ratio": ((current_assets - current_liabilities) / revenue) if current_assets and current_liabilities and revenue else None
    }
    
    # 3. Leverage/Solvency Ratios
    leverage = {
        "debt_to_equity": (total_debt / shareholders_equity) if total_debt and shareholders_equity else None,
        "debt_to_assets": (total_debt / total_assets) if total_debt and total_assets else None,
        "equity_ratio": (shareholders_equity / total_assets) if shareholders_equity and total_assets else None,
        "debt_service_coverage": (operating_cf / total_debt) if operating_cf and total_debt else None,
        "interest_coverage": (ebit / interest_expense) if ebit and interest_expense and interest_expense != 0 else None,
        "long_term_debt_ratio": (long_term_debt / (long_term_debt + shareholders_equity)) if long_term_debt and shareholders_equity else None
    }
    
    # 4. Efficiency Ratios
    efficiency = {
        "inventory_turnover": (revenue / inventory) if revenue and inventory else None,
        "receivables_turnover": (revenue / receivables) if revenue and receivables else None,
        "days_sales_outstanding": (receivables * 365 / revenue) if receivables and revenue else None,
        "days_inventory_outstanding": (inventory * 365 / revenue) if inventory and revenue else None,
        "asset_turnover": (revenue / total_assets) if revenue and total_assets else None,
        "fixed_asset_turnover": None  # Would need fixed assets breakdown
    }
    
    # 5. Cash Flow Ratios
    cash_flow = {
        "operating_cf_ratio": (operating_cf / current_liabilities) if operating_cf and current_liabilities else None,
        "free_cf_yield": (free_cf / market_cap) if free_cf and market_cap else None,
        "capex_intensity": (abs(capex) / revenue) if capex and revenue else None,
        "fcf_conversion": (free_cf / net_income) if free_cf and net_income and net_income != 0 else None,
        "operating_cf_margin": (operating_cf / revenue) if operating_cf and revenue else None
    }
    
    # 6. Market Valuation Ratios (if market data available)
    valuation = {}
    if market_cap and shares_outstanding:
        eps = net_income / shares_outstanding if net_income and shares_outstanding else None
        book_value_per_share = shareholders_equity / shares_outstanding if shareholders_equity and shares_outstanding else None
        
        valuation = {
            "pe_ratio": (share_price / eps) if share_price and eps and eps != 0 else None,
            "pb_ratio": (share_price / book_value_per_share) if share_price and book_value_per_share and book_value_per_share != 0 else None,
            "ps_ratio": (market_cap / revenue) if market_cap and revenue else None,
            "pcf_ratio": (market_cap / operating_cf) if market_cap and operating_cf else None,
            "ev_revenue": None,  # Would need enterprise value
            "ev_ebitda": None,   # Would need enterprise value
            "dividend_yield": None  # Would need dividend data
        }
    
    return {
        "profitability": profitability,
        "liquidity": liquidity,
        "leverage": leverage,
        "efficiency": efficiency,
        "cash_flow": cash_flow,
        "valuation": valuation,
        "calculation_date": latest_income.get("date"),
        "data_quality": {
            "has_income": bool(income_data),
            "has_balance": bool(balance_data),
            "has_cashflow": bool(cashflow_data),
            "has_market_data": bool(market_data)
        }
    }

def scenario_analysis(base_model: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform scenario analysis on DCF model with different assumptions."""
    
    if "dcf_valuation" not in base_model:
        return {"error": "Base model must contain DCF valuation"}
    
    base_dcf = base_model["dcf_valuation"]
    base_assumptions = base_dcf["assumptions"]
    
    scenario_results = {}
    
    for i, scenario in enumerate(scenarios):
        scenario_name = scenario.get("name", f"Scenario_{i+1}")
        
        # Override base assumptions with scenario assumptions
        scenario_assumptions = {**base_assumptions, **scenario.get("assumptions", {})}
        
        # Recalculate DCF with new assumptions
        try:
            # Get base FCF from original model
            base_fcf = None
            if "core_financials" in base_model:
                base_fcf = base_model["core_financials"]["reported"].get("free_cash_flow")
            
            if base_fcf is None:
                continue
            
            scenario_dcf = multi_stage_dcf(
                base_fcf=base_fcf,
                years_stage1=scenario_assumptions.get("years_stage1", 5),
                g1=scenario_assumptions.get("g1", 0.06),
                years_stage2=scenario_assumptions.get("years_stage2", 5),
                g2=scenario_assumptions.get("g2", 0.04),
                g_terminal=scenario_assumptions.get("g_terminal", 0.025),
                wacc=scenario_assumptions.get("wacc", 0.09)
            )
            
            # Calculate variance from base case
            base_ev = base_dcf.get("enterprise_value", 0)
            scenario_ev = scenario_dcf.get("enterprise_value", 0)
            variance_pct = ((scenario_ev - base_ev) / base_ev * 100) if base_ev != 0 else 0
            
            scenario_results[scenario_name] = {
                "assumptions": scenario_assumptions,
                "enterprise_value": scenario_ev,
                "variance_from_base": variance_pct,
                "terminal_value_pct": scenario_dcf.get("terminal_value_pct"),
                "description": scenario.get("description", "")
            }
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario_name}: {e}")
            scenario_results[scenario_name] = {"error": str(e)}
    
    return {
        "base_case": {
            "enterprise_value": base_dcf.get("enterprise_value"),
            "assumptions": base_assumptions
        },
        "scenarios": scenario_results,
        "scenario_count": len(scenarios)
    }

def monte_carlo_simulation(base_model: Dict[str, Any], num_simulations: int = 1000, 
                          parameter_ranges: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """Perform Monte Carlo simulation on DCF model."""
    import random
    import statistics
    
    if "dcf_valuation" not in base_model:
        return {"error": "Base model must contain DCF valuation"}
    
    # Default parameter ranges if not provided
    if parameter_ranges is None:
        parameter_ranges = {
            "g1": {"min": 0.02, "max": 0.12, "distribution": "normal"},
            "g2": {"min": 0.02, "max": 0.06, "distribution": "normal"},
            "g_terminal": {"min": 0.015, "max": 0.035, "distribution": "normal"},
            "wacc": {"min": 0.06, "max": 0.12, "distribution": "normal"}
        }
    
    base_dcf = base_model["dcf_valuation"]
    base_assumptions = base_dcf["assumptions"]
    
    # Get base FCF
    base_fcf = None
    if "core_financials" in base_model:
        base_fcf = base_model["core_financials"]["reported"].get("free_cash_flow")
    
    if base_fcf is None:
        return {"error": "Cannot determine base FCF for simulation"}
    
    simulation_results = []
    
    for _ in range(num_simulations):
        # Generate random parameters
        sim_assumptions = base_assumptions.copy()
        
        for param, param_range in parameter_ranges.items():
            if param in sim_assumptions:
                min_val = param_range["min"]
                max_val = param_range["max"]
                
                if param_range.get("distribution") == "normal":
                    # Use normal distribution centered on current value
                    current_val = sim_assumptions[param]
                    std_dev = (max_val - min_val) / 6  # 99.7% within range
                    new_val = random.normalvariate(current_val, std_dev)
                    new_val = max(min_val, min(max_val, new_val))  # Clamp to range
                else:
                    # Uniform distribution
                    new_val = random.uniform(min_val, max_val)
                
                sim_assumptions[param] = new_val
        
        # Run DCF with simulated parameters
        try:
            sim_dcf = multi_stage_dcf(
                base_fcf=base_fcf,
                years_stage1=sim_assumptions.get("years_stage1", 5),
                g1=sim_assumptions["g1"],
                years_stage2=sim_assumptions.get("years_stage2", 5),
                g2=sim_assumptions["g2"],
                g_terminal=sim_assumptions["g_terminal"],
                wacc=sim_assumptions["wacc"]
            )
            
            simulation_results.append(sim_dcf["enterprise_value"])
            
        except Exception:
            continue  # Skip failed simulations
    
    if not simulation_results:
        return {"error": "All simulations failed"}
    
    # Calculate statistics
    mean_ev = statistics.mean(simulation_results)
    median_ev = statistics.median(simulation_results)
    std_dev = statistics.stdev(simulation_results) if len(simulation_results) > 1 else 0
    
    # Calculate percentiles
    sorted_results = sorted(simulation_results)
    percentiles = {
        "p5": sorted_results[int(0.05 * len(sorted_results))],
        "p10": sorted_results[int(0.10 * len(sorted_results))],
        "p25": sorted_results[int(0.25 * len(sorted_results))],
        "p75": sorted_results[int(0.75 * len(sorted_results))],
        "p90": sorted_results[int(0.90 * len(sorted_results))],
        "p95": sorted_results[int(0.95 * len(sorted_results))]
    }
    
    return {
        "simulation_count": len(simulation_results),
        "base_case_ev": base_dcf.get("enterprise_value"),
        "statistics": {
            "mean": mean_ev,
            "median": median_ev,
            "std_dev": std_dev,
            "min": min(simulation_results),
            "max": max(simulation_results)
        },
        "percentiles": percentiles,
        "parameter_ranges": parameter_ranges,
        "confidence_intervals": {
            "80%": [percentiles["p10"], percentiles["p90"]],
            "90%": [percentiles["p5"], percentiles["p95"]]
        }
    }

def comprehensive_financial_model(symbol: str, period: str = "annual", 
                                force_refresh: bool = False, 
                                peers: Optional[List[str]] = None,
                                include_scenarios: bool = True,
                                include_monte_carlo: bool = False) -> Dict[str, Any]:
    """Build the most comprehensive financial model with all enhancements."""
    
    try:
        # Build enhanced base model
        base_model = enhanced_build_model(symbol, period, force_refresh, peers)
        
        if "error" in base_model:
            return base_model
        
        # Add comprehensive ratio analysis
        income_data = fmp.income_statement(symbol, period=period, limit=8)
        balance_data = fmp.balance_sheet(symbol, period=period, limit=8)
        cashflow_data = fmp.cash_flow(symbol, period=period, limit=8)
        
        # Get market data for valuation ratios
        try:
            market_data = {}
            ev_data = fmp.enterprise_values(symbol, period="quarter", limit=1)
            if ev_data:
                market_data["market_cap"] = ev_data[0].get("marketCapitalization")
            
            price_data = fmp.latest_price(symbol)
            if price_data:
                market_data["share_price"] = price_data
        except Exception:
            market_data = {}
        
        comprehensive_ratios = calculate_comprehensive_ratios(
            income_data, balance_data, cashflow_data, market_data
        )
        
        # Add to model
        base_model["comprehensive_ratios"] = comprehensive_ratios
        
        # Add scenario analysis if requested
        if include_scenarios:
            default_scenarios = [
                {
                    "name": "Bull Case",
                    "description": "Optimistic growth assumptions",
                    "assumptions": {"g1": 0.10, "g2": 0.06, "wacc": 0.08}
                },
                {
                    "name": "Bear Case", 
                    "description": "Conservative growth assumptions",
                    "assumptions": {"g1": 0.02, "g2": 0.02, "wacc": 0.11}
                },
                {
                    "name": "High Interest Rate",
                    "description": "Higher cost of capital environment",
                    "assumptions": {"wacc": 0.12}
                }
            ]
            
            scenario_results = scenario_analysis(base_model, default_scenarios)
            base_model["scenario_analysis"] = scenario_results
        
        # Add Monte Carlo simulation if requested
        if include_monte_carlo:
            mc_results = monte_carlo_simulation(base_model, num_simulations=1000)
            base_model["monte_carlo_simulation"] = mc_results
        
        # Update model type
        base_model["model_type"] = "comprehensive_institutional"
        
        return base_model
        
    except Exception as e:
        logger.error(f"Error building comprehensive model for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}

