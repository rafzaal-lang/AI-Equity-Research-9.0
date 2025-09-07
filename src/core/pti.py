from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def filter_series_asof(rows: List[Dict], as_of: Optional[str], date_key: str = "date") -> List[Dict]:
    """Filter time series data to only include records as of a specific date."""
    if not as_of: 
        return rows
    return [r for r in rows if r.get(date_key) and r[date_key] <= as_of]

def last_date(rows: List[Dict], date_key: str = "date") -> Optional[str]:
    """Get the most recent date from a list of records."""
    ds = sorted([r.get(date_key) for r in rows if r.get(date_key)])
    return ds[-1] if ds else None

def get_latest_value(rows: List[Dict], value_key: str, date_key: str = "date") -> Optional[float]:
    """Get the latest value from a time series."""
    if not rows:
        return None
    sorted_rows = sorted([r for r in rows if r.get(date_key) and r.get(value_key) is not None], 
                        key=lambda x: x[date_key], reverse=True)
    return sorted_rows[0][value_key] if sorted_rows else None

def calculate_cagr(rows: List[Dict], value_key: str, date_key: str = "date", years: int = 3) -> Optional[float]:
    """Calculate compound annual growth rate from time series data."""
    if len(rows) < 2:
        return None
    
    sorted_rows = sorted([r for r in rows if r.get(date_key) and r.get(value_key) is not None], 
 # REPLACE lines 33-43 with:
def calculate_cagr(rows: List[Dict], value_key: str, date_key: str = "date", years: int = 3) -> Optional[float]:
    """Calculate compound annual growth rate from time series data."""
    if len(rows) < 2:
        return None
    
    sorted_rows = sorted([r for r in rows if r.get(date_key) and r.get(value_key) is not None], 
                        key=lambda x: x[date_key])
    
    if len(sorted_rows) < 2:
        return None
        
    first_val = sorted_rows[0][value_key]
    last_val = sorted_rows[-1][value_key]
    
    if first_val is None or last_val is None or first_val <= 0:
        return None
    
    # FIX: Calculate actual years between dates
    from datetime import datetime
    try:
        start_date = datetime.strptime(sorted_rows[0][date_key], "%Y-%m-%d")
        end_date = datetime.strptime(sorted_rows[-1][date_key], "%Y-%m-%d")
        actual_years = (end_date - start_date).days / 365.25
        
        if actual_years <= 0:
            return None
            
        return (last_val / first_val) ** (1 / actual_years) - 1
    except (ValueError, TypeError):
        return None       
    n_years = len(sorted_rows) - 1
    return (last_val / first_val) ** (1 / n_years) - 1

def validate_time_series(rows: List[Dict], date_key: str = "date", 
                        value_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate time series data quality and completeness."""
    if not rows:
        return {"valid": False, "errors": ["Empty dataset"]}
    
    errors = []
    warnings = []
    
    # Check date consistency
    dates = [r.get(date_key) for r in rows if r.get(date_key)]
    if len(dates) != len(rows):
        errors.append(f"Missing dates in {len(rows) - len(dates)} records")
    
    # Check for duplicate dates
    unique_dates = set(dates)
    if len(unique_dates) != len(dates):
        warnings.append(f"Found {len(dates) - len(unique_dates)} duplicate dates")
    
    # Check chronological order
    sorted_dates = sorted(dates)
    if dates != sorted_dates:
        warnings.append("Data not in chronological order")
    
    # Check value completeness if specified
    if value_keys:
        for key in value_keys:
            values = [r.get(key) for r in rows if r.get(key) is not None]
            missing_pct = (len(rows) - len(values)) / len(rows) * 100
            if missing_pct > 50:
                errors.append(f"More than 50% missing values for {key}")
            elif missing_pct > 10:
                warnings.append(f"{missing_pct:.1f}% missing values for {key}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "record_count": len(rows),
        "date_range": {"start": min(dates) if dates else None, "end": max(dates) if dates else None},
        "unique_dates": len(unique_dates)
    }

def get_period_data(rows: List[Dict], as_of: str, lookback_periods: int = 4, 
                   date_key: str = "date") -> List[Dict]:
    """Get the most recent N periods of data as of a specific date."""
    filtered = filter_series_asof(rows, as_of, date_key)
    if not filtered:
        return []
    
    # Sort by date descending and take the most recent periods
    sorted_data = sorted(filtered, key=lambda x: x[date_key], reverse=True)
    return sorted_data[:lookback_periods]

def calculate_growth_metrics(rows: List[Dict], value_key: str, date_key: str = "date") -> Dict[str, Optional[float]]:
    """Calculate comprehensive growth metrics from time series data."""
    if len(rows) < 2:
        return {"yoy_growth": None, "cagr_3y": None, "cagr_5y": None, "avg_growth": None}
    
    sorted_rows = sorted([r for r in rows if r.get(date_key) and r.get(value_key) is not None], 
                        key=lambda x: x[date_key])
    
    if len(sorted_rows) < 2:
        return {"yoy_growth": None, "cagr_3y": None, "cagr_5y": None, "avg_growth": None}
    
    # Year-over-year growth (most recent vs previous)
    yoy_growth = None
    if len(sorted_rows) >= 2:
        current = sorted_rows[-1][value_key]
        previous = sorted_rows[-2][value_key]
        if previous and previous != 0:
            yoy_growth = (current / previous) - 1
    
    # CAGR calculations
    cagr_3y = calculate_cagr(sorted_rows[-4:] if len(sorted_rows) >= 4 else sorted_rows, value_key, date_key)
    cagr_5y = calculate_cagr(sorted_rows[-6:] if len(sorted_rows) >= 6 else sorted_rows, value_key, date_key)
    
    # Average growth rate
    avg_growth = None
    if len(sorted_rows) >= 3:
        growth_rates = []
        for i in range(1, len(sorted_rows)):
            prev_val = sorted_rows[i-1][value_key]
            curr_val = sorted_rows[i][value_key]
            if prev_val and prev_val != 0:
                growth_rates.append((curr_val / prev_val) - 1)
        
        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
    
    return {
        "yoy_growth": yoy_growth,
        "cagr_3y": cagr_3y,
        "cagr_5y": cagr_5y,
        "avg_growth": avg_growth
    }

def create_pti_snapshot(symbol: str, as_of: str, data_sources: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Create a comprehensive point-in-time snapshot of a company's financial data."""
    snapshot = {
        "symbol": symbol.upper(),
        "as_of": as_of,
        "data_quality": {},
        "financials": {},
        "metrics": {}
    }
    
    # Validate each data source
    for source_name, data in data_sources.items():
        filtered_data = filter_series_asof(data, as_of)
        validation = validate_time_series(filtered_data)
        snapshot["data_quality"][source_name] = validation
        
        if validation["valid"] and filtered_data:
            # Get the most recent data point
            latest = sorted(filtered_data, key=lambda x: x.get("date", ""), reverse=True)[0]
            snapshot["financials"][source_name] = latest
    
    return snapshot


