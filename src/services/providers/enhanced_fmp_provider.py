import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import requests
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    before_sleep_log, after_log
)
from src.services.cache.redis_client import get_json, set_json

logger = logging.getLogger(__name__)

# Configuration
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/api")
FMP_RATE_LIMIT = int(os.getenv("FMP_RATE_LIMIT", "300"))  # requests per minute
FMP_TIMEOUT = int(os.getenv("FMP_TIMEOUT", "30"))

# Global session with connection pooling
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'EquityResearch/1.0',
    'Accept': 'application/json',
    'Connection': 'keep-alive'
})

class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class DataValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality: DataQuality
    errors: List[str]
    warnings: List[str]
    completeness_score: float
    freshness_score: float
    metadata: Dict[str, Any]

@dataclass
class APICallMetrics:
    """Metrics for API calls."""
    endpoint: str
    symbol: str
    start_time: float
    end_time: float
    success: bool
    cached: bool
    error_message: Optional[str] = None
    response_size: Optional[int] = None
    rate_limited: bool = False

class FMPError(Exception):
    """Custom FMP API error."""
    pass

class RateLimitError(FMPError):
    """Rate limit exceeded error."""
    pass

class DataValidationError(FMPError):
    """Data validation error."""
    pass

class EnhancedFMPProvider:
    """Enhanced FMP provider with comprehensive error handling and validation."""
    
    def __init__(self):
        self.api_key = FMP_API_KEY
        self.base_url = FMP_BASE_URL
        self.session = SESSION
        self.call_metrics: List[APICallMetrics] = []
        self.rate_limiter = self._init_rate_limiter()
        
        if not self.api_key:
            raise FMPError("FMP_API_KEY environment variable is required")
    
    def _init_rate_limiter(self) -> Dict[str, Any]:
        """Initialize rate limiter."""
        return {
            "calls": [],
            "max_calls_per_minute": FMP_RATE_LIMIT,
            "window_size": 60  # seconds
        }
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        now = time.time()
        window_start = now - self.rate_limiter["window_size"]
        
        # Remove old calls outside the window
        self.rate_limiter["calls"] = [
            call_time for call_time in self.rate_limiter["calls"] 
            if call_time > window_start
        ]
        
        # Check if we're at the limit
        if len(self.rate_limiter["calls"]) >= self.rate_limiter["max_calls_per_minute"]:
            sleep_time = self.rate_limiter["calls"][0] - window_start + 1
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            raise RateLimitError("Rate limit exceeded")
        
        # Record this call
        self.rate_limiter["calls"].append(now)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=0.5, max=10),
        retry=retry_if_exception_type((requests.RequestException, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )
    def _make_request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, APICallMetrics]:
        """Make API request with comprehensive error handling."""
        if params is None:
            params = {}
        
        # Add API key
        params = {**params, "apikey": self.api_key}
        url = f"{self.base_url}/v3/{path.lstrip('/')}"
        
        # Extract symbol for metrics (if present in path or params)
        symbol = params.get("symbol", path.split("/")[-1] if "/" in path else "unknown")
        
        start_time = time.time()
        metrics = APICallMetrics(
            endpoint=path,
            symbol=symbol,
            start_time=start_time,
            end_time=0,
            success=False,
            cached=False
        )
        
        try:
            # Check rate limits
            self._check_rate_limit()
            
            # Make request
            response = self.session.get(url, params=params, timeout=FMP_TIMEOUT)
            end_time = time.time()
            
            # Update metrics
            metrics.end_time = end_time
            metrics.response_size = len(response.content) if response.content else 0
            
            # Handle HTTP errors
            if response.status_code == 429:
                metrics.rate_limited = True
                raise RateLimitError("API rate limit exceeded")
            
            response.raise_for_status()
            
            # Parse JSON
            try:
                data = response.json()
            except ValueError as e:
                raise FMPError(f"Invalid JSON response: {e}")
            
            # Check for API-level errors
            if isinstance(data, dict) and "Error Message" in data:
                raise FMPError(f"API Error: {data['Error Message']}")
            
            metrics.success = True
            self.call_metrics.append(metrics)
            
            logger.debug(f"API call successful: {path} ({end_time - start_time:.2f}s)")
            return data, metrics
            
        except requests.exceptions.Timeout:
            metrics.end_time = time.time()
            metrics.error_message = "Request timeout"
            self.call_metrics.append(metrics)
            raise FMPError("Request timeout")
            
        except requests.exceptions.ConnectionError:
            metrics.end_time = time.time()
            metrics.error_message = "Connection error"
            self.call_metrics.append(metrics)
            raise FMPError("Connection error")
            
        except requests.exceptions.HTTPError as e:
            metrics.end_time = time.time()
            metrics.error_message = str(e)
            self.call_metrics.append(metrics)
            
            if response.status_code == 401:
                raise FMPError("Invalid API key")
            elif response.status_code == 403:
                raise FMPError("API access forbidden")
            elif response.status_code == 404:
                raise FMPError(f"Endpoint not found: {path}")
            else:
                raise FMPError(f"HTTP {response.status_code}: {e}")
        
        except Exception as e:
            metrics.end_time = time.time()
            metrics.error_message = str(e)
            self.call_metrics.append(metrics)
            raise
    
    def _get_cached_or_fetch(self, cache_key: str, path: str, params: Dict[str, Any], 
                           ttl: int, force_refresh: bool = False) -> Tuple[Any, bool]:
        """Get data from cache or fetch from API."""
        # Try cache first
        if not force_refresh:
            cached_data = get_json(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_data, True
        
        # Fetch from API
        data, metrics = self._make_request(path, params)
        
        # Cache the result
        set_json(cache_key, data, ttl=ttl)
        logger.debug(f"Cached data: {cache_key} (TTL: {ttl}s)")
        
        return data, False
    
    def validate_financial_data(self, data: List[Dict], data_type: str, 
                              symbol: str) -> DataValidationResult:
        """Validate financial statement data."""
        errors = []
        warnings = []
        
        if not data:
            return DataValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                errors=["No data returned"],
                warnings=[],
                completeness_score=0.0,
                freshness_score=0.0,
                metadata={"symbol": symbol, "data_type": data_type}
            )
        
        # Check data structure
        if not isinstance(data, list):
            errors.append("Data is not a list")
            return DataValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                errors=errors,
                warnings=warnings,
                completeness_score=0.0,
                freshness_score=0.0,
                metadata={"symbol": symbol, "data_type": data_type}
            )
        
        # Define required fields by data type
        required_fields = {
            "income_statement": ["date", "revenue", "netIncome"],
            "balance_sheet": ["date", "totalAssets", "totalStockholdersEquity"],
            "cash_flow": ["date", "operatingCashFlow"],
            "key_metrics": ["date"],
            "enterprise_values": ["date", "marketCapitalization"],
            "historical_prices": ["date", "close"]
        }
        
        required = required_fields.get(data_type, ["date"])
        
        # Validate each record
        valid_records = 0
        total_fields = 0
        present_fields = 0
        dates = []
        
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                errors.append(f"Record {i} is not a dictionary")
                continue
            
            # Check required fields
            record_valid = True
            for field in required:
                total_fields += 1
                if field in record and record[field] is not None:
                    present_fields += 1
                else:
                    if field == "date":
                        errors.append(f"Record {i} missing required field: {field}")
                        record_valid = False
                    else:
                        warnings.append(f"Record {i} missing field: {field}")
            
            # Validate date format
            if "date" in record and record["date"]:
                try:
                    date_obj = datetime.strptime(record["date"], "%Y-%m-%d")
                    dates.append(date_obj)
                except ValueError:
                    errors.append(f"Record {i} has invalid date format: {record['date']}")
                    record_valid = False
            
            if record_valid:
                valid_records += 1
        
        # Calculate scores
        completeness_score = present_fields / total_fields if total_fields > 0 else 0
        
        # Calculate freshness score
        freshness_score = 0.0
        if dates:
            latest_date = max(dates)
            days_old = (datetime.now() - latest_date).days
            # Score decreases as data gets older (100% for current quarter, 0% for >2 years old)
            freshness_score = max(0, min(1, (730 - days_old) / 730))
        
        # Determine quality level
        if len(errors) > 0:
            quality = DataQuality.INVALID
        elif completeness_score >= 0.9 and freshness_score >= 0.8:
            quality = DataQuality.HIGH
        elif completeness_score >= 0.7 and freshness_score >= 0.5:
            quality = DataQuality.MEDIUM
        else:
            quality = DataQuality.LOW
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            quality=quality,
            errors=errors,
            warnings=warnings,
            completeness_score=completeness_score,
            freshness_score=freshness_score,
            metadata={
                "symbol": symbol,
                "data_type": data_type,
                "record_count": len(data),
                "valid_records": valid_records,
                "date_range": {
                    "start": min(dates).isoformat() if dates else None,
                    "end": max(dates).isoformat() if dates else None
                }
            }
        )
    
    def income_statement(self, symbol: str, period: str = "annual", limit: int = 8, 
                        force_refresh: bool = False, validate: bool = True) -> Dict[str, Any]:
        """Get income statement with validation."""
        cache_key = f"fmp:income_statement:{symbol}:{period}:{limit}"
        path = f"income-statement/{symbol}"
        params = {"period": period, "limit": limit}
        
        try:
            data, cached = self._get_cached_or_fetch(cache_key, path, params, ttl=21600, force_refresh=force_refresh)
            
            result = {
                "symbol": symbol.upper(),
                "data": data,
                "cached": cached,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if validate:
                validation = self.validate_financial_data(data, "income_statement", symbol)
                result["validation"] = validation
                
                if not validation.is_valid:
                    logger.warning(f"Income statement validation failed for {symbol}: {validation.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def balance_sheet(self, symbol: str, period: str = "annual", limit: int = 8,
                     force_refresh: bool = False, validate: bool = True) -> Dict[str, Any]:
        """Get balance sheet with validation."""
        cache_key = f"fmp:balance_sheet:{symbol}:{period}:{limit}"
        path = f"balance-sheet-statement/{symbol}"
        params = {"period": period, "limit": limit}
        
        try:
            data, cached = self._get_cached_or_fetch(cache_key, path, params, ttl=21600, force_refresh=force_refresh)
            
            result = {
                "symbol": symbol.upper(),
                "data": data,
                "cached": cached,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if validate:
                validation = self.validate_financial_data(data, "balance_sheet", symbol)
                result["validation"] = validation
                
                if not validation.is_valid:
                    logger.warning(f"Balance sheet validation failed for {symbol}: {validation.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def cash_flow(self, symbol: str, period: str = "annual", limit: int = 8,
                 force_refresh: bool = False, validate: bool = True) -> Dict[str, Any]:
        """Get cash flow statement with validation."""
        cache_key = f"fmp:cash_flow:{symbol}:{period}:{limit}"
        path = f"cash-flow-statement/{symbol}"
        params = {"period": period, "limit": limit}
        
        try:
            data, cached = self._get_cached_or_fetch(cache_key, path, params, ttl=21600, force_refresh=force_refresh)
            
            result = {
                "symbol": symbol.upper(),
                "data": data,
                "cached": cached,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if validate:
                validation = self.validate_financial_data(data, "cash_flow", symbol)
                result["validation"] = validation
                
                if not validation.is_valid:
                    logger.warning(f"Cash flow validation failed for {symbol}: {validation.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching cash flow for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get provider health metrics."""
        if not self.call_metrics:
            return {"status": "no_data", "metrics": {}}
        
        recent_calls = [m for m in self.call_metrics if time.time() - m.end_time < 3600]  # Last hour
        
        if not recent_calls:
            return {"status": "no_recent_data", "metrics": {}}
        
        success_rate = sum(1 for m in recent_calls if m.success) / len(recent_calls)
        avg_response_time = sum(m.end_time - m.start_time for m in recent_calls) / len(recent_calls)
        cache_hit_rate = sum(1 for m in recent_calls if m.cached) / len(recent_calls)
        
        return {
            "status": "healthy" if success_rate > 0.95 else "degraded" if success_rate > 0.8 else "unhealthy",
            "metrics": {
                "total_calls_last_hour": len(recent_calls),
                "success_rate": round(success_rate, 3),
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "cache_hit_rate": round(cache_hit_rate, 3),
                "rate_limit_hits": sum(1 for m in recent_calls if m.rate_limited),
                "error_rate": round(1 - success_rate, 3)
            }
        }

# Global instance
enhanced_fmp_provider = EnhancedFMPProvider()

