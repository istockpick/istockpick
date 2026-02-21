# Stock Analysis package

from .fundamental import FundamentalAnalyzer
from .technical import TechnicalAnalyzer
from .web_analyzer import generate_full_analysis, generate_media_analysis, AssetType
from .congress import fetch_trades, yearly_report, seasonal_summary
from .crypto import get_crypto_snapshot, get_crypto_sentiment
from .futures import get_futures_snapshot, get_futures_sentiment
from .options import get_options_chain, get_options_snapshot, get_options_sentiment

__version__ = "1.1.0"
__all__ = [
    "TechnicalAnalyzer",
    "FundamentalAnalyzer",
    "generate_full_analysis",
    "generate_media_analysis",
    "AssetType",
    "fetch_trades",
    "yearly_report",
    "seasonal_summary",
    "get_crypto_snapshot",
    "get_crypto_sentiment",
    "get_futures_snapshot",
    "get_futures_sentiment",
    "get_options_chain",
    "get_options_snapshot",
    "get_options_sentiment",
]
