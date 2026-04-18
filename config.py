"""
config.py — v5 (Production)
Central configuration loaded from .env. Fails fast with clear errors.
All scalping parameters tuned for structure‑first v5 engine.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _get(key, default=None, cast=str, required=False):
    val = os.environ.get(key, default)
    if required and val is None:
        raise EnvironmentError(f"Required env var '{key}' not set. Copy .env.example → .env")
    if val is None:
        return None
    try:
        return cast(val)
    except Exception:
        raise EnvironmentError(f"Invalid value for '{key}': '{val}' cannot be cast to {cast.__name__}")


def _get_bool(key, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() not in ("false", "0", "no", "off", "")


@dataclass(frozen=True)
class WhatsAppConfig:
    number:              str = field(default_factory=lambda: _get("WHATSAPP_NUMBER", required=True))
    api_key:             str = field(default_factory=lambda: _get("WHAPI_TOKEN", required=True))
    api_url:             str = "https://gate.whapi.cloud/messages/text"
    request_timeout:     int = 15
    retry_attempts:      int = 3
    retry_delay_seconds: int = 5


@dataclass(frozen=True)
class ScanConfig:
    top_gainers_count:     int   = field(default_factory=lambda: _get("TOP_GAINERS_COUNT", 40, int))
    min_gain_percent:      float = field(default_factory=lambda: _get("MIN_GAIN_PERCENT", 2.0, float))
    scan_interval_minutes: int   = field(default_factory=lambda: _get("SCAN_INTERVAL_MINUTES", 1, int))
    timeframe:             str   = field(default_factory=lambda: _get("TIMEFRAME", "5m"))
    candle_limit:          int   = field(default_factory=lambda: _get("CANDLE_LIMIT", 80, int))
    min_quote_volume:      float = 5_000_000.0
    stable_coins:          tuple = ("USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "UST", "USDD")


@dataclass(frozen=True)
class SignalConfig:
    rsi_overbought:        float = field(default_factory=lambda: _get("RSI_OVERBOUGHT", 70, float))
    rsi_buy_min:           float = field(default_factory=lambda: _get("RSI_BUY_MIN", 42, float))
    rsi_buy_max:           float = field(default_factory=lambda: _get("RSI_BUY_MAX", 65, float))
    btc_strong_threshold:  int   = field(default_factory=lambda: _get("BTC_STRONG_THRESHOLD", 60, int))
    btc_weak_threshold:    int   = field(default_factory=lambda: _get("BTC_WEAK_THRESHOLD", 45, int))
    btc_rsi_danger:        float = field(default_factory=lambda: _get("BTC_RSI_DANGER", 72, float))
    ema_distance_strong:   float = field(default_factory=lambda: _get("EMA_DISTANCE_STRONG", 2.0, float))
    ema_distance_moderate: float = field(default_factory=lambda: _get("EMA_DISTANCE_MODERATE", 1.5, float))
    volume_climax:         float = field(default_factory=lambda: _get("VOLUME_CLIMAX", 3.0, float))
    volume_strong:         float = field(default_factory=lambda: _get("VOLUME_STRONG", 2.0, float))
    volume_buy_min:        float = field(default_factory=lambda: _get("VOLUME_BUY_MIN", 1.3, float))
    min_sell_confluence:   float = field(default_factory=lambda: _get("MIN_SELL_CONFLUENCE", 4.0, float))
    min_buy_confluence:    float = field(default_factory=lambda: _get("MIN_BUY_CONFLUENCE", 4.0, float))

    # v5 new fields – adaptive threshold & ranking
    adaptive_threshold_start: int = field(default_factory=lambda: _get("ADAPTIVE_THRESHOLD_START", 48, int))
    adaptive_threshold_min:   int = field(default_factory=lambda: _get("ADAPTIVE_THRESHOLD_MIN", 42, int))
    max_candidates_per_cycle: int = field(default_factory=lambda: _get("MAX_CANDIDATES_PER_CYCLE", 3, int))


@dataclass(frozen=True)
class RiskConfig:
    tp1_pct:       float = field(default_factory=lambda: _get("TP1_PCT", 0.8, float))
    tp2_pct:       float = field(default_factory=lambda: _get("TP2_PCT", 1.2, float))
    tp3_pct:       float = field(default_factory=lambda: _get("TP3_PCT", 2.0, float))
    sl_pct:        float = field(default_factory=lambda: _get("SL_PCT", 0.8, float))
    tp1_close_pct: int   = field(default_factory=lambda: _get("TP1_CLOSE", 40, int))
    tp2_close_pct: int   = field(default_factory=lambda: _get("TP2_CLOSE", 35, int))
    tp3_close_pct: int   = field(default_factory=lambda: _get("TP3_CLOSE", 25, int))

    def __post_init__(self):
        total = self.tp1_close_pct + self.tp2_close_pct + self.tp3_close_pct
        if total != 100:
            raise ValueError(f"TP1_CLOSE+TP2_CLOSE+TP3_CLOSE must equal 100, got {total}")


@dataclass(frozen=True)
class AlertConfig:
    cooldown_hours:              float = field(default_factory=lambda: _get("COOLDOWN_HOURS", 0.05, float))
    btc_update_every_hours:      int   = field(default_factory=lambda: _get("BTC_UPDATE_EVERY_HOURS", 4, int))
    whatsapp_rate_limit_seconds: int   = 3
    binance_rate_limit_seconds:  float = 0.8


@dataclass(frozen=True)
class LogConfig:
    level:        str  = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())
    log_dir:      Path = Path(__file__).parent / "logs"
    log_filename: str  = "btc_bot.log"
    max_bytes:    int  = 5 * 1024 * 1024
    backup_count: int  = 5


@dataclass(frozen=True)
class BinanceConfig:
    base_url:        str = "https://api.binance.com/api/v3"
    api_key:         str = field(default_factory=lambda: _get("BINANCE_API_KEY", ""))
    api_secret:      str = field(default_factory=lambda: _get("BINANCE_API_SECRET", ""))
    request_timeout: int = 12


@dataclass(frozen=True)
class AutoTradeConfig:
    api_key:    str  = field(default_factory=lambda: _get("BINANCE_API_KEY", ""))
    api_secret: str  = field(default_factory=lambda: _get("BINANCE_API_SECRET", ""))
    testnet:    bool = field(default_factory=lambda: _get_bool("BINANCE_TESTNET", default=True))
    risk_pct_per_trade:   float = field(default_factory=lambda: _get("AUTO_RISK_PCT", 1.0, float))
    daily_loss_limit_pct: float = field(default_factory=lambda: _get("AUTO_LOSS_LIMIT", 3.0, float))
    max_hold_minutes:     int   = field(default_factory=lambda: _get("MAX_HOLD_MINUTES", 15, int))
    max_concurrent_pos:   int   = field(default_factory=lambda: _get("MAX_CONCURRENT_POS", 5, int))

    @property
    def has_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass(frozen=True)
class AppConfig:
    whatsapp: WhatsAppConfig  = field(default_factory=WhatsAppConfig)
    scan:     ScanConfig      = field(default_factory=ScanConfig)
    signal:   SignalConfig    = field(default_factory=SignalConfig)
    risk:     RiskConfig      = field(default_factory=RiskConfig)
    alert:    AlertConfig     = field(default_factory=AlertConfig)
    auto:     AutoTradeConfig = field(default_factory=AutoTradeConfig)
    log:      LogConfig       = field(default_factory=LogConfig)
    binance:  BinanceConfig   = field(default_factory=BinanceConfig)
    version:  str             = "5.0.0"


def load_config() -> AppConfig:
    return AppConfig()