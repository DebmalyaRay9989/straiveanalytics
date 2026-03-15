"""
STRAIVE Pricing Platform – Data Gathering Module (v5.0 Cloud-Light)

Changes vs v4.1:
  • Removed all real HTTP scraping (BaseScraper, CompetitorPriceScraper,
    MarketIntelligenceGatherer) — these require Selenium, Redis, and open
    network access which are incompatible with most PaaS cloud environments.
  • DataGatheringOrchestrator now returns empty lists (no-op) with a clear
    log message rather than attempting network calls.
  • EnhancedDataGenerator falls back to pure synthetic data.
  • ScheduledDataGatherer is a no-op stub so existing imports don't break.
  • SignalDetector and dataclasses kept intact for type compatibility.
  • All public API symbols preserved so app.py imports don't need changes.

  To re-enable real scraping: replace this file with the v4.1 version and
  install the heavy optional deps (selenium, redis, lxml, beautifulsoup4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    COMPETITORS, CURRENT_DATE, CUSTOMER_SEGMENTS,
    MONTHLY_SEASONALITY, PRODUCT_CATALOG, REGIONS,
)

log = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES  (unchanged from v4.1 for type compatibility)
# ============================================================================

@dataclass
class CompetitorPricePoint:
    competitor_name: str
    service_name: str
    price: float
    currency: str = "USD"
    source_url: str = ""
    confidence_score: float = 0.5
    scraped_at: Optional[datetime] = None
    raw_text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scraped_at is None:
            self.scraped_at = datetime.now()
        self.confidence_score = float(max(0.0, min(1.0, self.confidence_score)))


@dataclass
class MarketIntelligence:
    source: str
    data_type: str
    content: str
    timestamp: Optional[datetime] = None
    relevance_score: float = 0.5
    competitor: str = ""
    url: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()
        self.relevance_score = float(max(0.0, min(1.0, self.relevance_score)))


@dataclass
class PricingSignal:
    signal_type: str
    competitor: str
    magnitude: float
    confidence: float
    detected_at: datetime
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SIGNAL DETECTOR  (lightweight — works only on in-memory history)
# ============================================================================

class SignalDetector:
    _MIN_MAGNITUDE_PCT: float = 3.0

    def __init__(self) -> None:
        self._history: List[CompetitorPricePoint] = []

    def add_price_points(self, points: List[CompetitorPricePoint]) -> None:
        self._history.extend(points)

    def get_recent_signals(self, hours: int = 168) -> List[PricingSignal]:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [p for p in self._history if p.scraped_at is not None and p.scraped_at >= cutoff]
        signals: List[PricingSignal] = []
        by_comp: Dict[str, List[CompetitorPricePoint]] = {}
        for pt in recent:
            by_comp.setdefault(pt.competitor_name, []).append(pt)
        for comp, pts in by_comp.items():
            if len(pts) < 2:
                continue
            pts_sorted = sorted(pts, key=lambda p: p.scraped_at)  # type: ignore[arg-type]
            first_price = pts_sorted[0].price
            last_price  = pts_sorted[-1].price
            if first_price <= 0:
                continue
            magnitude = (last_price - first_price) / first_price * 100
            if abs(magnitude) < self._MIN_MAGNITUDE_PCT:
                continue
            signal_type = "price_increase" if magnitude > 0 else "price_decrease"
            confidence  = min(0.95, pts_sorted[-1].confidence_score * 1.1)
            signals.append(PricingSignal(
                signal_type=signal_type, competitor=comp,
                magnitude=round(magnitude, 2), confidence=round(confidence, 3),
                detected_at=pts_sorted[-1].scraped_at,  # type: ignore[arg-type]
                details={"first_price": round(first_price, 2), "last_price": round(last_price, 2)},
            ))
        return signals


# ============================================================================
# DATA GATHERING ORCHESTRATOR  (no-op stubs for cloud-light mode)
# ============================================================================

class DataGatheringOrchestrator:
    """
    Cloud-light stub.  All network methods return empty lists immediately.
    The signal_detector attribute is kept so app.py callbacks don't break.
    """

    def __init__(self, use_redis: bool = False, redis_host: str = "localhost") -> None:
        self.signal_detector = SignalDetector()
        if use_redis:
            log.info("DataGatheringOrchestrator: Redis disabled in cloud-light mode")

    def gather_all_competitor_prices(self, use_cache: bool = True) -> List[CompetitorPricePoint]:
        log.debug("DataGatheringOrchestrator: scraping disabled in cloud-light mode")
        return []

    def gather_market_intelligence(self) -> List[MarketIntelligence]:
        log.debug("DataGatheringOrchestrator: market intelligence disabled in cloud-light mode")
        return []

    def generate_market_report(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(),
            "competitor_prices": {"total_points": 0, "avg_price_by_competitor": {}},
            "signals": [],
            "recommendations": [
                "Scraping is disabled in cloud-light mode.",
                "Continue monitoring competitor pricing manually.",
                "Review pricing strategy for high-elasticity segments quarterly.",
            ],
        }


# ============================================================================
# ENHANCED DATA GENERATOR  (pure synthetic fallback)
# ============================================================================

class EnhancedDataGenerator:
    """
    In cloud-light mode this always generates pure synthetic data.
    The real_data_ratio parameter is accepted but ignored.
    """

    def __init__(
        self,
        orchestrator: DataGatheringOrchestrator,
        real_data_ratio: float = 0.30,
    ) -> None:
        self.orchestrator    = orchestrator
        self.real_data_ratio = float(max(0.0, min(1.0, real_data_ratio)))

    def generate_with_real_data(self, n: int = 2_000) -> pd.DataFrame:
        from engine import DataGenerator  # local import to avoid circular dep
        log.info("EnhancedDataGenerator: cloud-light mode — using pure synthetic data (%d rows)", n)
        return DataGenerator(seed=42).generate(n=n)


# ============================================================================
# SCHEDULED DATA GATHERER  (no-op stub)
# ============================================================================

class ScheduledDataGatherer:
    """No-op stub. In cloud-light mode no background thread is started."""

    def __init__(self, orchestrator: DataGatheringOrchestrator) -> None:
        self.orchestrator = orchestrator
        self._running     = False

    async def run_periodic_gathering(self, interval_hours: float = 12) -> None:
        log.info("ScheduledDataGatherer: disabled in cloud-light mode")

    def stop(self) -> None:
        self._running = False


# ============================================================================
# PUBLIC API
# ============================================================================

__all__: List[str] = [
    "CompetitorPricePoint",
    "DataGatheringOrchestrator",
    "EnhancedDataGenerator",
    "MarketIntelligence",
    "PricingSignal",
    "ScheduledDataGatherer",
    "SignalDetector",
]
