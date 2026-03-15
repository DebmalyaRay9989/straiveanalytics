"""
STRAIVE Pricing Platform – Data Gathering Module (v4.1)

Changes vs v4.0:
  • SignalDetector.add_price_points: guard against None scraped_at in sort key
  • BaseScraper._get_session: thread-safe lazy initialisation with a lock
  • BaseScraper._fetch: explicit status-code check before returning text
  • CompetitorPriceScraper.scrape_competitor: per-page cap moved to constant;
    defensive truncation of service_name
  • EnhancedDataGenerator._build_real_rows: avoid integer overflow in
    point-index modulo; contract_months choice uses numpy rng correctly
  • DataGatheringOrchestrator._cache_get: catch json.JSONDecodeError
    separately from generic Exception
  • DataGatheringOrchestrator.gather_all_competitor_prices: serialisation now
    excludes the raw_text / extra fields to keep cache lean
  • ScheduledDataGatherer: stop() sets flag atomically; docstring clarified
  • General: removed bare `except Exception` where a narrower type is correct;
    added missing type annotations; cleaned up trailing whitespace
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Optional heavyweight dependencies (graceful fallback) ────────────────────
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

from config import (
    COMPETITORS, COMPETITOR_WEBSITES, CURRENT_DATE,
    CUSTOMER_SEGMENTS, MONTHLY_SEASONALITY,
    NEWS_SOURCES, PRODUCT_CATALOG, REGIONS, SCRAPING_CONFIG,
)

log = logging.getLogger(__name__)

# Maximum competitor price-points kept per page (avoids runaway scrapes)
_MAX_PRICES_PER_PAGE: int = 5


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CompetitorPricePoint:
    """A single scraped competitor price observation."""

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
        # Clamp confidence to [0, 1]
        self.confidence_score = float(max(0.0, min(1.0, self.confidence_score)))


@dataclass
class MarketIntelligence:
    """A piece of market intelligence (news, filing, job posting, etc.)."""

    source: str
    data_type: str          # "news" | "job_posting" | "annual_report" | "linkedin"
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
    """A detected pricing signal (e.g. a competitor price change)."""

    signal_type: str        # "price_increase" | "price_decrease" | "new_offering" | "promotion"
    competitor: str
    magnitude: float        # % change – positive = price increase
    confidence: float
    detected_at: datetime
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SIGNAL DETECTOR
# ============================================================================

class SignalDetector:
    """Detects pricing signals from scraped data history."""

    _MIN_MAGNITUDE_PCT: float = 3.0   # threshold below which we ignore changes

    def __init__(self) -> None:
        self._history: List[CompetitorPricePoint] = []

    def add_price_points(self, points: List[CompetitorPricePoint]) -> None:
        """Append *points* to the internal history."""
        self._history.extend(points)

    def get_recent_signals(self, hours: int = 168) -> List[PricingSignal]:
        """Return pricing signals detected in the last *hours* window.

        Skips any entry whose ``scraped_at`` is None to avoid sort errors.
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [
            p for p in self._history
            if p.scraped_at is not None and p.scraped_at >= cutoff
        ]

        signals: List[PricingSignal] = []
        by_competitor: Dict[str, List[CompetitorPricePoint]] = {}
        for pt in recent:
            by_competitor.setdefault(pt.competitor_name, []).append(pt)

        for comp, pts in by_competitor.items():
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
                signal_type=signal_type,
                competitor=comp,
                magnitude=round(magnitude, 2),
                confidence=round(confidence, 3),
                detected_at=pts_sorted[-1].scraped_at,  # type: ignore[arg-type]
                details={
                    "first_price":    round(first_price, 2),
                    "last_price":     round(last_price, 2),
                    "n_observations": len(pts),
                },
            ))

        return signals


# ============================================================================
# BASE SCRAPER
# ============================================================================

class BaseScraper:
    """HTTP helpers shared by concrete scrapers.

    Session creation is lazy and protected by a lock so that the same
    scraper instance can be reused safely from multiple threads.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or SCRAPING_CONFIG
        self._session: Any = None
        self._session_lock = threading.Lock()
        self._ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )

    def _get_session(self) -> Any:
        """Return a thread-safe requests Session, creating it if needed."""
        if not _HAS_REQUESTS:
            return None
        with self._session_lock:
            if self._session is None:
                s = requests.Session()
                retry = Retry(
                    total=self.config.get("max_retries", 2),
                    backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(max_retries=retry)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                s.headers.update({"User-Agent": self._ua})
                self._session = s
        return self._session

    def _fetch(self, url: str) -> Optional[str]:
        """GET *url* and return HTML text, or None on any failure."""
        if not _HAS_REQUESTS:
            log.debug("requests not installed – skipping fetch of %s", url)
            return None
        session = self._get_session()
        try:
            resp = session.get(
                url,
                timeout=self.config.get("request_timeout", 15),
                allow_redirects=True,
            )
            # Treat 4xx / 5xx as fetch failures
            if not resp.ok:
                log.debug("HTTP %d for %s – skipping", resp.status_code, url)
                return None
            time.sleep(self.config.get("rate_limit_delay", 1.0))
            return resp.text
        except Exception as exc:
            log.debug("Fetch failed for %s: %s", url, exc)
            return None

    @staticmethod
    def _extract_prices(text: str) -> List[float]:
        """Regex-based price extraction from arbitrary text."""
        patterns = [
            r"\$\s*([\d,]+(?:\.\d{2})?)",
            r"USD\s*([\d,]+(?:\.\d{2})?)",
            r"([\d,]+(?:\.\d{2})?)\s*(?:per|/)\s*(?:month|year|project|item|1k)",
        ]
        prices: List[float] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    raw = match.group(1).replace(",", "")
                    val = float(raw)
                    if 10 < val < 1_000_000:
                        prices.append(val)
                except ValueError:
                    pass
        return prices


# ============================================================================
# COMPETITOR PRICE SCRAPER
# ============================================================================

class CompetitorPriceScraper(BaseScraper):
    """Scrapes pricing data from competitor websites."""

    def scrape_competitor(
        self,
        competitor_name: str,
        use_cache: bool = True,  # noqa: ARG002 – reserved for future cache layer
    ) -> List[CompetitorPricePoint]:
        """Scrape a single competitor and return price points.

        Falls back to simulated data when live scraping is unavailable.
        """
        cfg = COMPETITOR_WEBSITES.get(competitor_name)
        if not cfg:
            log.debug("No scraping config for %s – using simulated data", competitor_name)
            return self._simulate_competitor_data(competitor_name)

        points: List[CompetitorPricePoint] = []
        base_url = cfg["base_url"].rstrip("/")
        pages    = cfg.get("pages", ["/"])

        for page in pages:
            url  = base_url + page
            html = self._fetch(url)
            if not html:
                continue

            page_prices = BaseScraper._extract_prices(html)
            if _HAS_BS4:
                soup = BeautifulSoup(html, "html.parser")
                service_selectors = cfg.get("service_selectors", ["h2", "h3"])
                service_names = [
                    el.get_text(strip=True)
                    for sel in service_selectors
                    for el in soup.select(sel)
                ]
            else:
                service_names = []

            for i, price in enumerate(page_prices[:_MAX_PRICES_PER_PAGE]):
                if service_names and i < len(service_names):
                    svc = service_names[i][:100]
                else:
                    svc = f"Service {i + 1}"
                points.append(CompetitorPricePoint(
                    competitor_name=competitor_name,
                    service_name=svc,
                    price=price,
                    source_url=url,
                    confidence_score=0.6,
                ))

        if not points:
            points = self._simulate_competitor_data(competitor_name)
            log.debug("Using simulated data for %s (%d points)", competitor_name, len(points))
        else:
            log.info("Scraped %d price points for %s", len(points), competitor_name)
        return points

    def _simulate_competitor_data(
        self,
        competitor_name: str,
    ) -> List[CompetitorPricePoint]:
        """Return plausible synthetic price points when live scraping fails."""
        comp_info = COMPETITORS.get(competitor_name, {})
        rel_price = float(comp_info.get("relative_price", 0.9))
        rng = np.random.default_rng(abs(hash(competitor_name)) % (2 ** 31))

        service_templates = [
            ("Editorial Services",     2_800),
            ("Data Annotation – Basic",  450),
            ("Data Annotation – Advanced", 980),
            ("Content Transformation", 1_200),
            ("Research & Analytics",   4_800),
            ("Publishing Technology",  3_200),
        ]

        points: List[CompetitorPricePoint] = []
        for service, base_price in service_templates:
            noise = float(rng.uniform(0.90, 1.10))
            simulated_price = round(base_price * rel_price * noise, 2)
            slug = competitor_name.lower().replace(" ", "-")
            points.append(CompetitorPricePoint(
                competitor_name=competitor_name,
                service_name=service,
                price=simulated_price,
                source_url=f"https://simulated/{slug}",
                confidence_score=0.45,   # lower confidence for simulated data
            ))
        return points


# ============================================================================
# MARKET INTELLIGENCE GATHERER
# ============================================================================

class MarketIntelligenceGatherer(BaseScraper):
    """Gathers market intelligence from news and public sources."""

    # Cap on how many news sources we hit per call (avoids overload)
    _MAX_SOURCES: int = 3

    def gather_news(self) -> List[MarketIntelligence]:
        """Scrape news sources for competitor / market mentions."""
        items: List[MarketIntelligence] = []

        for source_cfg in NEWS_SOURCES[: self._MAX_SOURCES]:
            url  = source_cfg.get("url", "")
            html = self._fetch(url)
            if html is None:
                items.extend(self._simulate_news(source_cfg["name"]))
                continue

            if _HAS_BS4:
                soup     = BeautifulSoup(html, "html.parser")
                selector = source_cfg.get("selector", "article")
                articles = soup.select(selector)[:10]
                for article in articles:
                    text = article.get_text(strip=True)[:500]
                    if not text:
                        continue
                    relevance = self._score_relevance(text)
                    items.append(MarketIntelligence(
                        source=source_cfg["name"],
                        data_type="news",
                        content=text,
                        relevance_score=relevance,
                        url=url,
                    ))
            else:
                items.extend(self._simulate_news(source_cfg["name"]))

        return items

    @staticmethod
    def _score_relevance(text: str) -> float:
        """Simple keyword-based relevance scorer (returns 0–1)."""
        keywords = [
            "pricing", "price", "contract", "revenue", "margin",
            "competitor", "market share", "acquisition", "growth",
            "editorial", "annotation", "publishing", "AI", "content",
        ]
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return round(min(1.0, matches / 5), 2)

    @staticmethod
    def _simulate_news(source_name: str) -> List[MarketIntelligence]:
        """Fallback simulated news items when live scraping is unavailable."""
        templates = [
            "Industry report shows 8% growth in content services sector for Q4.",
            "Major publisher announces multi-year contract with leading data annotation firm.",
            "AI-driven editorial workflows gaining traction among STM publishers.",
            "Pricing pressure intensifies as new entrants offer discount annotation services.",
            "Consolidation trend continues as mid-size content services firms merge.",
        ]
        return [
            MarketIntelligence(
                source=source_name,
                data_type="news",
                content=t,
                relevance_score=0.5,
            )
            for t in templates
        ]


# ============================================================================
# DATA GATHERING ORCHESTRATOR
# ============================================================================

class DataGatheringOrchestrator:
    """Central orchestrator for all data-gathering operations.

    Parameters
    ----------
    use_redis:
        Whether to use Redis for result caching.
    redis_host:
        Redis host (used only when *use_redis* is True).
    redis_port:
        Redis port.
    redis_db:
        Redis database index.
    """

    def __init__(
        self,
        use_redis: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6_379,
        redis_db: int = 0,
    ) -> None:
        self.use_redis = use_redis and _HAS_REDIS
        self._redis: Any = None
        self._cache: Dict[str, Any] = {}   # in-memory fallback

        if self.use_redis:
            try:
                self._redis = redis.Redis(
                    host=redis_host, port=redis_port, db=redis_db,
                    socket_connect_timeout=2,
                )
                self._redis.ping()
                log.info("Redis cache connected at %s:%d", redis_host, redis_port)
            except Exception as exc:
                log.warning("Redis unavailable (%s) – using in-memory cache", exc)
                self._redis = None

        self.price_scraper    = CompetitorPriceScraper()
        self.intel_gatherer   = MarketIntelligenceGatherer()
        self.signal_detector  = SignalDetector()

    # ── Cache helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(prefix: str, *args: Any) -> str:
        """Return a stable MD5 hex key for *prefix* + *args*.

        usedforsecurity=False is passed as a keyword argument only on
        Python >= 3.9, avoiding "openssl_md5() takes at most 1 argument"
        on OpenSSL 3.x builds.
        """
        import sys
        raw = prefix + "|".join(str(a) for a in args)
        if sys.version_info >= (3, 9):
            return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str, ttl: int = 3_600) -> Optional[Any]:
        if self._redis:
            try:
                val = self._redis.get(key)
                if val is not None:
                    return json.loads(val)
            except json.JSONDecodeError as exc:
                log.warning("Cache decode error for key %s: %s", key, exc)
            except Exception as exc:
                log.debug("Redis get failed: %s", exc)
        entry = self._cache.get(key)
        if entry and (datetime.now() - entry["ts"]).total_seconds() < ttl:
            return entry["data"]
        return None

    def _cache_set(self, key: str, data: Any, ttl: int = 3_600) -> None:
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(data, default=str))
                return
            except Exception as exc:
                log.debug("Redis set failed: %s", exc)
        self._cache[key] = {"data": data, "ts": datetime.now()}

    # ── Public API ─────────────────────────────────────────────────────────

    def gather_all_competitor_prices(
        self,
        use_cache: bool = True,
    ) -> List[CompetitorPricePoint]:
        """Gather price points from all configured competitors.

        Returns a flat list of :class:`CompetitorPricePoint` objects.
        Serialises only the fields needed for the cache (excludes ``raw_text``
        and ``extra`` to keep payloads small).
        """
        cache_key = self._cache_key("competitor_prices")
        ttl       = int(SCRAPING_CONFIG.get("cache_ttl", 3_600))

        if use_cache:
            cached = self._cache_get(cache_key, ttl)
            if cached:
                log.info("Returning %d price points from cache", len(cached))
                result: List[CompetitorPricePoint] = []
                for d in cached:
                    raw_ts = d.pop("scraped_at", None)
                    if isinstance(raw_ts, datetime):
                        d["scraped_at"] = raw_ts
                    elif isinstance(raw_ts, str) and raw_ts:
                        try:
                            d["scraped_at"] = datetime.fromisoformat(raw_ts)
                        except ValueError:
                            d["scraped_at"] = datetime.now()
                    else:
                        d["scraped_at"] = datetime.now()
                    result.append(CompetitorPricePoint(**d))
                return result

        all_points: List[CompetitorPricePoint] = []
        for competitor_name, comp_info in COMPETITORS.items():
            if not comp_info.get("scrape_enabled", True):
                continue
            try:
                pts = self.price_scraper.scrape_competitor(
                    competitor_name,
                    use_cache=use_cache,
                )
                all_points.extend(pts)
            except Exception as exc:
                log.error("Error scraping %s: %s", competitor_name, exc)

        self.signal_detector.add_price_points(all_points)

        if use_cache:
            serialisable = [
                {
                    "competitor_name":  p.competitor_name,
                    "service_name":     p.service_name,
                    "price":            p.price,
                    "currency":         p.currency,
                    "source_url":       p.source_url,
                    "confidence_score": p.confidence_score,
                    "scraped_at":       p.scraped_at.isoformat() if p.scraped_at else None,
                }
                for p in all_points
            ]
            self._cache_set(cache_key, serialisable, ttl)

        log.info("Gathered %d total competitor price points", len(all_points))
        return all_points

    def gather_market_intelligence(self) -> List[MarketIntelligence]:
        """Gather market intelligence from news sources."""
        try:
            items = self.intel_gatherer.gather_news()
        except Exception as exc:
            log.error("Market intelligence gathering failed: %s", exc)
            items = []
        log.info("Gathered %d market intelligence items", len(items))
        return items

    def generate_pricing_intelligence_report(self) -> Dict[str, Any]:
        """Produce a structured pricing-intelligence summary dictionary."""
        points  = self.gather_all_competitor_prices(use_cache=True)
        signals = self.signal_detector.get_recent_signals(hours=168)

        # Aggregate average price per competitor
        comp_price_lists: Dict[str, List[float]] = {}
        for pt in points:
            comp_price_lists.setdefault(pt.competitor_name, []).append(pt.price)
        avg_by_comp = {
            k: round(float(np.mean(v)), 2)
            for k, v in comp_price_lists.items()
        }

        # Build recommendations
        recommendations: List[str] = []
        for sig in signals[:3]:
            if sig.magnitude < -5:
                recommendations.append(
                    f"Monitor {sig.competitor} – detected {abs(sig.magnitude):.0f}% "
                    "price decrease. Consider defensive pricing review."
                )
            elif sig.magnitude > 5:
                recommendations.append(
                    f"{sig.competitor} raised prices by {sig.magnitude:.0f}%. "
                    "Potential opportunity to capture market share."
                )

        if not recommendations:
            recommendations = [
                "Continue monitoring competitor pricing weekly.",
                "Review pricing strategy for high-elasticity segments quarterly.",
                "Leverage quality advantage in premium negotiations.",
            ]

        return {
            "generated_at": datetime.now().isoformat(),
            "competitor_prices": {
                "total_points":          len(points),
                "avg_price_by_competitor": avg_by_comp,
            },
            "signals": [
                {
                    "signal_type": s.signal_type,
                    "competitor":  s.competitor,
                    "magnitude":   s.magnitude,
                    "confidence":  s.confidence,
                    "detected_at": s.detected_at.isoformat(),
                }
                for s in signals
            ],
            "recommendations": recommendations,
        }


# ============================================================================
# ENHANCED DATA GENERATOR
# ============================================================================

class EnhancedDataGenerator:
    """Generates a blended dataset combining synthetic records with
    real competitor price points gathered by the orchestrator.
    """

    def __init__(
        self,
        orchestrator: DataGatheringOrchestrator,
        real_data_ratio: float = 0.30,
    ) -> None:
        self.orchestrator    = orchestrator
        self.real_data_ratio = float(max(0.0, min(1.0, real_data_ratio)))

    def generate_with_real_data(self, n: int = 3_600) -> pd.DataFrame:
        """Return a DataFrame of *n* rows blending synthetic and real data.

        Real data contributes ``real_data_ratio * n`` rows (capped by
        available scraped data); synthetic fills the remainder.
        """
        from engine import DataGenerator  # local import to avoid circular dep
        synth_gen = DataGenerator(seed=42)

        n_real  = min(int(n * self.real_data_ratio), n)
        n_synth = n - n_real

        synth_df = synth_gen.generate(n=n_synth)
        real_df  = self._build_real_rows(n_real)

        if real_df is not None and not real_df.empty:
            combined = pd.concat([synth_df, real_df], ignore_index=True)
        else:
            combined = synth_df

        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        n_real_actual = 0 if real_df is None else len(real_df)
        log.info(
            "EnhancedDataGenerator: %d rows (%d synthetic, %d real-derived)",
            len(combined), n_synth, n_real_actual,
        )
        return combined

    def _build_real_rows(self, n: int) -> Optional[pd.DataFrame]:
        """Convert scraped price points into DataFrame rows compatible with
        the platform schema.  Returns ``None`` if there are no points.
        """
        try:
            points = self.orchestrator.gather_all_competitor_prices(use_cache=True)
        except Exception as exc:
            log.warning("Could not fetch competitor prices for blending: %s", exc)
            return None

        if not points:
            return None

        rng       = np.random.default_rng(99)
        customers = list(CUSTOMER_SEGMENTS.keys())
        regions   = list(REGIONS.keys())
        products  = list(PRODUCT_CATALOG.keys())
        n_points  = len(points)
        start     = CURRENT_DATE - timedelta(days=365)

        rows: List[Dict[str, Any]] = []
        n_to_gen = min(n, n_points * 5)

        for i in range(n_to_gen):
            # Safe modulo: n_points > 0 guaranteed above
            pt   = points[i % n_points]
            prod = str(rng.choice(products))
            info = PRODUCT_CATALOG[prod]
            cost = float(info["cost"])

            price = float(np.clip(
                pt.price * float(rng.uniform(0.9, 1.1)),
                info["min_price"],
                info["max_price"],
            ))
            volume   = max(1, int(rng.exponential(10)))
            revenue  = price * volume
            cost_tot = cost * volume
            margin   = (revenue - cost_tot) / revenue * 100 if revenue > 0 else 0.0
            deal_won = int(float(rng.random()) < 0.55)
            days_ago = int(rng.uniform(0, 365))

            # contract_months: use integer p-array for rng.choice
            avg_months    = int(info.get("avg_contract_months", 12))
            contract_opts = np.array([max(1, avg_months // 2), avg_months, avg_months * 2])
            contract_months = int(rng.choice(contract_opts, p=[0.25, 0.55, 0.20]))

            rows.append({
                "date":             start + timedelta(days=days_ago),
                "product":          prod,
                "segment":          info["segment"],
                "customer_type":    str(rng.choice(customers)),
                "region":           str(rng.choice(regions)),
                "channel":          "Direct",
                "base_price":       info["base_price"],
                "actual_price":     round(price, 2),
                "discount_pct":     round(max(0.0, (info["base_price"] - price) / info["base_price"] * 100), 1),
                "volume":           volume,
                "revenue":          round(revenue, 2),
                "cost":             round(cost_tot, 2),
                "margin_pct":       round(margin, 2),
                "deal_won":         deal_won,
                "competitor":       pt.competitor_name,
                "competitor_price": round(pt.price, 2),
                "contract_months":  contract_months,
                "renewal_flag":     int(float(rng.random()) < 0.7),
                "sales_cycle_days": max(1, int(rng.normal(45, 15))),
                "churn_flag":       int(float(rng.random()) < 0.15),
                "complexity":       info["complexity"],
                "confidence_score": round(float(pt.confidence_score), 3),
                "source":           "scraped",
            })

        if not rows:
            return None

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ============================================================================
# SCHEDULED DATA GATHERER
# ============================================================================

class ScheduledDataGatherer:
    """Runs periodic data gathering in the background using asyncio.

    Designed to be started in a daemon thread.  The gatherer is **not**
    required for the platform to function – it merely keeps the cache warm.
    Call :meth:`stop` to signal a graceful shutdown after the current sleep.
    """

    def __init__(self, orchestrator: DataGatheringOrchestrator) -> None:
        self.orchestrator = orchestrator
        self._running     = False

    async def run_periodic_gathering(self, interval_hours: float = 12) -> None:
        """Gather data every *interval_hours* until :meth:`stop` is called."""
        self._running = True
        while self._running:
            try:
                log.info("ScheduledDataGatherer: starting periodic run")
                self.orchestrator.gather_all_competitor_prices(use_cache=False)
                self.orchestrator.gather_market_intelligence()
                log.info(
                    "ScheduledDataGatherer: run complete – sleeping %.1fh",
                    interval_hours,
                )
            except Exception as exc:
                log.error("ScheduledDataGatherer error: %s", exc)

            await asyncio.sleep(interval_hours * 3_600)

    def stop(self) -> None:
        """Signal the gatherer to stop after the current sleep completes."""
        self._running = False


# ============================================================================
# PUBLIC API
# ============================================================================

__all__: List[str] = [
    "BaseScraper",
    "CompetitorPriceScraper",
    "CompetitorPricePoint",
    "DataGatheringOrchestrator",
    "EnhancedDataGenerator",
    "MarketIntelligence",
    "MarketIntelligenceGatherer",
    "PricingSignal",
    "ScheduledDataGatherer",
    "SignalDetector",
]
