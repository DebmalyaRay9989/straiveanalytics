"""
STRAIVE Pricing Platform – Configuration Module (v5.0 Cloud-Light)

Changes vs v4.1:
  • Removed all web-scraping, Selenium, Redis, LinkedIn config (heavy deps)
  • Removed COMPETITOR_WEBSITES, LINKEDIN_CONFIG, NEWS_SOURCES
  • SCRAPING_CONFIG stripped to minimal stub (disabled by default)
  • Removed OUTPUT_DIR filesystem side-effect at import
  • Trimmed unused PREPROCESSING_OPTIONS, FORECAST_CONFIG verbosity
  • Kept all business logic: products, segments, competitors, strategies, nav
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

CURRENT_DATE: datetime = datetime.now()
APP_VERSION: str = "5.0.0"
APP_TITLE: str = "STRAIVE · Dynamic Pricing & Revenue Intelligence Platform"

# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING CONFIG (disabled in cloud-light mode)
# ─────────────────────────────────────────────────────────────────────────────
SCRAPING_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "cache_enabled": False,
    "enable_scheduled_gathering": False,
}

# Stub exports so any old import still resolves without error
COMPETITOR_WEBSITES: Dict[str, Any] = {}
LINKEDIN_CONFIG: Dict[str, Any] = {"enabled": False}
NEWS_SOURCES: List[Dict[str, Any]] = []

# ─────────────────────────────────────────────────────────────────────────────
# CSV UPLOAD SPECIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
CSV_UPLOAD_SPEC: Dict[str, Any] = {
    "template_columns": [
        "date", "product", "customer_type", "region",
        "actual_price", "volume", "revenue", "cost",
        "deal_won", "base_price", "discount_pct",
        "competitor", "competitor_price", "source", "confidence_score",
    ],
    "date_formats": [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y",
    ],
    "supported_file_formats": [".csv", ".xlsx", ".xls"],
    "max_file_size_mb": 20,
    "min_rows": 100,
    "encoding_options": ["utf-8", "latin-1", "cp1252"],
}

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING OPTIONS (kept lean)
# ─────────────────────────────────────────────────────────────────────────────
PREPROCESSING_OPTIONS: Dict[str, Any] = {
    "handle_missing": ["drop_rows", "impute_mean", "impute_median"],
    "outlier_detection": ["iqr", "zscore", "none"],
    "normalization": ["none", "minmax", "standard"],
}

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT CATALOG
# ─────────────────────────────────────────────────────────────────────────────
PRODUCT_CATALOG: Dict[str, Any] = {
    # ── Editorial ──────────────────────────────────────────────────────────
    "Editorial Services – Standard": {
        "segment": "Editorial", "base_price": 2_800, "cost": 1_120,
        "min_price": 1_400, "max_price": 5_600, "complexity": 2,
        "avg_contract_months": 12,
        "keywords": ["editorial", "copyediting", "proofreading"],
        "tags": ["core", "high-volume"],
    },
    "Editorial Services – Premium": {
        "segment": "Editorial", "base_price": 5_500, "cost": 1_980,
        "min_price": 2_750, "max_price": 11_000, "complexity": 3,
        "avg_contract_months": 18,
        "keywords": ["editorial", "premium", "journal"],
        "tags": ["premium"],
    },
    "Editorial Services – Express": {
        "segment": "Editorial", "base_price": 3_800, "cost": 1_596,
        "min_price": 1_900, "max_price": 7_600, "complexity": 2,
        "avg_contract_months": 6,
        "keywords": ["editorial", "express", "rush"],
        "tags": ["time-sensitive"],
    },
    # ── Data Services ──────────────────────────────────────────────────────
    "Data Annotation – Basic": {
        "segment": "Data Services", "base_price": 450, "cost": 180,
        "min_price": 225, "max_price": 900, "complexity": 1,
        "avg_contract_months": 6,
        "keywords": ["annotation", "labeling", "tagging"],
        "tags": ["high-volume", "scalable"],
    },
    "Data Annotation – Advanced": {
        "segment": "Data Services", "base_price": 850, "cost": 306,
        "min_price": 425, "max_price": 1_700, "complexity": 3,
        "avg_contract_months": 12,
        "keywords": ["annotation", "complex", "NLP"],
        "tags": ["specialist"],
    },
    "Data Conversion Services": {
        "segment": "Data Services", "base_price": 1_200, "cost": 480,
        "min_price": 600, "max_price": 2_400, "complexity": 2,
        "avg_contract_months": 9,
        "keywords": ["conversion", "migration", "XML"],
        "tags": ["core"],
    },
    "Metadata Enrichment": {
        "segment": "Data Services", "base_price": 680, "cost": 272,
        "min_price": 340, "max_price": 1_360, "complexity": 2,
        "avg_contract_months": 12,
        "keywords": ["metadata", "enrichment", "tagging"],
        "tags": ["recurring"],
    },
    # ── AI/ML ──────────────────────────────────────────────────────────────
    "AI Content Extraction": {
        "segment": "AI/ML", "base_price": 4_200, "cost": 1_260,
        "min_price": 2_100, "max_price": 8_400, "complexity": 4,
        "avg_contract_months": 18,
        "keywords": ["AI", "extraction", "NLP", "machine learning"],
        "tags": ["high-growth", "premium"],
    },
    "ML Model Training Support": {
        "segment": "AI/ML", "base_price": 8_500, "cost": 2_550,
        "min_price": 4_250, "max_price": 17_000, "complexity": 5,
        "avg_contract_months": 24,
        "keywords": ["ML", "training", "model", "deep learning"],
        "tags": ["enterprise", "strategic"],
    },
    "AI Quality Assurance": {
        "segment": "AI/ML", "base_price": 3_200, "cost": 960,
        "min_price": 1_600, "max_price": 6_400, "complexity": 3,
        "avg_contract_months": 12,
        "keywords": ["AI", "QA", "validation", "testing"],
        "tags": ["growing"],
    },
    # ── Content ────────────────────────────────────────────────────────────
    "Content Production – Standard": {
        "segment": "Content", "base_price": 1_800, "cost": 720,
        "min_price": 900, "max_price": 3_600, "complexity": 2,
        "avg_contract_months": 12,
        "keywords": ["content", "writing", "production"],
        "tags": ["core"],
    },
    "Content Production – Premium": {
        "segment": "Content", "base_price": 3_500, "cost": 1_260,
        "min_price": 1_750, "max_price": 7_000, "complexity": 3,
        "avg_contract_months": 18,
        "keywords": ["content", "premium", "multimedia"],
        "tags": ["premium"],
    },
    "Digital Publishing Services": {
        "segment": "Content", "base_price": 2_600, "cost": 1_040,
        "min_price": 1_300, "max_price": 5_200, "complexity": 3,
        "avg_contract_months": 12,
        "keywords": ["digital", "publishing", "ebook"],
        "tags": ["digital-first"],
    },
    # ── Analytics ──────────────────────────────────────────────────────────
    "Analytics Dashboard": {
        "segment": "Analytics", "base_price": 6_500, "cost": 1_950,
        "min_price": 3_250, "max_price": 13_000, "complexity": 4,
        "avg_contract_months": 24,
        "keywords": ["analytics", "dashboard", "reporting"],
        "tags": ["SaaS", "recurring"],
    },
    "Usage & Compliance Reporting": {
        "segment": "Analytics", "base_price": 3_800, "cost": 1_140,
        "min_price": 1_900, "max_price": 7_600, "complexity": 3,
        "avg_contract_months": 12,
        "keywords": ["compliance", "reporting", "usage"],
        "tags": ["regulatory"],
    },
    # ── Technology ─────────────────────────────────────────────────────────
    "Platform Integration Services": {
        "segment": "Technology", "base_price": 12_000, "cost": 3_600,
        "min_price": 6_000, "max_price": 24_000, "complexity": 5,
        "avg_contract_months": 24,
        "keywords": ["integration", "API", "platform"],
        "tags": ["enterprise", "high-value"],
    },
    "Workflow Automation": {
        "segment": "Technology", "base_price": 7_500, "cost": 2_250,
        "min_price": 3_750, "max_price": 15_000, "complexity": 4,
        "avg_contract_months": 18,
        "keywords": ["automation", "workflow", "efficiency"],
        "tags": ["high-growth"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOMER_SEGMENTS: Dict[str, Any] = {
    "STM Publishers": {
        "price_sensitivity": 0.55, "volume_multiplier": 1.4,
        "loyalty": 0.82, "avg_contract_months": 18,
        "description": "Scientific, Technical & Medical publishers",
    },
    "Academic Publishers": {
        "price_sensitivity": 0.70, "volume_multiplier": 1.2,
        "loyalty": 0.75, "avg_contract_months": 12,
        "description": "University presses & academic societies",
    },
    "Trade Publishers": {
        "price_sensitivity": 0.80, "volume_multiplier": 0.9,
        "loyalty": 0.60, "avg_contract_months": 9,
        "description": "Consumer & trade book publishers",
    },
    "EdTech Companies": {
        "price_sensitivity": 0.65, "volume_multiplier": 1.3,
        "loyalty": 0.70, "avg_contract_months": 15,
        "description": "Educational technology companies",
    },
    "Media Corporations": {
        "price_sensitivity": 0.50, "volume_multiplier": 1.6,
        "loyalty": 0.85, "avg_contract_months": 24,
        "description": "Large media & publishing groups",
    },
    "Government & NGO": {
        "price_sensitivity": 0.40, "volume_multiplier": 0.8,
        "loyalty": 0.90, "avg_contract_months": 36,
        "description": "Government bodies & non-profits",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# REGIONS
# ─────────────────────────────────────────────────────────────────────────────
REGIONS: Dict[str, Any] = {
    "North America": {
        "demand_index": 1.30, "price_premium": 1.15,
        "currency": "USD",
    },
    "Europe": {
        "demand_index": 1.15, "price_premium": 1.10,
        "currency": "EUR",
    },
    "Asia Pacific": {
        "demand_index": 1.20, "price_premium": 0.90,
        "currency": "USD",
    },
    "Middle East & Africa": {
        "demand_index": 0.75, "price_premium": 0.95,
        "currency": "USD",
    },
    "Latin America": {
        "demand_index": 0.80, "price_premium": 0.85,
        "currency": "USD",
    },
    "South Asia": {
        "demand_index": 0.90, "price_premium": 0.80,
        "currency": "USD",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# COMPETITORS
# ─────────────────────────────────────────────────────────────────────────────
COMPETITORS: Dict[str, Any] = {
    "Aptara":              {"relative_price": 0.92, "quality_score": 0.78, "market_share": 0.18},
    "Innodata":            {"relative_price": 0.85, "quality_score": 0.72, "market_share": 0.14},
    "MPS Limited":         {"relative_price": 0.95, "quality_score": 0.80, "market_share": 0.12},
    "Cenveo":              {"relative_price": 0.88, "quality_score": 0.75, "market_share": 0.10},
    "Techbooks":           {"relative_price": 0.80, "quality_score": 0.68, "market_share": 0.08},
    "SPi Global":          {"relative_price": 0.90, "quality_score": 0.76, "market_share": 0.11},
    "Scribendi":           {"relative_price": 0.75, "quality_score": 0.65, "market_share": 0.07},
    "Clarivate Analytics": {"relative_price": 1.10, "quality_score": 0.88, "market_share": 0.15},
}

# ─────────────────────────────────────────────────────────────────────────────
# PRICING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
PRICING_STRATEGIES: Dict[str, Any] = {
    "value_based":     {"description": "Price based on perceived customer value",    "typical_premium": 1.20},
    "cost_plus":       {"description": "Cost plus target margin",                    "typical_premium": 1.00},
    "competitive":     {"description": "Price relative to competitors",              "typical_premium": 0.97},
    "penetration":     {"description": "Low price to gain market share",             "typical_premium": 0.85},
    "premium":         {"description": "Premium pricing for quality differentiation","typical_premium": 1.35},
    "dynamic":         {"description": "Real-time price adjustment",                 "typical_premium": 1.10},
}

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_LIST: List[str] = ["Direct", "Partner", "Online", "Reseller"]
CHANNEL_WEIGHTS: List[float] = [0.50, 0.25, 0.15, 0.10]
CHANNEL_PRICE_ADJ: Dict[str, float] = {
    "Direct": 0.00, "Partner": -0.05, "Online": -0.10, "Reseller": -0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# SEASONALITY
# ─────────────────────────────────────────────────────────────────────────────
MONTHLY_SEASONALITY: Dict[int, float] = {
    1: 0.88, 2: 0.92, 3: 1.05, 4: 1.08, 5: 1.10, 6: 1.06,
    7: 0.90, 8: 0.87, 9: 1.08, 10: 1.12, 11: 1.15, 12: 0.95,
}

# ─────────────────────────────────────────────────────────────────────────────
# MARGIN WATERFALL BUCKETS
# ─────────────────────────────────────────────────────────────────────────────
MARGIN_WATERFALL_BUCKETS: Dict[str, float] = {
    "Base Revenue":      1.00,
    "Volume Discount":  -0.08,
    "Promotional":      -0.04,
    "Channel Cost":     -0.06,
    "COGS":             -0.38,
    "Overhead":         -0.12,
    "Net Margin":        0.32,
}

_mwb_total = sum(MARGIN_WATERFALL_BUCKETS.values())
if abs(_mwb_total) > 0.001:
    log.debug("MARGIN_WATERFALL_BUCKETS net = %.4f", _mwb_total)

# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION GUIDANCE
# ─────────────────────────────────────────────────────────────────────────────
DATA_COLLECTION_GUIDANCE: Dict[str, Any] = {
    "minimum_required_columns": [
        "date", "product", "actual_price", "volume", "revenue", "cost", "deal_won",
    ],
    "column_mappings": {
        "actual_price":  ["price", "unit_price", "selling_price", "sale_price"],
        "deal_won":      ["won", "closed_won", "outcome", "result", "win"],
        "customer_type": ["customer", "client_type", "segment", "account_type"],
        "volume":        ["qty", "quantity", "units"],
    },
    "tips": [
        "Include at least 500 rows for reliable elasticity estimates.",
        "Ensure date spans ≥ 6 months for seasonality detection.",
        "Competitor price columns improve competitive analysis.",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# FORECAST CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FORECAST_CONFIG: Dict[str, Any] = {
    "horizon_months": 12,
    "confidence_interval": 0.90,
    "trend_dampening": 0.85,
    "seasonality_weight": 0.60,
    "min_obs_for_arima": 24,
    "ensemble_methods": ["linear", "exp_smoothing"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
SCORING_WEIGHTS: Dict[str, float] = {
    "margin_pct":            0.25,
    "revenue_growth":        0.20,
    "win_rate":              0.15,
    "customer_loyalty":      0.10,
    "competitive_moat":      0.10,
    "market_position":       0.10,
    "price_competitiveness": 0.10,
}

_sw_total = sum(SCORING_WEIGHTS.values())
if abs(_sw_total - 1.0) > 0.001:
    log.warning("SCORING_WEIGHTS sum to %.4f – normalising.", _sw_total)
    SCORING_WEIGHTS = {k: v / _sw_total for k, v in SCORING_WEIGHTS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PRODUCT_SEGMENTS: List[str] = sorted({v["segment"] for v in PRODUCT_CATALOG.values()})
PRODUCT_NAMES: List[str] = sorted(PRODUCT_CATALOG.keys())
COMPETITOR_NAMES: List[str] = sorted(COMPETITORS.keys())

SEGMENT_COLORS: Dict[str, str] = {
    "Editorial":      "#4f9eff",
    "Data Services":  "#36d97b",
    "AI/ML":          "#b48eff",
    "Content":        "#ff9800",
    "Analytics":      "#ffd700",
    "Technology":     "#ff6b6b",
}

COMPETITOR_COLORS: Dict[str, str] = {
    "Aptara":              "#1f77b4",
    "Innodata":            "#ff7f0e",
    "MPS Limited":         "#2ca02c",
    "Cenveo":              "#d62728",
    "Techbooks":           "#9467bd",
    "SPi Global":          "#8c564b",
    "Scribendi":           "#e377c2",
    "Clarivate Analytics": "#7f7f7f",
}

# ─────────────────────────────────────────────────────────────────────────────
# THEME & STYLING
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG    = "#070b14"
CARD_BG    = "#0c1221"
CARD_BG2   = "#111827"
SIDEBAR_BG = "#090e1c"
BORDER     = "#1a2540"
BORDER2    = "#243058"
ACCENT     = "#4f9eff"
ACCENT2    = "#ff6b6b"
ACCENT3    = "#ffd700"
TEXT       = "#e8edf5"
MUTED      = "#6b7a99"
GREEN      = "#2ee89a"
YELLOW     = "#f5a623"
RED        = "#ff4060"
PURPLE     = "#a78bfa"
CYAN       = "#06d6c8"
ORANGE     = "#ff9800"

PLOTLY_DARK: Dict[str, Any] = {
    "paper_bgcolor": DARK_BG,
    "plot_bgcolor":  "#0c1525",
    "font":        {"color": TEXT, "family": "'DM Sans','Segoe UI',sans-serif", "size": 12},
    "title_font":  {"color": TEXT, "size": 15, "family": "'Space Grotesk','Segoe UI',sans-serif"},
    "legend":      {"bgcolor": "rgba(12,18,33,0.92)", "bordercolor": BORDER2, "borderwidth": 1,
                    "font": {"color": TEXT, "size": 11}},
    "margin":      {"l": 14, "r": 14, "t": 52, "b": 14},
    "hoverlabel":  {"bgcolor": CARD_BG, "bordercolor": BORDER2, "font": {"color": TEXT, "size": 12}},
    "xaxis":       {"gridcolor": BORDER, "zerolinecolor": BORDER2, "linecolor": BORDER},
    "yaxis":       {"gridcolor": BORDER, "zerolinecolor": BORDER2, "linecolor": BORDER},
}

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION  (Market Intelligence tab removed in cloud-light)
# ─────────────────────────────────────────────────────────────────────────────
NAV_OPTIONS: List[str] = [
    "📊 Executive Dashboard",
    "🔍 Elasticity Analysis",
    "💡 Optimal Pricing",
    "📈 Revenue Simulator",
    "🎯 Price-Volume Curves",
    "⚔️ Competitive Positioning",
    "🌍 Regional Pricing",
    "👥 Segment Intelligence",
    "🔧 What-If Scenarios",
    "🤝 Win-Rate Analysis",
    "📉 Margin Waterfall",
    "📦 Product Portfolio",
    "⚠️ Risk & Sensitivity",
    "🗓️ Seasonality & Trends",
    "🔮 Revenue Forecast",
]

NAV_GROUPS: Dict[str, List[str]] = {
    "OVERVIEW":     ["📊 Executive Dashboard"],
    "PRICING":      [
        "🔍 Elasticity Analysis", "💡 Optimal Pricing",
        "📈 Revenue Simulator",   "🎯 Price-Volume Curves",
    ],
    "MARKET":       [
        "⚔️ Competitive Positioning", "🌍 Regional Pricing",
        "👥 Segment Intelligence",
    ],
    "SCENARIOS":    [
        "🔧 What-If Scenarios", "🤝 Win-Rate Analysis",
        "📉 Margin Waterfall",  "📦 Product Portfolio",
    ],
    "INTELLIGENCE": [
        "⚠️ Risk & Sensitivity", "🗓️ Seasonality & Trends",
        "🔮 Revenue Forecast",
    ],
}

NAV_INDEX: Dict[str, int] = {label: i for i, label in enumerate(NAV_OPTIONS)}


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def validate_product(name: str) -> bool:
    return isinstance(name, str) and name in PRODUCT_CATALOG

def validate_segment(name: str) -> bool:
    return isinstance(name, str) and name in CUSTOMER_SEGMENTS

def validate_region(name: str) -> bool:
    return isinstance(name, str) and name in REGIONS

def validate_competitor(name: str) -> bool:
    return isinstance(name, str) and name in COMPETITORS

def validate_strategy(name: str) -> bool:
    return isinstance(name, str) and name in PRICING_STRATEGIES

def get_product(name: str) -> Optional[Dict[str, Any]]:
    return PRODUCT_CATALOG.get(name)

def get_segment(name: str) -> Optional[Dict[str, Any]]:
    return CUSTOMER_SEGMENTS.get(name)

def get_region(name: str) -> Optional[Dict[str, Any]]:
    return REGIONS.get(name)

def get_competitor(name: str) -> Optional[Dict[str, Any]]:
    return COMPETITORS.get(name)


__all__: List[str] = [
    "APP_TITLE", "APP_VERSION", "CURRENT_DATE",
    "SCRAPING_CONFIG", "COMPETITOR_WEBSITES", "LINKEDIN_CONFIG", "NEWS_SOURCES",
    "CSV_UPLOAD_SPEC", "PREPROCESSING_OPTIONS",
    "COMPETITORS", "CUSTOMER_SEGMENTS", "PRODUCT_CATALOG", "REGIONS",
    "CHANNEL_LIST", "CHANNEL_PRICE_ADJ", "CHANNEL_WEIGHTS",
    "FORECAST_CONFIG", "MARGIN_WATERFALL_BUCKETS", "MONTHLY_SEASONALITY",
    "PRICING_STRATEGIES", "SCORING_WEIGHTS",
    "DATA_COLLECTION_GUIDANCE",
    "COMPETITOR_COLORS", "COMPETITOR_NAMES", "PRODUCT_NAMES",
    "PRODUCT_SEGMENTS", "SEGMENT_COLORS",
    "ACCENT", "ACCENT2", "ACCENT3", "BORDER", "BORDER2",
    "CARD_BG", "CARD_BG2", "CYAN", "DARK_BG", "GREEN",
    "MUTED", "ORANGE", "PLOTLY_DARK", "PURPLE", "RED",
    "SIDEBAR_BG", "TEXT", "YELLOW",
    "NAV_GROUPS", "NAV_INDEX", "NAV_OPTIONS",
    "validate_competitor", "validate_product", "validate_region",
    "validate_segment", "validate_strategy",
    "get_competitor", "get_product", "get_region", "get_segment",
]
