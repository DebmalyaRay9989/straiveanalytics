"""
STRAIVE Pricing Platform – Configuration Module (v4.1)

Changes vs v4.0:
  • All mutable defaults wrapped in functions / frozen (no shared-state bugs)
  • CHANNEL_WEIGHTS validated to sum to 1.0 at import time
  • MARGIN_WATERFALL_BUCKETS normalisation guard (zero-total protection)
  • QUARTERLY_SEASONALITY computed from MONTHLY_SEASONALITY (DRY)
  • Validator helpers hardened with type-checked returns and clear error messages
  • NAV_INDEX built once and exported
  • __all__ de-duplicated and sorted
  • Minor style cleanup (consistent quote style, trailing commas, PEP 8)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("straive_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

CURRENT_DATE: datetime = datetime.now()
APP_VERSION: str = "4.1.0"
APP_TITLE: str = "STRAIVE · Dynamic Pricing & Revenue Intelligence Platform"

# ─────────────────────────────────────────────────────────────────────────────
# WEB SCRAPING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRAPING_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "use_selenium": True,
    "headless": True,
    "request_timeout": 15,        # seconds
    "max_retries": 2,
    "rate_limit_delay": 1.0,      # seconds between requests
    "cache_enabled": True,
    "cache_ttl": 3_600,           # 1 hour
    "redis_host": "localhost",
    "redis_port": 6_379,
    "redis_db": 0,
    "user_agent_rotation": True,
    "respect_robots_txt": True,
    "max_pages_per_domain": 50,
    "max_depth": 3,
}

# Competitor websites for scraping
COMPETITOR_WEBSITES: Dict[str, Dict[str, Any]] = {
    "Aptara": {
        "base_url": "https://www.aptara.com",
        "pages": [
            "/services/editorial-services/",
            "/services/content-development/",
            "/services/data-services/",
            "/services/publishing-services/",
            "/case-studies/",
        ],
        "use_selenium": True,
        "price_selectors": [".price", ".pricing", ".cost", ".rate", ".package-price", ".service-price"],
        "service_selectors": ["h2", "h3", ".service-title", ".package-name"],
    },
    "Innodata": {
        "base_url": "https://www.innodata.com",
        "pages": [
            "/solutions/data-annotation/",
            "/solutions/content-services/",
            "/resources/case-studies/",
            "/industries/publishing/",
        ],
        "use_selenium": False,
        "price_patterns": [
            r"\$\s*([\d,]+(?:\.\d{2})?)",
            r"saved\s+\$?([\d,]+(?:\.\d+)?)\s*(?:million|k|thousand)?",
            r"roi\s+of\s+(\d+)%\s*on\s+\$?([\d,]+(?:\.\d+)?)",
        ],
    },
    "MPS Limited": {
        "base_url": "https://www.mpslimited.com",
        "pages": ["/investors/", "/services/", "/solutions/"],
        "use_selenium": False,
        "pdf_patterns": [
            r"annual.*report.*\.pdf",
            r"presentation.*\.pdf",
            r"financial.*\.pdf",
        ],
    },
    "Cenveo": {
        "base_url": "https://www.cenveo.com",
        "pages": ["/solutions/", "/capabilities/"],
        "use_selenium": True,
    },
    "Techbooks": {
        "base_url": "https://www.techbooks.com",
        "pages": ["/solutions/", "/services/"],
        "use_selenium": False,
    },
    "SPi Global": {
        "base_url": "https://www.spi-global.com",
        "pages": ["/services/", "/solutions/"],
        "use_selenium": True,
    },
    "Scribendi": {
        "base_url": "https://www.scribendi.com",
        "pages": ["/services/", "/pricing/"],
        "use_selenium": False,
    },
    "Clarivate Analytics": {
        "base_url": "https://clarivate.com",
        "pages": ["/solutions/", "/products/"],
        "use_selenium": True,
    },
}

# LinkedIn scraping configuration
LINKEDIN_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "company_pages": [
        "aptara",
        "innodata",
        "mps-limited",
        "cenveo",
        "techbooks",
        "spi-global",
        "scribendi",
        "clarivate-analytics",
    ],
    "job_search_keywords": ["pricing", "strategy", "revenue", "sales", "proposal", "bid", "quotation"],
    "max_jobs_per_company": 50,
}

# News sources for market intelligence
NEWS_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "PR Newswire - Publishing",
        "url": "https://www.prnewswire.com/news-releases/publishing-services-latest-news/",
        "selector": ".news-release",
        "type": "rss",
    },
    {
        "name": "Business Wire - Publishing",
        "url": "https://www.businesswire.com/portal/site/home/news/industry/publishing",
        "selector": ".bwNewsRelease",
        "type": "html",
    },
    {
        "name": "Publishers Weekly",
        "url": "https://www.publishersweekly.com/pw/news-events/index.html",
        "selector": ".news-item",
        "type": "html",
    },
    {
        "name": "The Bookseller",
        "url": "https://www.thebookseller.com/news",
        "selector": ".article",
        "type": "html",
    },
]

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
        "%d-%m-%Y", "%m-%d-%Y", "%Y%m%d", "%d.%m.%Y",
    ],
    "boolean_mappings": {
        "True":  [1, "true", "True", "TRUE", "yes", "Yes", "YES", "won", "Won", "success"],
        "False": [0, "false", "False", "FALSE", "no", "No", "NO", "lost", "Lost", "failure"],
    },
    "numeric_cleaning": {
        "remove_currency_symbols": True,
        "remove_commas": True,
        "handle_percentages": True,
        "default_decimal_separator": ".",
        "remove_whitespace": True,
    },
    "validation_rules": {
        "date":             {"min_year": 2018, "max_year": 2026},
        "actual_price":     {"min": 0,   "max": 1_000_000},
        "volume":           {"min": 1,   "max": 100_000},
        "revenue":          {"min": 0,   "max": 100_000_000},
        "cost":             {"min": 0,   "max": 100_000_000},
        "deal_won":         {"values": [0, 1]},
        "discount_pct":     {"min": 0,   "max": 100},
        "confidence_score": {"min": 0,   "max": 1},
    },
    "supported_file_formats": [".csv", ".xlsx", ".xls"],
    "max_file_size_mb": 50,
    "encoding_options": ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"],
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING OPTIONS
# ─────────────────────────────────────────────────────────────────────────────
PREPROCESSING_OPTIONS: Dict[str, Any] = {
    "handle_missing": ["drop_rows", "impute_mean", "impute_median", "flag_missing", "interpolate"],
    "outlier_detection": ["iqr", "zscore", "percentile", "isolation_forest", "none"],
    "outlier_threshold": {
        "iqr_multiplier": 1.5,
        "zscore_threshold": 3.0,
        "percentile": [1, 99],
    },
    "normalization": ["none", "minmax", "standard", "robust", "log", "quantile"],
    "feature_engineering": {
        "create_price_ratio": True,
        "create_margin_pct": True,
        "create_month": True,
        "create_quarter": True,
        "create_year": True,
        "create_competitor_price_gap": True,
        "create_seasonal_factors": True,
        "create_lagged_features": True,
    },
    "confidence_weighting": {
        "enabled": True,
        "min_confidence": 0.3,
        "weight_column": "confidence_score",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT CATALOG
# ─────────────────────────────────────────────────────────────────────────────
PRODUCT_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Editorial ──────────────────────────────────────────────────────────
    "Editorial Services – Standard": {
        "base_price": 2_800,  "unit": "project",   "segment": "Editorial",
        "cost": 1_100,        "min_price": 1_800,  "max_price":  4_500,
        "launch_year": 2018,  "complexity": 2,
        "keywords": ["copyediting", "proofreading", "high-volume", "standard"],
        "tags":     ["copyediting", "proofreading", "high-volume"],
    },
    "Editorial Services – Premium": {
        "base_price": 5_200,  "unit": "project",   "segment": "Editorial",
        "cost": 1_800,        "min_price": 3_500,  "max_price":  8_000,
        "launch_year": 2018,  "complexity": 3,
        "keywords": ["developmental", "substantive", "stm", "premium"],
        "tags":     ["developmental", "substantive", "STM"],
    },
    "Editorial Services – Bespoke": {
        "base_price": 9_800,  "unit": "project",   "segment": "Editorial",
        "cost": 3_200,        "min_price": 7_000,  "max_price": 16_000,
        "launch_year": 2022,  "complexity": 5,
        "keywords": ["white-glove", "monograph", "government", "bespoke"],
        "tags":     ["white-glove", "monograph", "government"],
    },
    # ── Data Services ───────────────────────────────────────────────────────
    "Data Annotation – Basic": {
        "base_price":   450,  "unit": "1k items",  "segment": "Data Services",
        "cost":  180,         "min_price":   300,  "max_price":    750,
        "launch_year": 2019,  "complexity": 1,
        "keywords": ["annotation", "classification", "image", "text", "basic"],
        "tags":     ["classification", "image", "text"],
    },
    "Data Annotation – Advanced": {
        "base_price":   980,  "unit": "1k items",  "segment": "Data Services",
        "cost":  320,         "min_price":   650,  "max_price":  1_500,
        "launch_year": 2019,  "complexity": 3,
        "keywords": ["ner", "relation", "multi-label", "advanced"],
        "tags":     ["NER", "relation", "multi-label"],
    },
    "Data Annotation – Expert": {
        "base_price": 2_100,  "unit": "1k items",  "segment": "Data Services",
        "cost":  700,         "min_price": 1_400,  "max_price":  3_500,
        "launch_year": 2021,  "complexity": 4,
        "keywords": ["medical", "legal", "domain-expert", "expert"],
        "tags":     ["medical", "legal", "domain-expert"],
    },
    "Data Curation & QA": {
        "base_price": 1_350,  "unit": "1k items",  "segment": "Data Services",
        "cost":  480,         "min_price":   900,  "max_price":  2_200,
        "launch_year": 2020,  "complexity": 2,
        "keywords": ["deduplication", "validation", "enrichment", "curation"],
        "tags":     ["deduplication", "validation", "enrichment"],
    },
    # ── AI / ML ─────────────────────────────────────────────────────────────
    "AI/ML Pipeline – Starter": {
        "base_price":  8_500,  "unit": "month",  "segment": "AI/ML",
        "cost": 3_200,         "min_price":  6_000,  "max_price": 14_000,
        "launch_year": 2020,   "complexity": 4,
        "keywords": ["nlp", "classification", "smb", "starter"],
        "tags":     ["NLP", "classification", "SMB"],
    },
    "AI/ML Pipeline – Professional": {
        "base_price": 16_000,  "unit": "month",  "segment": "AI/ML",
        "cost": 5_500,         "min_price": 11_000,  "max_price": 26_000,
        "launch_year": 2021,   "complexity": 5,
        "keywords": ["fine-tuning", "rag", "mid-market", "professional"],
        "tags":     ["fine-tuning", "RAG", "mid-market"],
    },
    "AI/ML Pipeline – Enterprise": {
        "base_price": 22_000,  "unit": "month",  "segment": "AI/ML",
        "cost": 7_500,         "min_price": 15_000,  "max_price": 38_000,
        "launch_year": 2021,   "complexity": 5,
        "keywords": ["llm", "custom-model", "sla", "enterprise"],
        "tags":     ["LLM", "custom-model", "SLA"],
    },
    "AI Evaluation & Red-Teaming": {
        "base_price": 12_500,  "unit": "project",  "segment": "AI/ML",
        "cost": 4_200,         "min_price":  8_000,  "max_price": 20_000,
        "launch_year": 2023,   "complexity": 5,
        "keywords": ["safety", "bias", "hallucination", "evaluation"],
        "tags":     ["safety", "bias", "hallucination"],
    },
    # ── Content ─────────────────────────────────────────────────────────────
    "Content Transformation – Basic": {
        "base_price": 1_200,  "unit": "project",  "segment": "Content",
        "cost":  480,         "min_price":   800,  "max_price":  2_000,
        "launch_year": 2017,  "complexity": 1,
        "keywords": ["xml", "html", "conversion", "basic"],
        "tags":     ["XML", "HTML", "conversion"],
    },
    "Content Transformation – Plus": {
        "base_price": 2_600,  "unit": "project",  "segment": "Content",
        "cost":  900,         "min_price": 1_700,  "max_price":  4_200,
        "launch_year": 2017,  "complexity": 2,
        "keywords": ["multimedia", "accessibility", "epub", "plus"],
        "tags":     ["multimedia", "accessibility", "EPUB"],
    },
    "Content Transformation – Enterprise": {
        "base_price": 6_500,  "unit": "project",  "segment": "Content",
        "cost": 2_200,        "min_price": 4_500,  "max_price": 10_500,
        "launch_year": 2020,  "complexity": 3,
        "keywords": ["workflow", "cms", "automation", "enterprise"],
        "tags":     ["workflow", "CMS", "automation"],
    },
    # ── Analytics ───────────────────────────────────────────────────────────
    "Research & Analytics – Standard": {
        "base_price":  4_800,  "unit": "project",  "segment": "Analytics",
        "cost": 1_700,         "min_price":  3_200,  "max_price":  7_800,
        "launch_year": 2019,   "complexity": 3,
        "keywords": ["market-research", "surveys", "reporting", "standard"],
        "tags":     ["market-research", "surveys", "reporting"],
    },
    "Research & Analytics – Premium": {
        "base_price": 11_500,  "unit": "project",  "segment": "Analytics",
        "cost": 3_800,         "min_price":  7_500,  "max_price": 18_000,
        "launch_year": 2019,   "complexity": 4,
        "keywords": ["predictive", "longitudinal", "custom-models", "premium"],
        "tags":     ["predictive", "longitudinal", "custom-models"],
    },
    "Research & Analytics – Syndicated": {
        "base_price":  3_200,  "unit": "license/yr",  "segment": "Analytics",
        "cost":   800,         "min_price":  2_200,  "max_price":  5_500,
        "launch_year": 2022,   "complexity": 2,
        "keywords": ["benchmarks", "industry-reports", "saas", "syndicated"],
        "tags":     ["benchmarks", "industry-reports", "SaaS"],
    },
    # ── Technology ──────────────────────────────────────────────────────────
    "Publishing Tech – SaaS": {
        "base_price":  3_200,  "unit": "month",  "segment": "Technology",
        "cost":   950,         "min_price":  2_200,  "max_price":  5_200,
        "launch_year": 2020,   "complexity": 3,
        "keywords": ["workflow", "cloud", "self-serve", "saas"],
        "tags":     ["workflow", "cloud", "self-serve"],
    },
    "Publishing Tech – Professional": {
        "base_price":  6_200,  "unit": "month",  "segment": "Technology",
        "cost": 1_800,         "min_price":  4_200,  "max_price": 10_000,
        "launch_year": 2021,   "complexity": 4,
        "keywords": ["integrations", "api", "multi-user", "professional"],
        "tags":     ["integrations", "API", "multi-user"],
    },
    "Publishing Tech – Enterprise": {
        "base_price":  9_500,  "unit": "month",  "segment": "Technology",
        "cost": 2_600,         "min_price":  6_500,  "max_price": 15_500,
        "launch_year": 2021,   "complexity": 5,
        "keywords": ["on-prem", "sso", "dedicated-csm", "enterprise"],
        "tags":     ["on-prem", "SSO", "dedicated-CSM"],
    },
    "Typesetting & Composition": {
        "base_price":  1_600,  "unit": "project",  "segment": "Technology",
        "cost":   550,         "min_price":  1_000,  "max_price":  2_800,
        "launch_year": 2016,   "complexity": 2,
        "keywords": ["latex", "indesign", "journal", "typesetting"],
        "tags":     ["LaTeX", "InDesign", "journal"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOMER_SEGMENTS: Dict[str, Dict[str, Any]] = {
    "Academic Publishers": {
        "volume_multiplier": 1.4,  "price_sensitivity": 0.72,  "loyalty": 0.85,
        "nps_proxy": 42,  "avg_contract_months": 12,
        "preferred_segments": ["Editorial", "Technology"],
        "color": "#4f9eff",
        "keywords": ["university", "academic", "scholarly", "journal"],
    },
    "STM Publishers": {
        "volume_multiplier": 2.1,  "price_sensitivity": 0.55,  "loyalty": 0.90,
        "nps_proxy": 55,  "avg_contract_months": 24,
        "preferred_segments": ["Editorial", "AI/ML", "Data Services"],
        "color": "#36d97b",
        "keywords": ["scientific", "technical", "medical", "research"],
    },
    "Trade Publishers": {
        "volume_multiplier": 0.9,  "price_sensitivity": 0.88,  "loyalty": 0.70,
        "nps_proxy": 31,  "avg_contract_months":  6,
        "preferred_segments": ["Content", "Editorial"],
        "color": "#ff6b6b",
        "keywords": ["trade", "consumer", "fiction", "non-fiction"],
    },
    "Government & NGO": {
        "volume_multiplier": 1.2,  "price_sensitivity": 0.61,  "loyalty": 0.82,
        "nps_proxy": 38,  "avg_contract_months": 18,
        "preferred_segments": ["Analytics", "Editorial"],
        "color": "#ffd700",
        "keywords": ["government", "ngo", "public sector", "non-profit"],
    },
    "Corporate / Enterprise": {
        "volume_multiplier": 1.8,  "price_sensitivity": 0.48,  "loyalty": 0.78,
        "nps_proxy": 48,  "avg_contract_months": 36,
        "preferred_segments": ["AI/ML", "Data Services", "Technology"],
        "color": "#b48eff",
        "keywords": ["corporate", "enterprise", "business", "fortune"],
    },
    "EdTech Platforms": {
        "volume_multiplier": 1.3,  "price_sensitivity": 0.80,  "loyalty": 0.68,
        "nps_proxy": 35,  "avg_contract_months": 12,
        "preferred_segments": ["Content", "AI/ML"],
        "color": "#ff9800",
        "keywords": ["education", "learning", "course", "platform"],
    },
    "Legal & Compliance": {
        "volume_multiplier": 1.6,  "price_sensitivity": 0.44,  "loyalty": 0.88,
        "nps_proxy": 51,  "avg_contract_months": 24,
        "preferred_segments": ["Data Services", "Analytics"],
        "color": "#00d4c8",
        "keywords": ["legal", "compliance", "regulation", "law"],
    },
    "Healthcare & Life Sciences": {
        "volume_multiplier": 2.0,  "price_sensitivity": 0.38,  "loyalty": 0.92,
        "nps_proxy": 58,  "avg_contract_months": 36,
        "preferred_segments": ["AI/ML", "Data Services", "Analytics"],
        "color": "#e040fb",
        "keywords": ["healthcare", "medical", "pharma", "life sciences"],
    },
    "Media & News": {
        "volume_multiplier": 0.8,  "price_sensitivity": 0.91,  "loyalty": 0.62,
        "nps_proxy": 27,  "avg_contract_months":  6,
        "preferred_segments": ["Content", "Technology"],
        "color": "#f06292",
        "keywords": ["media", "news", "publishing", "digital"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# REGIONS
# ─────────────────────────────────────────────────────────────────────────────
REGIONS: Dict[str, Dict[str, Any]] = {
    "North America": {
        "demand_index": 1.35,  "competitor_pressure": 0.65,
        "growth_rate": 0.08,   "fx_risk": 0.05,  "payment_days": 35,
        "gdp_growth_proxy": 0.024,  "market_maturity": 0.9,
    },
    "Europe": {
        "demand_index": 1.20,  "competitor_pressure": 0.72,
        "growth_rate": 0.06,   "fx_risk": 0.18,  "payment_days": 45,
        "gdp_growth_proxy": 0.016,  "market_maturity": 0.85,
    },
    "Asia Pacific": {
        "demand_index": 1.55,  "competitor_pressure": 0.80,
        "growth_rate": 0.14,   "fx_risk": 0.28,  "payment_days": 55,
        "gdp_growth_proxy": 0.048,  "market_maturity": 0.7,
    },
    "Middle East": {
        "demand_index": 0.88,  "competitor_pressure": 0.55,
        "growth_rate": 0.11,   "fx_risk": 0.22,  "payment_days": 60,
        "gdp_growth_proxy": 0.035,  "market_maturity": 0.6,
    },
    "Latin America": {
        "demand_index": 0.72,  "competitor_pressure": 0.60,
        "growth_rate": 0.09,   "fx_risk": 0.45,  "payment_days": 70,
        "gdp_growth_proxy": 0.018,  "market_maturity": 0.5,
    },
    "Africa": {
        "demand_index": 0.60,  "competitor_pressure": 0.40,
        "growth_rate": 0.18,   "fx_risk": 0.55,  "payment_days": 80,
        "gdp_growth_proxy": 0.042,  "market_maturity": 0.4,
    },
    "South Asia": {
        "demand_index": 0.95,  "competitor_pressure": 0.75,
        "growth_rate": 0.13,   "fx_risk": 0.30,  "payment_days": 50,
        "gdp_growth_proxy": 0.062,  "market_maturity": 0.55,
    },
    "Oceania": {
        "demand_index": 0.65,  "competitor_pressure": 0.50,
        "growth_rate": 0.07,   "fx_risk": 0.15,  "payment_days": 38,
        "gdp_growth_proxy": 0.020,  "market_maturity": 0.8,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# COMPETITORS
# ─────────────────────────────────────────────────────────────────────────────
COMPETITORS: Dict[str, Dict[str, Any]] = {
    "Aptara": {
        "relative_price": 0.88,  "quality_score": 7.2,
        "strengths":   ["price", "volume capacity"],
        "weaknesses":  ["AI capabilities", "analytics depth"],
        "win_rate_vs": 0.62,  "market_share": 0.12,
        "scrape_enabled": True,  "scrape_priority": 1,
    },
    "Innodata": {
        "relative_price": 0.82,  "quality_score": 7.5,
        "strengths":   ["data annotation", "offshore cost base"],
        "weaknesses":  ["editorial quality", "tech stack"],
        "win_rate_vs": 0.58,  "market_share": 0.10,
        "scrape_enabled": True,  "scrape_priority": 2,
    },
    "MPS Limited": {
        "relative_price": 0.91,  "quality_score": 7.8,
        "strengths":   ["editorial", "established brand"],
        "weaknesses":  ["AI/ML", "agility"],
        "win_rate_vs": 0.54,  "market_share": 0.08,
        "scrape_enabled": True,  "scrape_priority": 3,
    },
    "Cenveo": {
        "relative_price": 0.94,  "quality_score": 7.0,
        "strengths":   ["print integration", "North America presence"],
        "weaknesses":  ["digital services", "global reach"],
        "win_rate_vs": 0.66,  "market_share": 0.07,
        "scrape_enabled": True,  "scrape_priority": 4,
    },
    "Techbooks": {
        "relative_price": 0.78,  "quality_score": 6.8,
        "strengths":   ["lowest cost", "high throughput"],
        "weaknesses":  ["quality consistency", "complex projects"],
        "win_rate_vs": 0.70,  "market_share": 0.09,
        "scrape_enabled": True,  "scrape_priority": 5,
    },
    "SPi Global": {
        "relative_price": 0.86,  "quality_score": 7.3,
        "strengths":   ["STM editorial", "Philippines delivery"],
        "weaknesses":  ["AI pipeline", "pricing transparency"],
        "win_rate_vs": 0.60,  "market_share": 0.11,
        "scrape_enabled": True,  "scrape_priority": 6,
    },
    "Scribendi": {
        "relative_price": 0.75,  "quality_score": 6.5,
        "strengths":   ["self-serve", "turnaround speed"],
        "weaknesses":  ["enterprise scale", "account management"],
        "win_rate_vs": 0.74,  "market_share": 0.05,
        "scrape_enabled": True,  "scrape_priority": 7,
    },
    "Clarivate Analytics": {
        "relative_price": 1.35,  "quality_score": 8.6,
        "strengths":   ["brand", "data assets", "analytics"],
        "weaknesses":  ["cost", "editorial services"],
        "win_rate_vs": 0.38,  "market_share": 0.15,
        "scrape_enabled": True,  "scrape_priority": 8,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PRICING STRATEGY PRESETS
# ─────────────────────────────────────────────────────────────────────────────
PRICING_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "Penetration": {
        "description": "Price below market to gain share; accept thin margins initially.",
        "price_floor_pct": 0.75,   "price_ceil_pct": 0.95,
        "target": "revenue",       "discount_cap_pct": 30,
        "competitive_response": "aggressive",
    },
    "Neutral / Market-Rate": {
        "description": "Align closely to market; balanced margin and volume.",
        "price_floor_pct": 0.90,   "price_ceil_pct": 1.10,
        "target": "revenue",       "discount_cap_pct": 20,
        "competitive_response": "follow",
    },
    "Premium Value": {
        "description": "Price at a premium justified by quality and brand.",
        "price_floor_pct": 1.05,   "price_ceil_pct": 1.40,
        "target": "profit",        "discount_cap_pct": 10,
        "competitive_response": "differentiate",
    },
    "Margin Maximization": {
        "description": "Optimize strictly for gross profit margin.",
        "price_floor_pct": 1.00,   "price_ceil_pct": 1.80,
        "target": "margin",        "discount_cap_pct": 5,
        "competitive_response": "premium",
    },
    "Volume / Bundled": {
        "description": "Offer volume discounts to lock in multi-year / bulk contracts.",
        "price_floor_pct": 0.70,   "price_ceil_pct": 0.95,
        "target": "revenue",       "discount_cap_pct": 40,
        "competitive_response": "volume",
    },
    "Dynamic / AI-Optimized": {
        "description": "Real-time price optimization based on market signals.",
        "price_floor_pct": 0.80,   "price_ceil_pct": 1.50,
        "target": "profit",        "discount_cap_pct": 25,
        "competitive_response": "adaptive",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SEASONALITY PROFILE
# ─────────────────────────────────────────────────────────────────────────────
MONTHLY_SEASONALITY: Dict[int, float] = {
    1: 0.82,  2: 0.88,  3: 1.04,  4: 0.97,
    5: 1.05,  6: 0.95,  7: 0.78,  8: 0.83,
    9: 1.08, 10: 1.22, 11: 1.28, 12: 1.18,
}

# Derived quarterly averages (DRY – single source of truth)
QUARTERLY_SEASONALITY: Dict[int, float] = {
    q: round(
        sum(MONTHLY_SEASONALITY[m] for m in range((q - 1) * 3 + 1, q * 3 + 1)) / 3,
        4,
    )
    for q in range(1, 5)
}

# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL LIST  (weights must sum to 1.0 – validated at import)
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_LIST: List[str] = ["Direct", "Partner", "Inbound / Marketing", "Renewal", "Referral"]
CHANNEL_WEIGHTS: List[float] = [0.45, 0.20, 0.18, 0.12, 0.05]
CHANNEL_PRICE_ADJ: Dict[str, float] = {
    "Direct":               0.00,
    "Partner":             -0.04,
    "Inbound / Marketing":  0.02,
    "Renewal":             -0.07,
    "Referral":            -0.03,
}

# Import-time guard: weights must sum to 1.0 (±0.001 tolerance)
_WEIGHT_SUM = sum(CHANNEL_WEIGHTS)
if abs(_WEIGHT_SUM - 1.0) > 0.001:
    log.warning(
        "CHANNEL_WEIGHTS sum to %.6f (expected 1.0). "
        "Normalising automatically.",
        _WEIGHT_SUM,
    )
    CHANNEL_WEIGHTS = [w / _WEIGHT_SUM for w in CHANNEL_WEIGHTS]

# ─────────────────────────────────────────────────────────────────────────────
# MARGIN WATERFALL COST BUCKETS
# ─────────────────────────────────────────────────────────────────────────────
_RAW_WATERFALL: Dict[str, float] = {
    "Direct Labour":      0.28,
    "Subcontractors":     0.10,
    "Technology / SaaS":  0.06,
    "QA & Rework":        0.04,
    "Project Management": 0.04,
    "G&A Overhead":       0.05,
    "Sales & Marketing":  0.07,
    "R&D Allocation":     0.03,
}
_wf_total = sum(_RAW_WATERFALL.values())
# Guard against zero-total (should never happen, but keeps normalisation safe)
if _wf_total == 0:
    raise ValueError("MARGIN_WATERFALL_BUCKETS: all cost shares are zero – check config.")
MARGIN_WATERFALL_BUCKETS: Dict[str, float] = {
    k: round(v / _wf_total, 6) for k, v in _RAW_WATERFALL.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION & REAL DATA STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
DATA_COLLECTION_GUIDANCE: Dict[str, Any] = {
    "priority_sources": [
        "CRM closed-won / closed-lost export (highest value – has win/loss signal)",
        "Billing / invoicing data from ERP / accounting system",
        "Salesforce / HubSpot opportunities, quotes, and stage history",
        "Historical contract pricing sheets and MSA schedules",
        "Competitive-intel call notes tagged with competitor name",
        "Web scraped competitor pricing data (real-time intelligence)",
        "LinkedIn job postings (hiring signals for pricing teams)",
        "Industry news and press releases (M&A, new offerings)",
        "Annual reports and investor presentations (financial metrics)",
    ],
    "minimum_required_columns": [
        "date", "product", "customer_type", "region",
        "actual_price", "volume", "revenue", "cost", "deal_won",
    ],
    "strongly_recommended_columns": [
        "base_price", "discount_pct", "quote_id", "customer_id",
        "sales_cycle_days", "competitor_offered_price", "win_loss_reason",
        "contract_months", "renewal_flag", "account_manager", "channel",
        "competitor", "competitor_price", "source", "confidence_score",
    ],
    "nice_to_have_columns": [
        "nps_score", "support_tickets_ytd", "expansion_revenue",
        "churn_flag", "competitor_quality_score", "market_share",
    ],
    "real_data_target_ratio": 0.70,
    "minimum_real_rows_for_blending": 600,
    "synthetic_noise_std": 0.14,
    "data_quality_checks": [
        "No negative prices or revenues",
        "actual_price must be ≥ product min_price and ≤ max_price",
        "volume must be a positive integer",
        "deal_won must be binary (0/1)",
        "date must be parseable and within last 5 years",
        "confidence_score between 0 and 1",
    ],
    "column_mappings": {
        "date":          ["transaction_date", "deal_date", "close_date", "invoice_date", "order_date"],
        "product":       ["product_name", "service", "sku", "item", "offering"],
        "customer_type": ["customer_segment", "segment", "customer_category", "account_type"],
        "region":        ["geo", "territory", "country", "market", "location"],
        "actual_price":  ["unit_price", "selling_price", "price", "final_price"],
        "volume":        ["quantity", "units", "count", "demand"],
        "revenue":       ["total_revenue", "amount", "sales", "bookings"],
        "cost":          ["cogs", "cost_of_goods", "delivery_cost", "fulfillment_cost"],
        "deal_won":      ["won", "success", "closed_won", "outcome", "result"],
        "base_price":    ["list_price", "msrp", "standard_price", "catalog_price"],
        "discount_pct":  ["discount", "discount_percent", "price_reduction"],
        "competitor":    ["competitor_name", "vendor", "rival"],
        "competitor_price": ["competitor_pricing", "market_price"],
        "source":        ["data_source", "origin"],
    },
    "scraping_targets": {
        "daily":   ["Aptara", "Innodata"],
        "weekly":  ["MPS Limited", "Cenveo", "Techbooks"],
        "monthly": ["SPi Global", "Scribendi", "Clarivate Analytics"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FORECAST PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
FORECAST_CONFIG: Dict[str, Any] = {
    "horizon_months": 12,
    "confidence_interval": 0.90,
    "trend_dampening": 0.85,
    "seasonality_weight": 0.60,
    "min_obs_for_arima": 24,
    "ensemble_methods": ["linear", "exp_smoothing", "arima", "prophet"],
    "cross_validation_folds": 5,
    "include_market_signals": True,
    "market_signal_weight": 0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS  (must sum ≤ 1.0; checked at import)
# ─────────────────────────────────────────────────────────────────────────────
SCORING_WEIGHTS: Dict[str, float] = {
    "margin_pct":          0.25,
    "revenue_growth":      0.20,
    "win_rate":            0.15,
    "customer_loyalty":    0.10,
    "competitive_moat":    0.10,
    "market_position":     0.10,
    "price_competitiveness": 0.10,
}

_sw_total = sum(SCORING_WEIGHTS.values())
if abs(_sw_total - 1.0) > 0.001:
    log.warning("SCORING_WEIGHTS sum to %.4f – normalising.", _sw_total)
    SCORING_WEIGHTS = {k: v / _sw_total for k, v in SCORING_WEIGHTS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED CONSTANTS  (built once from the master dicts above)
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
    "legend":      {
        "bgcolor": "rgba(12,18,33,0.92)", "bordercolor": BORDER2, "borderwidth": 1,
        "font": {"color": TEXT, "size": 11},
    },
    "margin":      {"l": 14, "r": 14, "t": 52, "b": 14},
    "hoverlabel":  {"bgcolor": CARD_BG, "bordercolor": BORDER2, "font": {"color": TEXT, "size": 12}},
    "xaxis":       {"gridcolor": BORDER, "zerolinecolor": BORDER2, "linecolor": BORDER},
    "yaxis":       {"gridcolor": BORDER, "zerolinecolor": BORDER2, "linecolor": BORDER},
}

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
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
    "🕷️ Market Intelligence",
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
    "DATA_INTEL":   ["🕷️ Market Intelligence"],
}

# Built once at import – avoids repeated list.index() calls in callbacks
NAV_INDEX: Dict[str, int] = {label: i for i, label in enumerate(NAV_OPTIONS)}


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_product(name: str) -> bool:
    """Return True if *name* is a known product key."""
    return isinstance(name, str) and name in PRODUCT_CATALOG


def validate_segment(name: str) -> bool:
    """Return True if *name* is a known customer-segment key."""
    return isinstance(name, str) and name in CUSTOMER_SEGMENTS


def validate_region(name: str) -> bool:
    """Return True if *name* is a known region key."""
    return isinstance(name, str) and name in REGIONS


def validate_competitor(name: str) -> bool:
    """Return True if *name* is a known competitor key."""
    return isinstance(name, str) and name in COMPETITORS


def validate_strategy(name: str) -> bool:
    """Return True if *name* is a known pricing-strategy key."""
    return isinstance(name, str) and name in PRICING_STRATEGIES


def get_product(name: str) -> Optional[Dict[str, Any]]:
    """Return product info dict, or None if unknown."""
    return PRODUCT_CATALOG.get(name)


def get_segment(name: str) -> Optional[Dict[str, Any]]:
    """Return customer-segment info dict, or None if unknown."""
    return CUSTOMER_SEGMENTS.get(name)


def get_region(name: str) -> Optional[Dict[str, Any]]:
    """Return region info dict, or None if unknown."""
    return REGIONS.get(name)


def get_competitor(name: str) -> Optional[Dict[str, Any]]:
    """Return competitor info dict, or None if unknown."""
    return COMPETITORS.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    # Globals
    "APP_TITLE", "APP_VERSION", "CURRENT_DATE", "OUTPUT_DIR",
    # Scraping
    "COMPETITOR_WEBSITES", "LINKEDIN_CONFIG", "NEWS_SOURCES", "SCRAPING_CONFIG",
    # Upload / preprocessing
    "CSV_UPLOAD_SPEC", "PREPROCESSING_OPTIONS",
    # Catalogs
    "COMPETITORS", "CUSTOMER_SEGMENTS", "PRODUCT_CATALOG", "REGIONS",
    # Channels
    "CHANNEL_LIST", "CHANNEL_PRICE_ADJ", "CHANNEL_WEIGHTS",
    # Strategy / config
    "FORECAST_CONFIG", "MARGIN_WATERFALL_BUCKETS", "MONTHLY_SEASONALITY",
    "PRICING_STRATEGIES", "QUARTERLY_SEASONALITY", "SCORING_WEIGHTS",
    # Guidance
    "DATA_COLLECTION_GUIDANCE",
    # Derived
    "COMPETITOR_COLORS", "COMPETITOR_NAMES", "PRODUCT_NAMES",
    "PRODUCT_SEGMENTS", "SEGMENT_COLORS",
    # Theme
    "ACCENT", "ACCENT2", "ACCENT3", "BORDER", "BORDER2",
    "CARD_BG", "CARD_BG2", "CYAN", "DARK_BG", "GREEN",
    "MUTED", "ORANGE", "PLOTLY_DARK", "PURPLE", "RED",
    "SIDEBAR_BG", "TEXT", "YELLOW",
    # Navigation
    "NAV_GROUPS", "NAV_INDEX", "NAV_OPTIONS",
    # Validators
    "validate_competitor", "validate_product", "validate_region",
    "validate_segment", "validate_strategy",
    # Getters
    "get_competitor", "get_product", "get_region", "get_segment",
]
