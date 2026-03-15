"""
STRAIVE Pricing Platform – Dash Application  (v4.1)

Improvements over v4.0
-----------------------
_parse_upload
  • Explicit file-size validation guard BEFORE constructing the DataFrame
    (previously the decoded bytes were read twice via a second split(",",1))
  • Size estimate uses exact len(decoded) rather than the base64-length
    approximation (3/4 scaling was correct but applied to the wrong object)

handle_file_upload
  • Size check moved to _parse_upload so the callback body is simpler
  • min_rows constant extracted to CSV_UPLOAD_SPEC["min_rows"] with 100 fallback
  • preview_df.iterrows() replaced with itertuples for speed on wide DataFrames

build_enhanced_model
  • Blended mode: competitor DataFrame concat only when columns overlap with
    the transaction schema (avoids silent NaN pollution)
  • feedback always returns a Dash component (never a plain string) so the
    Output type is consistent

_build_intelligence_pdf
  • _on_page nested function hoisted to module level (_pdf_on_page) so
    ReportLab can pickle it when building the document in a thread
  • COMPETITORS imported at module level (not inside callback)

render_tab (score_deal callback)
  • Validates price > 0 before computing margin; shows user-friendly error
    if price is missing or zero

export_results
  • JSON serialisation uses default=str to handle any non-serialisable values
    (e.g. numpy int64, datetime) instead of silently crashing

update_nav
  • NAV_INDEX from config used for O(1) lookup instead of list.index()

General
  • Logging format aligned with engine.py (%(asctime)s %(levelname)-8s)
  • start_scheduled_gathering guarded with SCRAPING_CONFIG flag so it can be
    enabled via config without editing source code
  • dcc.Download for template extracted to layout (was referenced before definition)
"""

from __future__ import annotations
import json
import base64
import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL, MATCH, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import threading
import asyncio
import logging

# ---------------------------------------------------------------------------
# Compatibility patch: ReportLab's internal C-extension passes
# `usedforsecurity=False` as a positional arg on Python >= 3.9 / OpenSSL 3,
# which raises "openssl_md5() takes at most 1 argument (2 given)".
# Wrapping hashlib.md5 to accept and silently drop that kwarg fixes it.
# Must be applied BEFORE any reportlab import.
# ---------------------------------------------------------------------------



import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



import hashlib as _hashlib

_original_md5 = _hashlib.md5

def _patched_md5(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return _original_md5(*args, **kwargs)

_hashlib.md5 = _patched_md5  # type: ignore[assignment]
# ---------------------------------------------------------------------------

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

from config import (
    PRODUCT_CATALOG, PRODUCT_SEGMENTS, SEGMENT_COLORS, COMPETITOR_COLORS,
    PLOTLY_DARK, NAV_OPTIONS, NAV_GROUPS, DARK_BG, CARD_BG, CARD_BG2, 
    SIDEBAR_BG, BORDER, BORDER2, ACCENT, ACCENT2, ACCENT3, GREEN, YELLOW, RED,
    PURPLE, CYAN, ORANGE, MUTED, CURRENT_DATE, DATA_COLLECTION_GUIDANCE,
    CUSTOMER_SEGMENTS, REGIONS, COMPETITORS, PRICING_STRATEGIES,
    MONTHLY_SEASONALITY, MARGIN_WATERFALL_BUCKETS, APP_TITLE, APP_VERSION,
    CSV_UPLOAD_SPEC, PREPROCESSING_OPTIONS, SCRAPING_CONFIG, COMPETITOR_WEBSITES,
    LINKEDIN_CONFIG, NEWS_SOURCES, COMPETITOR_NAMES,
)
from engine import (
    DataGenerator, ModelComparator, PricingOptimizer, SimulationEngine,
    CompetitiveAnalyzer, RevenueForecaster, DealScorer,
    MarginWaterfallBuilder, PortfolioScorer, MarketIntelligenceIntegrator,
)
from data_gathering import (  # New import
    DataGatheringOrchestrator, EnhancedDataGenerator, ScheduledDataGatherer,
    CompetitorPricePoint, MarketIntelligence, PricingSignal,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# SHARED UPLOAD PARSER
# ============================================================================
def _parse_upload(
    contents: str,
    filename: str,
) -> tuple[pd.DataFrame | None, str]:
    """
    Decode and parse a Dash-upload base64 payload.

    Returns (DataFrame, "") on success or (None, error_message) on failure.
    File-size validation is performed here so callers do not need to re-decode
    the payload.
    """
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
    except Exception as exc:
        return None, f"Could not decode upload payload: {exc}"

    # Size check using exact decoded byte length
    file_size_mb = len(decoded) / (1024 * 1024)
    max_size_mb  = CSV_UPLOAD_SPEC.get("max_file_size_mb", 50)
    if file_size_mb > max_size_mb:
        return None, (
            f"File too large: {file_size_mb:.1f} MB (max {max_size_mb} MB). "
            "Please reduce the file size and try again."
        )

    file_ext = (filename or "").lower().rsplit(".", 1)[-1]

    if file_ext == "csv":
        df: pd.DataFrame | None = None
        for enc in CSV_UPLOAD_SPEC.get("encoding_options", ["utf-8", "latin-1"]):
            try:
                df = pd.read_csv(io.StringIO(decoded.decode(enc)))
                break
            except Exception:
                continue
        if df is None:
            return None, "Could not decode CSV with any supported encoding."
    elif file_ext in ("xlsx", "xls"):
        try:
            df = pd.read_excel(io.BytesIO(decoded))
        except Exception as exc:
            return None, f"Could not read Excel file: {exc}"
    else:
        return None, f"Unsupported format: .{file_ext}. Use .csv or .xlsx"

    return df, ""

# ============================================================================
# INITIALIZE DATA GATHERING COMPONENTS
# ============================================================================
data_orchestrator = DataGatheringOrchestrator(
    use_redis=SCRAPING_CONFIG.get("cache_enabled", False),
    redis_host=SCRAPING_CONFIG.get("redis_host", "localhost"),
)
enhanced_generator = EnhancedDataGenerator(data_orchestrator)
market_integrator = MarketIntelligenceIntegrator()

# Start scheduled data gathering in background
def start_scheduled_gathering() -> None:
    """Start scheduled data gathering in a daemon background thread."""
    gatherer = ScheduledDataGatherer(data_orchestrator)

    async def _run() -> None:
        await gatherer.run_periodic_gathering(interval_hours=12)

    def _thread_target() -> None:
        asyncio.run(_run())

    thread = threading.Thread(target=_thread_target, daemon=True, name="data-gatherer")
    thread.start()
    logger.info("Started scheduled data gathering (every 12 hours)")


# Enable via SCRAPING_CONFIG["enable_scheduled_gathering"] = True in config.py
if SCRAPING_CONFIG.get("enable_scheduled_gathering", False):
    start_scheduled_gathering()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _numpy_default(obj: Any) -> Any:
    """JSON serialiser for numpy scalar types produced by sklearn/XGBoost models.

    Converts numpy int*/uint* → int, numpy float* → float, numpy bool_ → bool,
    numpy ndarray → list.  Falls back to str() for anything else so json.dumps
    never raises TypeError regardless of what the model returns.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _json_dumps(obj: Any) -> str:
    """json.dumps with numpy-aware default — use instead of bare json.dumps
    whenever serialising model outputs (elasticity dict, win_meta, etc.)."""
    return json.dumps(obj, default=_numpy_default)


def _fmt_date(value, fallback: str = "—", length: int = 10) -> str:
    """Safely format a timestamp value (str, datetime, or None) to a date string.

    Handles the three types that can arrive from the JSON store:
    • None / missing  → fallback
    • datetime object → isoformat()[:length]
    • str             → str[:length]  (already serialised)
    Any other type is converted via str() before slicing.
    """
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value.isoformat()[:length]
    try:
        return str(value)[:length]
    except Exception:
        return fallback


def _card(title: str, value: str, color: str = ACCENT, sub: str = "", tooltip: str = "") -> dbc.Col:
    """Create a KPI card with optional tooltip.

    dbc.Card v1.6 dropped the ``title`` kwarg, so the native browser tooltip
    is applied via a wrapping html.Div instead.
    """
    card = dbc.Card(
        dbc.CardBody([
            html.P(title, style={
                "color": MUTED, "fontSize": "0.72rem", "textTransform": "uppercase",
                "letterSpacing": "0.8px", "marginBottom": "0.3rem", "fontWeight": "500",
            }),
            html.H4(value, style={
                "color": color, "margin": 0, "fontFamily": "'Space Grotesk', sans-serif",
                "fontWeight": "700", "fontSize": "1.4rem",
            }),
            html.P(sub, style={"color": MUTED, "fontSize": "0.75rem", "margin": "0.2rem 0 0"}) if sub is not None and sub != "" else None,
        ], style={"padding": "1rem 1.2rem"}),
        style={
            "background": f"linear-gradient(135deg, {CARD_BG} 0%, #0f1a2e 100%)",
            "border": f"1px solid {BORDER2}",
            "borderRadius": "10px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
            "transition": "transform 0.2s, box-shadow 0.2s",
            "cursor": "help" if tooltip else "default",
        },
    )
    # Wrap in a plain div so the native browser tooltip still works
    inner = html.Div(card, title=tooltip) if tooltip else card
    return dbc.Col(inner, md=2)

def _section(title: str, color: str = ACCENT) -> html.Div:
    """Create a section header"""
    return html.Div([
        html.H5(title, style={
            "color": color, "marginTop": "2.5rem", "marginBottom": "0.8rem",
            "fontFamily": "'Space Grotesk', sans-serif", "fontWeight": "600",
            "fontSize": "0.95rem", "textTransform": "uppercase", "letterSpacing": "1px",
        }),
        html.Hr(style={"borderColor": BORDER, "marginTop": 0, "marginBottom": "1rem"}),
    ])

def _empty(msg: str = "Build a model first — click  ▶  Build / Refresh Model  in the sidebar") -> html.Div:
    """Create empty state message"""
    return html.Div([
        html.Div(style={
            "width": "80px", "height": "80px", "borderRadius": "50%",
            "background": f"linear-gradient(135deg, {CARD_BG}, {CARD_BG2})",
            "border": f"2px solid {BORDER2}",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "margin": "8rem auto 1.5rem",
            "fontSize": "2rem",
        }, children="🔒"),
        html.H5(msg, style={
            "textAlign": "center", "color": MUTED, "maxWidth": "480px",
            "margin": "0 auto", "lineHeight": "1.6", "fontWeight": "400",
        }),
    ])

def _dd_style() -> Dict:
    """Dropdown style"""
    return {
        "background": CARD_BG2, "color": "#fff",
        "borderColor": BORDER2, "borderRadius": "7px",
    }

def _input_style() -> Dict:
    """Input style"""
    return {
        "width": "100%", "background": CARD_BG2,
        "border": f"1px solid {BORDER2}", "borderRadius": "7px",
        "color": "#fff", "padding": "0.5rem 0.75rem",
    }

def _badge(text: str, color: str) -> html.Span:
    """Create a badge/pill."""
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        bg = f"rgba({r},{g},{b},0.18)"
    else:
        bg = f"rgba(79,158,255,0.18)"
    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": "2px 8px",
            "borderRadius": "12px",
            "background": bg,
            "color": color,
            "fontSize": "0.7rem",
            "fontWeight": "500",
            "marginLeft": "5px",
        }
    )

# ============================================================================
# KPI BAR BUILDER (shared between build_model and score_deal callbacks)
# ============================================================================
def _build_kpi_bar(
    df_json: Optional[str],
    scored_deals: Optional[list] = None,
    win_meta: Optional[dict] = None,
) -> dbc.Row:
    """
    Build the top KPI bar, blending base dataset metrics with any scored deals.
    
    win_meta: optional win-model metadata for AUC display
    """
    # ── Base dataset metrics ─────────────────────────────────────────────────
    total_rev = 0.0
    total_prof = 0.0
    avg_margin = 35.0
    win_rate = None
    avg_deal = 0.0
    n_products = 0
    avg_confidence = 1.0

    if df_json:
        try:
            df = pd.read_json(df_json, orient="split")
            total_rev = float(df["revenue"].sum())
            total_prof = float((df["revenue"] - df["cost"]).sum())
            avg_margin = float(df["margin_pct"].mean()) if "margin_pct" in df.columns else 35.0
            win_rate = float(df["deal_won"].mean() * 100) if "deal_won" in df.columns else None
            avg_deal = float(df["revenue"].mean())
            n_products = int(df["product"].nunique())
            avg_confidence = float(df["confidence_score"].mean()) if "confidence_score" in df.columns else 1.0
        except Exception:
            pass

    # ── Layer in scored deals ────────────────────────────────────────────────
    n_deals = 0
    if scored_deals:
        n_deals = len(scored_deals)
        deals_rev = sum(d.get("revenue", 0) for d in scored_deals)
        deals_prof = sum(
            d.get("revenue", 0) * d.get("margin_pct", 0) / 100
            for d in scored_deals
        )
        deals_avg_margin = float(np.mean([d.get("margin_pct", 0) for d in scored_deals]))
        deals_avg_win = float(np.mean([d.get("win_probability", 0) for d in scored_deals]) * 100)
        deals_avg_score = float(np.mean([d.get("score", 0) for d in scored_deals]))

        # Combine with base dataset
        total_rev += deals_rev
        total_prof += deals_prof

        # Weighted blend of margin
        if df_json and total_rev > 0:
            avg_margin = (avg_margin * (total_rev - deals_rev) + deals_avg_margin * deals_rev) / total_rev
        elif deals_rev > 0:
            avg_margin = deals_avg_margin

        deals_badge = _badge(f"+{n_deals} deals", CYAN)
        score_badge = _badge(f"Avg score: {deals_avg_score:.0f}", GREEN if deals_avg_score >= 70 else YELLOW)
    else:
        deals_badge = None
        score_badge = None

    confidence_badge = _badge(f"Conf: {avg_confidence:.2f}", GREEN if avg_confidence > 0.8 else YELLOW)

    auc_text = f"AUC {win_meta.get('auc', 0):.3f}" if win_meta else ""

    # ── Compose KPI cards ────────────────────────────────────────────────────
    cards = [
        _card("Total Revenue", f"${total_rev/1e6:,.2f}M", ACCENT,
              sub=deals_badge if n_deals else ""),
        _card("Gross Profit", f"${total_prof/1e6:,.2f}M", GREEN),
        _card("Avg Margin", f"{avg_margin:.1f}%", PURPLE),
        _card("Win Rate", f"{win_rate:.1f}%" if win_rate else "—", YELLOW, auc_text),
        _card("Avg Deal Size", f"${avg_deal:,.0f}", CYAN,
              sub=score_badge if n_deals else ""),
        _card("Active Products", str(n_products), ORANGE, sub=confidence_badge),
    ]

    return dbc.Row(cards, className="g-3", style={"marginBottom": "0.5rem"})



# ============================================================================
# GLOBAL CSS
# ============================================================================
GLOBAL_CSS = f"""
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{
    background: {DARK_BG};
    color: #e0e8f4;
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
    min-height: 100vh;
}}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #3a4a6a; }}
.nav-tab {{
    transition: all 0.18s ease !important;
    cursor: pointer;
}}
.nav-tab:hover {{
    background: linear-gradient(90deg, rgba(79,158,255,0.12), rgba(79,158,255,0.04)) !important;
    border-color: rgba(79,158,255,0.35) !important;
    color: #c8d8f0 !important;
    transform: translateX(3px);
}}
.nav-tab.active {{
    background: linear-gradient(90deg, rgba(79,158,255,0.22), rgba(79,158,255,0.08)) !important;
    border-left: 3px solid {ACCENT} !important;
    color: #fff !important;
}}
.Select-control {{ background: {CARD_BG2} !important; border-color: {BORDER2} !important; color: #fff !important; }}
.Select-menu-outer {{ background: {CARD_BG2} !important; border-color: {BORDER2} !important; }}
.Select-option {{ background: {CARD_BG2} !important; color: #e0e8f4 !important; }}
.Select-option:hover, .Select-option.is-focused {{ background: {BORDER2} !important; }}
.Select-value-label {{ color: #fff !important; }}
.Select-placeholder {{ color: {MUTED} !important; }}
.rc-slider-track {{ background-color: {ACCENT}; }}
.rc-slider-handle {{ border-color: {ACCENT}; background: {ACCENT}; }}
.rc-slider-dot-active {{ border-color: {ACCENT}; }}
table.dataframe th, .table th {{
    background: {CARD_BG2} !important;
    color: {ACCENT} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    border-color: {BORDER2} !important;
}}
table.dataframe td, .table td {{
    background: {CARD_BG} !important;
    color: #c8d8f0 !important;
    border-color: {BORDER} !important;
    font-size: 0.88rem !important;
}}
.table-striped > tbody > tr:nth-of-type(odd) > td {{
    background: rgba(255,255,255,0.02) !important;
}}
.table-hover > tbody > tr:hover > td {{
    background: rgba(79,158,255,0.07) !important;
}}
.status-pill {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 500; letter-spacing: 0.4px;
}}
._dash-loading {{ color: {ACCENT} !important; }}
.kpi-card:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
}}
.section-divider {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 0.5rem 0 1rem;
}}
input[type=number]:focus, input[type=text]:focus {{
    outline: none;
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(79,158,255,0.15);
}}
.upload-area {{
    transition: all 0.2s;
    border: 2px dashed {BORDER2};
}}
.upload-area:hover {{
    border-color: {ACCENT};
    background: rgba(79,158,255,0.08);
}}
.scraping-status {{
    transition: all 0.3s;
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0% {{ opacity: 1; }}
    50% {{ opacity: 0.7; }}
    100% {{ opacity: 1; }}
}}
.tooltip-inner {{
    background: {CARD_BG2};
    border: 1px solid {BORDER2};
    color: #fff;
    max-width: 300px;
    font-size: 0.8rem;
}}
"""

# ============================================================================
# APP INIT
# ============================================================================
app = Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&"
        "family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title=APP_TITLE,
    suppress_callback_exceptions=True,
)
server = app.server

app.index_string = app.index_string.replace(
    "<head>",
    f"<head><style>{GLOBAL_CSS}</style>",
)

# ============================================================================
# SIDEBAR BUILDER (Enhanced with scraping controls)
# ============================================================================
def _build_sidebar() -> html.Div:
    """Build sidebar with grouped navigation and enhanced upload + scraping controls."""
    nav_items = []
    for group_label, options in NAV_GROUPS.items():
        # Group header
        nav_items.append(
            html.Div(group_label, style={
                "color": MUTED, "fontSize": "0.65rem", "letterSpacing": "1.4px",
                "textTransform": "uppercase", "fontWeight": "600",
                "padding": "0.9rem 0.6rem 0.3rem",
                "fontFamily": "'Space Grotesk', sans-serif",
            })
        )
        for opt in options:
            idx = NAV_OPTIONS.index(opt)
            nav_items.append(
                html.Div(
                    children=opt,
                    id={"type": "nav-tab", "index": idx},
                    className="nav-tab",
                    style={
                        "padding": "0.55rem 0.9rem",
                        "marginBottom": "2px",
                        "borderRadius": "7px",
                        "cursor": "pointer",
                        "background": "transparent",
                        "border": f"1px solid transparent",
                        "color": "#8a9bb8",
                        "fontSize": "0.82rem",
                        "fontWeight": "400",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "borderLeft": "3px solid transparent",
                    },
                )
            )

    return html.Div([
        # Logo area
        html.Div([
            html.Div([
                html.Span("S", style={
                    "fontFamily": "'Space Grotesk'", "fontWeight": "700",
                    "fontSize": "1.1rem", "color": ACCENT,
                }),
                html.Span("TRAIVE", style={
                    "fontFamily": "'Space Grotesk'", "fontWeight": "700",
                    "fontSize": "1.1rem", "color": "#c8d8f0", "letterSpacing": "3px",
                }),
            ]),
            html.Div("Pricing Intelligence", style={
                "color": MUTED, "fontSize": "0.7rem", "letterSpacing": "0.5px",
                "marginTop": "2px",
            }),
        ], style={"marginBottom": "1.2rem", "paddingBottom": "1rem",
                   "borderBottom": f"1px solid {BORDER}"}),

        # MODEL CONTROLS
        html.Div("MODEL CONTROLS", style={
            "color": MUTED, "fontSize": "0.65rem", "letterSpacing": "1.4px",
            "fontWeight": "600", "fontFamily": "'Space Grotesk', sans-serif",
            "marginBottom": "0.8rem",
        }),

        html.Label("Record Count", style={"color": "#8a9bb8", "fontSize": "0.78rem",
                                          "marginBottom": "0.3rem", "display": "block"}),
        dcc.Input(
            id="n-records-input", type="number", value=3_600,
            min=500, max=25_000, step=500,
            style={**_input_style(), "marginBottom": "0.9rem", "fontSize": "0.88rem"},
        ),

        html.Label("Data Source", style={"color": "#8a9bb8", "fontSize": "0.78rem",
                                         "marginBottom": "0.4rem", "display": "block"}),
        dcc.RadioItems(
            id="data-mode-radio",
            options=[
                {"label": "  Synthetic Data", "value": "synthetic"},
                {"label": "  Upload CSV / Excel", "value": "upload"},
                {"label": "  Scraped + Synthetic Blend", "value": "blended"},
            ],
            value="synthetic",
            labelStyle={"display": "block", "margin": "0.3rem 0",
                        "color": "#8a9bb8", "fontSize": "0.82rem"},
        ),

        # Enhanced Upload Section
        dcc.Upload(
            id="upload-csv",
            children=html.Div([
                html.Div([
                    html.Span("📁", style={"fontSize": "1.5rem", "marginRight": "8px"}),
                    html.Div([
                        html.Span("Drop file here or ", style={"color": MUTED}),
                        html.A("browse", style={"color": ACCENT, "fontWeight": "600"}),
                    ], style={"display": "inline-block", "verticalAlign": "middle"}),
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Div([
                    html.Span("Supported: .csv, .xlsx, .xls", style={
                        "color": MUTED, "fontSize": "0.7rem", "marginTop": "4px"
                    }),
                    html.Span(" | Max 50MB", style={
                        "color": MUTED, "fontSize": "0.7rem"
                    }),
                ], style={"textAlign": "center", "marginTop": "6px"}),
            ]),
            style={
                "width": "100%", "height": "80px",
                "border": f"2px dashed {BORDER2}", "borderRadius": "10px",
                "textAlign": "center", "margin": "0.8rem 0",
                "color": MUTED, "fontSize": "0.82rem", "cursor": "pointer",
                "background": "rgba(79,158,255,0.05)",
                "transition": "all 0.2s",
                "display": "flex", "flexDirection": "column",
                "justifyContent": "center", "alignItems": "center",
            },
            multiple=False,
        ),

        # Upload feedback
        html.Div(id="upload-feedback", style={
            "minHeight": "3rem", "fontSize": "0.8rem",
            "marginBottom": "0.5rem", "padding": "0.5rem",
            "borderRadius": "6px", "background": "rgba(0,0,0,0.2)",
        }),

        # Data preview toggle
        html.Div(id="upload-preview-container", style={
            "marginBottom": "0.8rem", "display": "none",
        }, children=[
            html.Button(
                "👁 Preview Data", id="toggle-preview-btn",
                style={
                    "width": "100%", "padding": "0.5rem",
                    "background": "transparent",
                    "border": f"1px solid {BORDER2}",
                    "borderRadius": "6px", "color": MUTED,
                    "fontSize": "0.75rem", "cursor": "pointer",
                    "marginBottom": "0.5rem",
                },
            ),
            html.Div(id="data-preview", style={
                "maxHeight": "200px", "overflowY": "auto",
                "fontSize": "0.7rem", "background": CARD_BG2,
                "borderRadius": "6px", "padding": "0.5rem",
            }),
        ]),

        # Column mapping info
        html.Div(id="column-mapping-info", style={
            "fontSize": "0.72rem", "color": MUTED,
            "marginBottom": "0.8rem", "padding": "0.5rem",
            "background": "rgba(79,158,255,0.05)",
            "borderRadius": "6px", "display": "none",
        }),

        # Build model button
        html.Button(
            [html.Span("▶ ", style={"marginRight": "7px"}), "Build / Refresh Model"],
            id="build-model-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "0.8rem",
                "background": f"linear-gradient(135deg, {ACCENT} 0%, #2a6dd6 100%)",
                "border": "none", "borderRadius": "8px",
                "color": "#fff", "fontWeight": "600",
                "fontSize": "0.85rem", "cursor": "pointer",
                "letterSpacing": "0.4px",
                "boxShadow": f"0 4px 16px rgba(79,158,255,0.3)",
                "transition": "all 0.2s",
                "margin": "0.5rem 0",
                "fontFamily": "'Space Grotesk', sans-serif",
            },
        ),

        dcc.Loading(
            id="model-loading", type="circle", color=ACCENT,
            children=html.Div(id="model-loading-output", style={"height": "4px"}),
        ),

        # WEB SCRAPING SECTION
        html.Hr(style={"borderColor": BORDER, "margin": "0.9rem 0"}),
        
        html.Div("MARKET INTELLIGENCE", style={
            "color": MUTED, "fontSize": "0.65rem", "letterSpacing": "1.4px",
            "fontWeight": "600", "fontFamily": "'Space Grotesk', sans-serif",
            "marginBottom": "0.8rem",
        }),

        # Scrape now button
        html.Button(
            [html.Span("🕷️ ", style={"marginRight": "7px"}), "Scrape Competitor Prices"],
            id="scrape-now-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "0.6rem",
                "background": "transparent",
                "border": f"1px solid {ACCENT}",
                "borderRadius": "7px", "color": ACCENT,
                "fontSize": "0.78rem", "cursor": "pointer",
                "transition": "all 0.2s",
                "marginBottom": "0.5rem",
            },
        ),

        # Scraping status
        html.Div(id="scraping-status", style={
            "fontSize": "0.75rem", "color": MUTED,
            "marginBottom": "0.5rem", "padding": "0.3rem",
        }),

        # Real data blend slider
        html.Label("Real Data Blend %", style={
            "color": "#8a9bb8", "fontSize": "0.78rem",
            "marginBottom": "0.3rem", "display": "block",
        }),
        dcc.Slider(
            id="real-data-ratio",
            min=0, max=70, step=5, value=30,
            marks={i: f"{i}%" for i in range(0, 71, 10)},
            tooltip={"placement": "bottom"},
        ),

        # Intelligence report button
        html.Button(
            [html.Span("📊 ", style={"marginRight": "7px"}), "Generate Intelligence Report"],
            id="intel-report-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "0.6rem",
                "background": "transparent",
                "border": f"1px solid {GREEN}",
                "borderRadius": "7px", "color": GREEN,
                "fontSize": "0.78rem", "cursor": "pointer",
                "transition": "all 0.2s",
                "marginTop": "0.5rem",
                "marginBottom": "0.5rem",
            },
        ),

        html.Hr(style={"borderColor": BORDER, "margin": "0.9rem 0"}),

        # Download template button
        html.Button(
            [html.Span("📥 ", style={"marginRight": "7px"}), "Download Template CSV"],
            id="download-template-btn",
            style={
                "width": "100%", "padding": "0.6rem",
                "background": "transparent",
                "border": f"1px solid {BORDER2}",
                "borderRadius": "7px", "color": MUTED,
                "fontSize": "0.78rem", "cursor": "pointer",
                "transition": "all 0.2s",
                "marginBottom": "0.5rem",
            },
        ),
        dcc.Download(id="download-template-component"),

        # Navigation
        html.Div(nav_items, style={"overflowY": "auto"}),

        html.Hr(style={"borderColor": BORDER, "margin": "0.9rem 0"}),

        # Export button
        html.Button(
            [html.Span("⬇ ", style={"marginRight": "7px"}), "Export Results (JSON)"],
            id="download-btn",
            style={
                "width": "100%", "padding": "0.6rem",
                "background": "transparent",
                "border": f"1px solid {BORDER2}",
                "borderRadius": "7px", "color": MUTED,
                "fontSize": "0.78rem", "cursor": "pointer",
                "transition": "all 0.2s",
            },
        ),
        dcc.Download(id="download-component"),

        # Version footer
        html.Div([
            html.Span(f"v{APP_VERSION}", style={"color": MUTED, "fontSize": "0.7rem"}),
            html.Span(" · ", style={"color": BORDER2}),
            html.Span(CURRENT_DATE.strftime("%b %Y"),
                      style={"color": MUTED, "fontSize": "0.7rem"}),
        ], style={"marginTop": "1rem", "textAlign": "center"}),

    ], style={
        "width": "260px",
        "minWidth": "260px",
        "background": SIDEBAR_BG,
        "padding": "1.2rem 0.9rem 1.5rem",
        "height": "100vh",
        "position": "fixed",
        "top": 0,
        "left": 0,
        "overflowY": "auto",
        "borderRight": f"1px solid {BORDER}",
        "zIndex": 100,
    })

# ============================================================================
# LAYOUT
# ============================================================================
app.layout = html.Div([
    dcc.Store(id="data-store"),
    dcc.Store(id="elasticity-store"),
    dcc.Store(id="win-model-store"),
    dcc.Store(id="active-tab-store", data=NAV_OPTIONS[0]),
    dcc.Store(id="competitor-data-store"),  # New store for scraped competitor data
    dcc.Store(id="market-intel-store"),      # New store for market intelligence
    dcc.Store(id="pricing-signals-store"),   # New store for pricing signals
    dcc.Download(id="download-intel-pdf"),  # PDF report download
    
    # Sidebar
    _build_sidebar(),

    # Main content area
    html.Div([

        # Top header bar
        html.Div([
            html.Div([
                html.Div(id="page-title-area", children=[
                    html.H2("Executive Dashboard", style={
                        "fontFamily": "'Space Grotesk', sans-serif",
                        "fontWeight": "700", "fontSize": "1.5rem",
                        "margin": 0, "color": "#e8edf5",
                    }),
                ]),
            ], style={"flex": "1"}),
            html.Div([
                html.Span(CURRENT_DATE.strftime("%A, %d %B %Y"),
                          style={"color": MUTED, "fontSize": "0.8rem"}),
                html.Div(id="live-indicator", children=[
                    html.Span("●", style={"color": GREEN, "marginRight": "5px"}),
                    html.Span("Live", style={"color": MUTED, "fontSize": "0.8rem"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "1rem"}),
        ], style={
            "display": "flex", "alignItems": "center", "justifyContent": "space-between",
            "padding": "1.1rem 2rem",
            "background": f"linear-gradient(180deg, {CARD_BG} 0%, rgba(12,18,33,0.95) 100%)",
            "borderBottom": f"1px solid {BORDER}",
            "position": "sticky", "top": 0, "zIndex": 50,
            "backdropFilter": "blur(10px)",
        }),

        # KPI bar
        html.Div(id="kpi-bar", style={"padding": "1.2rem 2rem 0"}),

        # Intelligence report output (hidden by default)
        html.Div(id="intel-report-output", style={"display": "none"}),

        # Tab content
        html.Div(id="tab-content", style={"padding": "1.5rem 2rem 5rem"}),

    ], style={
        "marginLeft": "260px",
        "minHeight": "100vh",
        "background": DARK_BG,
    }),
], style={"background": DARK_BG, "minHeight": "100vh"})

# ============================================================================
# CALLBACK: NAV HIGHLIGHT
# ============================================================================
@app.callback(
    [Output({"type": "nav-tab", "index": ALL}, "style"),
     Output({"type": "nav-tab", "index": ALL}, "className"),
     Output("active-tab-store", "data"),
     Output("page-title-area", "children")],
    Input({"type": "nav-tab", "index": ALL}, "n_clicks"),
    State("active-tab-store", "data"),
    prevent_initial_call=True,
)
def update_nav(n_clicks_list, current_tab):
    triggered = ctx.triggered_id
    if not triggered:
        raise PreventUpdate
    active_idx = triggered["index"]
    active_label = NAV_OPTIONS[active_idx]
    n = len(NAV_OPTIONS)

    styles = []
    classes = []
    for i in range(n):
        is_active = (i == active_idx)
        if is_active:
            styles.append({
                "padding": "0.55rem 0.9rem",
                "marginBottom": "2px",
                "borderRadius": "7px",
                "cursor": "pointer",
                "background": "linear-gradient(90deg, rgba(79,158,255,0.18), rgba(79,158,255,0.06))",
                "border": f"1px solid rgba(79,158,255,0.3)",
                "borderLeft": f"3px solid {ACCENT}",
                "color": "#e8edf5",
                "fontSize": "0.82rem",
                "fontWeight": "500",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            })
            classes.append("nav-tab active")
        else:
            styles.append({
                "padding": "0.55rem 0.9rem",
                "marginBottom": "2px",
                "borderRadius": "7px",
                "cursor": "pointer",
                "background": "transparent",
                "border": "1px solid transparent",
                "borderLeft": "3px solid transparent",
                "color": "#8a9bb8",
                "fontSize": "0.82rem",
                "fontWeight": "400",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            })
            classes.append("nav-tab")

    # Update page title
    title_text = active_label
    page_title = html.H2(title_text, style={
        "fontFamily": "'Space Grotesk', sans-serif",
        "fontWeight": "700", "fontSize": "1.4rem",
        "margin": 0, "color": "#e8edf5",
    })

    return styles, classes, active_label, page_title

# ============================================================================
# CALLBACK: FILE UPLOAD (Enhanced)
# ============================================================================
@app.callback(
    [Output("upload-feedback", "children", allow_duplicate=True),
     Output("upload-feedback", "style"),
     Output("upload-preview-container", "style"),
     Output("data-preview", "children"),
     Output("column-mapping-info", "children"),
     Output("column-mapping-info", "style"),
     Output("data-store", "data", allow_duplicate=True)],
    Input("upload-csv", "contents"),
    [State("upload-csv", "filename")],
    prevent_initial_call=True,
)
def handle_file_upload(contents, filename):
    """Handle file upload with comprehensive validation."""
    if not contents:
        raise PreventUpdate

    df, err = _parse_upload(contents, filename)
    if df is None:
        file_ext = (filename or "").lower().rsplit(".", 1)[-1]
        is_format_err = "Unsupported" in err
        color = YELLOW if is_format_err else RED
        bg = "rgba(245,166,35,0.15)" if is_format_err else "rgba(255,64,96,0.15)"
        return (
            html.Div([html.Span("⚠ " if is_format_err else "✗ ", style={"color": color}),
                      html.Span(err)]),
            {"background": bg, "border": f"1px solid {color}", "borderRadius": "6px", "color": color},
            {"display": "none"}, "", {"display": "none"}, "", None,
        )

    try:
        # Check minimum rows
        min_rows = CSV_UPLOAD_SPEC.get("min_rows", 100)
        if len(df) < min_rows:
            return (
                html.Div([html.Span("⚠ ", style={"color": YELLOW}),
                          html.Span(f"Only {len(df)} rows. Minimum {min_rows} recommended.")]),
                {"background": "rgba(245,166,35,0.15)", "border": "1px solid " + YELLOW,
                 "borderRadius": "6px", "color": YELLOW},
                {"display": "block"}, "", {"display": "none"}, "", None,
            )
        
        # Column mapping check
        required_cols = DATA_COLLECTION_GUIDANCE["minimum_required_columns"]
        column_mappings = DATA_COLLECTION_GUIDANCE.get("column_mappings", {})
        
        df_columns_lower = {col.lower().strip(): col for col in df.columns}
        mapped_cols = {}
        unmapped_required = []
        
        for req_col in required_cols:
            found = False
            if req_col.lower() in df_columns_lower:
                mapped_cols[req_col] = df_columns_lower[req_col.lower()]
                found = True
            else:
                for alt_name in column_mappings.get(req_col, []):
                    if alt_name.lower() in df_columns_lower:
                        mapped_cols[req_col] = df_columns_lower[alt_name.lower()]
                        found = True
                        break
            
            if not found:
                unmapped_required.append(req_col)
        
        # Build feedback message
        feedback_parts = []
        feedback_style = {"background": "rgba(46,232,154,0.15)", 
                         "border": "1px solid " + GREEN,
                         "borderRadius": "6px", "color": GREEN}
        
        if unmapped_required:
            feedback_parts.append(
                html.Div([
                    html.Span("⚠ ", style={"color": YELLOW}),
                    html.Span(f"Missing columns: {', '.join(unmapped_required)}"),
                ], style={"marginBottom": "4px"})
            )
            feedback_style = {"background": "rgba(245,166,35,0.15)",
                             "border": "1px solid " + YELLOW,
                             "borderRadius": "6px", "color": YELLOW}
        else:
            feedback_parts.append(
                html.Div([
                    html.Span("✓ ", style={"color": GREEN}),
                    html.Span(f"{len(df):,} rows loaded | {len(df.columns)} columns"),
                ], style={"marginBottom": "4px"})
            )
        
        # Data preview
        preview_html = []
        preview_df = df.head(5).copy()
        preview_html.append(
            html.Table([
                html.Thead(html.Tr([
                    html.Th(col, style={
                        "background": BORDER2, "color": ACCENT,
                        "padding": "4px 8px", "fontSize": "0.65rem",
                        "textTransform": "uppercase",
                    }) for col in preview_df.columns
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(str(val)[:50], style={
                            "padding": "4px 8px", "borderBottom": f"1px solid {BORDER}",
                            "fontSize": "0.68rem", "color": "#e0e8f4",
                        }) for val in row[1:]  # skip the pandas Index field
                    ]) for row in preview_df.itertuples()
                ])
            ], style={"width": "100%", "borderCollapse": "collapse"})
        )
        
        # Column mapping info
        mapping_info = ""
        mapping_style = {"display": "none"}
        if mapped_cols:
            mapping_lines = [f"{req} ← {actual}" for req, actual in mapped_cols.items()]
            mapping_info = html.Div([
                html.Div("Column Mapping Detected:", style={
                    "fontWeight": "600", "marginBottom": "4px", "color": ACCENT
                }),
                html.Div(" | ".join(mapping_lines), style={
                    "fontSize": "0.68rem", "lineHeight": "1.4"
                })
            ])
            mapping_style = {"display": "block"}
        
        # Preprocess
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        
        # Add confidence score if not present
        if "confidence_score" not in df.columns:
            df["confidence_score"] = 1.0
        
        return (
            html.Div(feedback_parts),
            feedback_style,
            {"display": "block"},
            preview_html,
            mapping_info,
            mapping_style,
            df.to_json(date_format="iso", orient="split"),
        )
        
    except Exception as exc:
        return (
            html.Div([
                html.Span("✗ ", style={"color": RED}),
                html.Span(f"Upload error: {str(exc)[:150]}"),
            ]),
            {"background": "rgba(255,64,96,0.15)", "border": "1px solid " + RED,
             "borderRadius": "6px", "color": RED},
            {"display": "none"}, "", {"display": "none"}, "", None
        )

# ============================================================================
# CALLBACK: DOWNLOAD TEMPLATE CSV
# ============================================================================
@app.callback(
    Output("download-template-component", "data"),
    Input("download-template-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_template(n_clicks):
    """Generate and download CSV template with example data."""
    if not n_clicks:
        raise PreventUpdate
    
    # Create template with required columns
    template_df = pd.DataFrame(columns=CSV_UPLOAD_SPEC["template_columns"])
    
    # Add example rows
    example_rows = [
        {
            "date": CURRENT_DATE.strftime("%Y-%m-%d"),
            "product": "Editorial Services – Standard",
            "customer_type": "Academic Publishers",
            "region": "North America",
            "actual_price": 2800,
            "volume": 10,
            "revenue": 28000,
            "cost": 11000,
            "deal_won": 1,
            "base_price": 2800,
            "discount_pct": 0,
            "competitor": "Aptara",
            "competitor_price": 2500,
            "source": "internal",
            "confidence_score": 1.0,
        },
        {
            "date": (CURRENT_DATE - timedelta(days=30)).strftime("%Y-%m-%d"),
            "product": "Data Annotation – Basic",
            "customer_type": "STM Publishers",
            "region": "Europe",
            "actual_price": 420,
            "volume": 25,
            "revenue": 10500,
            "cost": 4500,
            "deal_won": 0,
            "base_price": 450,
            "discount_pct": 7,
            "competitor": "Innodata",
            "competitor_price": 400,
            "source": "scraped",
            "confidence_score": 0.85,
        },
    ]
    
    template_df = pd.concat([template_df, pd.DataFrame(example_rows)], ignore_index=True)
    
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dict(
        content=csv_buffer.getvalue(),
        filename=f"straive_data_template_{ts}.csv",
        type="text/csv",
    )

# ============================================================================
# CALLBACK: SCRAPE COMPETITOR PRICES
# ============================================================================
@app.callback(
    [Output("scraping-status", "children"),
     Output("scraping-status", "style"),
     Output("competitor-data-store", "data"),
     Output("market-intel-store", "data"),
     Output("pricing-signals-store", "data")],
    Input("scrape-now-btn", "n_clicks"),
    prevent_initial_call=True,
)
def scrape_competitor_prices(n_clicks):
    """Scrape competitor prices and update stores."""
    if not n_clicks:
        raise PreventUpdate
    
    try:
        # Scrape competitor prices
        price_points = data_orchestrator.gather_all_competitor_prices(use_cache=False)
        
        # Gather market intelligence
        market_intel = data_orchestrator.gather_market_intelligence()
        
        # Get signals
        signals = data_orchestrator.signal_detector.get_recent_signals(hours=168)
        
        # Convert to JSON-serializable format
        price_data = []
        for pp in price_points:
            d = {
                "competitor_name": pp.competitor_name,
                "service_name": pp.service_name,
                "price": pp.price,
                "currency": pp.currency,
                "source_url": pp.source_url,
                "confidence_score": pp.confidence_score,
                "scraped_at": pp.scraped_at.isoformat() if pp.scraped_at else None,
            }
            price_data.append(d)
        
        intel_data = []
        for mi in market_intel:
            d = {
                "source": mi.source,
                "data_type": mi.data_type,
                "content": mi.content,
                "timestamp": mi.timestamp.isoformat() if mi.timestamp else None,
                "relevance_score": mi.relevance_score,
            }
            intel_data.append(d)
        
        signal_data = []
        for s in signals:
            d = {
                "signal_type": s.signal_type,
                "competitor": s.competitor,
                "magnitude": s.magnitude,
                "confidence": s.confidence,
                "detected_at": s.detected_at.isoformat(),
                "details": s.details,
            }
            signal_data.append(d)
        
        status = html.Div([
            html.Div([
                html.Span("✓ ", style={"color": GREEN}),
                html.Span(f"Scraped {len(price_points)} competitor price points"),
            ]),
            html.Div([
                html.Span("📰 ", style={"color": ACCENT}),
                html.Span(f"Found {len(market_intel)} market intelligence items"),
            ]),
            html.Div([
                html.Span("🚨 ", style={"color": YELLOW if signals else MUTED}),
                html.Span(f"{len(signals)} pricing signals detected"),
            ]),
        ])
        
        status_style = {
            "background": "rgba(46,232,154,0.1)",
            "border": f"1px solid {GREEN}",
            "borderRadius": "6px", "color": "#fff",
            "padding": "0.5rem",
        }
        
        return status, status_style, _json_dumps(price_data), _json_dumps(intel_data), _json_dumps(signal_data)
        
    except Exception as e:
        error_status = html.Div([
            html.Span("✗ ", style={"color": RED}),
            html.Span(f"Scraping failed: {str(e)[:100]}"),
        ])
        error_style = {
            "background": "rgba(255,64,96,0.15)",
            "border": f"1px solid {RED}",
            "borderRadius": "6px", "color": RED,
            "padding": "0.5rem",
        }
        return error_status, error_style, None, None, None

# ============================================================================
# CALLBACK: BUILD MODEL (Enhanced)
# ============================================================================
@app.callback(
    [Output("data-store", "data", allow_duplicate=True),
     Output("elasticity-store", "data"),
     Output("win-model-store", "data"),
     Output("kpi-bar", "children"),
     Output("upload-feedback", "children", allow_duplicate=True),
     Output("model-loading-output", "children")],
    Input("build-model-btn", "n_clicks"),
    [State("n-records-input", "value"),
     State("data-mode-radio", "value"),
     State("upload-csv", "contents"),
     State("upload-csv", "filename"),
     State("data-store", "data"),
     State("real-data-ratio", "value"),
     State("competitor-data-store", "data"),
     ],
    prevent_initial_call=True,
)
def build_enhanced_model(n_clicks, n_rows, mode, contents, filename, existing_df_json, blend_ratio, competitor_json, scored_deals_json=None):
    """Enhanced build model with real data blending."""
    if not n_clicks:
        raise PreventUpdate
    
    df = None
    feedback = ""
    
    # Parse competitor data if available
    competitor_data = []
    if competitor_json:
        try:
            competitor_data = json.loads(competitor_json)
        except Exception:
            pass
    
    if mode == "upload" and contents:
        # Handle uploaded file
        df, err = _parse_upload(contents, filename)
        if df is None:
            return [None, None, None, [],
                    html.Span(f"⚠ {err}", style={"color": YELLOW}), ""]
        
        # Add confidence score if not present
        if "confidence_score" not in df.columns:
            df["confidence_score"] = 1.0
        
        feedback = html.Span(f"✓ Uploaded {len(df):,} rows", style={"color": GREEN})
        
    elif mode == "blended":
        # Use enhanced generator with real data blending
        try:
            enhanced_generator.real_data_ratio = blend_ratio / 100.0
            df = enhanced_generator.generate_with_real_data(n=n_rows or 3600)

            # Blend in competitor price-point rows only when they share the
            # transaction schema (otherwise NaN columns corrupt model fitting)
            if competitor_data:
                comp_df = pd.DataFrame(competitor_data)
                shared_cols = [c for c in comp_df.columns if c in df.columns]
                if shared_cols:
                    comp_df = comp_df[shared_cols].copy()
                    comp_df["source"] = "scraped"
                    df = pd.concat([df, comp_df], ignore_index=True)

            feedback = html.Div([
                html.Span(f"✓ Generated {len(df):,} rows", style={"color": GREEN}),
                html.Br(),
                html.Span(f"Blended with {blend_ratio}% real scraped data",
                          style={"color": MUTED, "fontSize": "0.75rem"}),
            ])
        except Exception as e:
            # Fallback to pure synthetic
            df = DataGenerator().generate(n=n_rows or 3600)
            feedback = html.Span(f"✓ Synthetic data (scraping failed: {str(e)[:50]})",
                               style={"color": YELLOW})
    
    else:  # synthetic mode
        df = DataGenerator().generate(n=n_rows or 3600)
        feedback = html.Span("✓ Synthetic dataset ready", style={"color": GREEN})
    
    if df is None or df.empty:
        return [None, None, None, [], feedback, ""]
    
    # Build models
    elast = ModelComparator.fit_elasticity_models(df)
    
    # Enhance elasticity with market data if available
    if competitor_data:
        elast = market_integrator.enhance_elasticity_with_market_data(elast, competitor_data)
    
    win_bundle = ModelComparator.fit_win_probability_model(df)
    win_meta = {k: v for k, v in win_bundle.items() if k != "model"}

    df_json_str = df.to_json(date_format="iso", orient="split")
    kpi_bar = _build_kpi_bar(df_json_str, win_meta=win_meta)
    
    return (
        df_json_str,
        _json_dumps(elast),
        _json_dumps(win_meta),
        kpi_bar,
        feedback,
        "",
    )

# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================
def _build_intelligence_pdf(
    df: "pd.DataFrame | None",
    competitor_data: list,
    market_intel: list,
    signals: list,
    report: dict,
    opportunities: list,
) -> bytes:
    """Build a detailed PDF pricing-intelligence report and return bytes."""

    buffer = io.BytesIO()

    # ── Colour palette (STRAIVE dark theme adapted for print) ────────────────
    NAVY   = colors.HexColor("#0c1221")
    BLUE   = colors.HexColor("#4f9eff")
    TEAL   = colors.HexColor("#00c6c6")
    GREEN  = colors.HexColor("#2ee89a")
    YELLOW = colors.HexColor("#f5a623")
    RED    = colors.HexColor("#ff4060")
    PURPLE = colors.HexColor("#b388ff")
    MUTED  = colors.HexColor("#6b7a99")
    WHITE  = colors.white
    LIGHT  = colors.HexColor("#e8edf5")
    CARD   = colors.HexColor("#111827")

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="STRAIVE Pricing Intelligence Report",
        author="STRAIVE Pricing Platform",
    )

    W, H = A4
    body_w = W - 4*cm

    # ── Styles ────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def _style(name, parent="Normal", **kw):
        s = ParagraphStyle(name, parent=base[parent], **kw)
        return s

    sTitle   = _style("sTitle",   "Title",
                      fontSize=28, textColor=BLUE,
                      fontName="Helvetica-Bold", spaceAfter=4,
                      alignment=TA_CENTER)
    sSub     = _style("sSub",     "Normal",
                      fontSize=11, textColor=MUTED,
                      alignment=TA_CENTER, spaceAfter=2)
    sH1      = _style("sH1",      "Heading1",
                      fontSize=14, textColor=BLUE,
                      fontName="Helvetica-Bold",
                      spaceBefore=14, spaceAfter=4)
    sH2      = _style("sH2",      "Heading2",
                      fontSize=11, textColor=TEAL,
                      fontName="Helvetica-Bold",
                      spaceBefore=8, spaceAfter=3)
    sBody    = _style("sBody",    "Normal",
                      fontSize=9, textColor=LIGHT,
                      leading=14, spaceAfter=4)
    sSmall   = _style("sSmall",   "Normal",
                      fontSize=8, textColor=MUTED, leading=12)
    sBullet  = _style("sBullet",  "Normal",
                      fontSize=9, textColor=LIGHT,
                      leading=14, leftIndent=12, spaceAfter=3,
                      bulletIndent=4)
    sKpiVal  = _style("sKpiVal",  "Normal",
                      fontSize=22, textColor=BLUE,
                      fontName="Helvetica-Bold",
                      alignment=TA_CENTER, spaceAfter=0)
    sKpiLbl  = _style("sKpiLbl",  "Normal",
                      fontSize=7.5, textColor=MUTED,
                      alignment=TA_CENTER, spaceAfter=0,
                      textTransform="uppercase")
    sGood    = _style("sGood",    "Normal",
                      fontSize=9, textColor=GREEN,  leading=14)
    sWarn    = _style("sWarn",    "Normal",
                      fontSize=9, textColor=YELLOW, leading=14)
    sBad     = _style("sBad",     "Normal",
                      fontSize=9, textColor=RED,    leading=14)
    sRight   = _style("sRight",   "Normal",
                      fontSize=9, textColor=LIGHT,
                      alignment=TA_RIGHT)

    # ── Table style helpers ───────────────────────────────────────────────────
    HDR_BG   = colors.HexColor("#1a2540")
    ROW1_BG  = colors.HexColor("#0f1a2e")
    ROW2_BG  = colors.HexColor("#111827")

    def _tbl_style(extra=None):
        base_cmds = [
            ("BACKGROUND",   (0, 0), (-1,  0), HDR_BG),
            ("TEXTCOLOR",    (0, 0), (-1,  0), BLUE),
            ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1,  0), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW1_BG, ROW2_BG]),
            ("TEXTCOLOR",    (0, 1), (-1, -1), LIGHT),
            ("FONTSIZE",     (0, 1), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#1e2d4a")),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 0), (-1, 0), [HDR_BG]),
        ]
        if extra:
            base_cmds.extend(extra)
        return TableStyle(base_cmds)

    def _hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#1e2d4a"), spaceAfter=8)

    def _kpi_row(kpis):
        """kpis: list of (label, value, color_hex) tuples – renders as inline KPI cards."""
        n = len(kpis)
        col_w = body_w / n
        data = [[Paragraph(v, _style(f"kv{i}", "Normal",
                                     fontSize=20, textColor=colors.HexColor(c),
                                     fontName="Helvetica-Bold",
                                     alignment=TA_CENTER))
                 for i, (_, v, c) in enumerate(kpis)],
                [Paragraph(l, _style(f"kl{i}", "Normal",
                                     fontSize=7, textColor=MUTED,
                                     alignment=TA_CENTER))
                 for i, (l, _, _) in enumerate(kpis)]]
        t = Table(data, colWidths=[col_w]*n, rowHeights=[26, 14])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), ROW1_BG),
            ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#1e2d4a")),
            ("LINEAFTER",    (0, 0), (-2, -1), 0.5, colors.HexColor("#1e2d4a")),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    # ── Page background / header / footer applied to every page ────────────
    # NOTE: this is defined as a closure inside the builder so it captures
    # the W, H, NAVY, BLUE, MUTED local colour constants.  ReportLab can call
    # it in the same thread without pickling issues.
    def _on_page(canvas, doc):  # noqa: ANN001
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        # Header bar
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.rect(0, H - 1.6*cm, W, 1.6*cm, fill=1, stroke=0)
        # STRAIVE wordmark
        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(BLUE)
        canvas.drawString(2*cm, H - 1.1*cm, "STRAIVE")
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(MUTED)
        canvas.drawString(2*cm + 55, H - 1.1*cm, "Pricing Intelligence Platform")
        # Page number (right)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - 2*cm, H - 1.1*cm,
                               f"Page {doc.page}  ·  {datetime.now().strftime('%d %b %Y')}")
        # Footer bar
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.rect(0, 0, W, 1.2*cm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2*cm, 0.45*cm,
                          "CONFIDENTIAL — STRAIVE Internal Use Only")
        canvas.drawRightString(W - 2*cm, 0.45*cm,
                               "Generated by STRAIVE Pricing Intelligence Platform")
        canvas.restoreState()

    # ── Story (PDF flowable list) ──────────────────────────────────────────────
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3.5*cm))
    story.append(Paragraph("STRAIVE", _style("cover_brand", "Normal",
                                              fontSize=38, textColor=BLUE,
                                              fontName="Helvetica-Bold",
                                              alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Pricing Intelligence Report",
                            _style("cover_title", "Normal",
                                   fontSize=20, textColor=LIGHT,
                                   fontName="Helvetica-Bold",
                                   alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%A, %d %B %Y  %H:%M UTC')}",
        _style("cover_date", "Normal", fontSize=10, textColor=MUTED,
               alignment=TA_CENTER)))
    story.append(Spacer(1, 1.2*cm))
    story.append(_hr())
    story.append(Spacer(1, 0.6*cm))

    # Summary KPIs on cover
    n_comp     = len({c.get("competitor_name","") for c in competitor_data})
    n_prices   = len(competitor_data)
    n_intel    = len(market_intel)
    n_signals  = len(signals)
    n_opps     = len(opportunities)
    ds_rows    = len(df) if df is not None else 0

    story.append(_kpi_row([
        ("Competitors Tracked",  str(n_comp),    "#4f9eff"),
        ("Price Points",         str(n_prices),   "#00c6c6"),
        ("Intel Items",          str(n_intel),    "#b388ff"),
        ("Signals Detected",     str(n_signals),  "#f5a623"),
        ("Opportunities",        str(n_opps),     "#2ee89a"),
        ("Dataset Rows",         f"{ds_rows:,}",  "#6b7a99"),
    ]))
    story.append(Spacer(1, 0.8*cm))

    # Executive summary blurb
    story.append(Paragraph(
        "This report provides a comprehensive overview of the competitive pricing "
        "landscape, market intelligence signals, and actionable pricing opportunities "
        "identified by the STRAIVE Pricing Intelligence Platform. All data is based "
        "on the most recent scraping cycle and internal model outputs.",
        sBody))
    story.append(PageBreak())

    # ── 1. Competitor Price Summary ───────────────────────────────────────────
    story.append(Paragraph("1. Competitor Price Summary", sH1))
    story.append(_hr())

    if competitor_data:
        # Build competitor summary table
        avg_by_comp = report.get("competitor_prices", {}).get("avg_price_by_competitor", {})
        from config import COMPETITORS
        comp_rows = []
        for comp, meta in COMPETITORS.items():
            comp_rows.append([
                Paragraph(comp, sBody),
                Paragraph(f"${avg_by_comp.get(comp, 0):,.2f}", sRight),
                Paragraph(f"{meta.get('market_share', 0)*100:.0f}%",
                          _style("ms", "Normal", fontSize=9,
                                 textColor=TEAL, alignment=TA_RIGHT)),
                Paragraph(f"{meta.get('win_rate_vs', 0.5)*100:.0f}%",
                          _style("wr", "Normal", fontSize=9,
                                 textColor=GREEN, alignment=TA_RIGHT)),
                Paragraph(f"{meta.get('quality_score', 7):.1f}/10",
                          _style("qs", "Normal", fontSize=9,
                                 textColor=YELLOW, alignment=TA_RIGHT)),
            ])

        tbl = Table(
            [["Competitor", "Avg Price", "Market Share", "Win Rate vs", "Quality"]] + comp_rows,
            colWidths=[body_w*0.30, body_w*0.18, body_w*0.18, body_w*0.18, body_w*0.16],
        )
        tbl.setStyle(_tbl_style())
        story.append(tbl)
        story.append(Spacer(1, 0.5*cm))

        # Raw price points table (latest 15)
        if n_prices > 0:
            story.append(Paragraph("Recent Price Points (latest 15)", sH2))
            pp_rows = []
            for pp in competitor_data[:15]:
                scraped = _fmt_date(pp.get("scraped_at"))
                conf = pp.get("confidence_score", 0)
                conf_style = sGood if conf >= 0.8 else sWarn
                pp_rows.append([
                    Paragraph(pp.get("competitor_name", "—"), sBody),
                    Paragraph(pp.get("service_name", "—")[:40], sSmall),
                    Paragraph(f"${pp.get('price', 0):,.2f}", sRight),
                    Paragraph(pp.get("currency", "USD"), sSmall),
                    Paragraph(f"{conf:.0%}", conf_style),
                    Paragraph(scraped, sSmall),
                ])
            price_tbl = Table(
                [["Competitor", "Service", "Price", "CCY", "Confidence", "Scraped"]] + pp_rows,
                colWidths=[body_w*0.22, body_w*0.28, body_w*0.14,
                           body_w*0.08, body_w*0.14, body_w*0.14],
            )
            price_tbl.setStyle(_tbl_style())
            story.append(price_tbl)
    else:
        story.append(Paragraph(
            "No competitor price data available. Run 'Scrape Competitor Prices' "
            "from the sidebar to populate this section.", sWarn))

    story.append(PageBreak())

    # ── 2. Market Intelligence ────────────────────────────────────────────────
    story.append(Paragraph("2. Market Intelligence", sH1))
    story.append(_hr())

    if market_intel:
        mi_rows = []
        for mi in market_intel[:20]:
            ts = _fmt_date(mi.get("timestamp"))
            rel = mi.get("relevance_score", 0)
            rel_col = GREEN if rel >= 0.7 else (YELLOW if rel >= 0.4 else MUTED)
            mi_rows.append([
                Paragraph(mi.get("source", "—")[:25], sSmall),
                Paragraph(mi.get("data_type", "—"), sSmall),
                Paragraph(str(mi.get("content", "—"))[:120], sSmall),
                Paragraph(f"{rel:.0%}",
                          _style("rel", "Normal", fontSize=8,
                                 textColor=rel_col, alignment=TA_RIGHT)),
                Paragraph(ts, sSmall),
            ])
        mi_tbl = Table(
            [["Source", "Type", "Content", "Relevance", "Date"]] + mi_rows,
            colWidths=[body_w*0.18, body_w*0.12, body_w*0.46,
                       body_w*0.10, body_w*0.14],
        )
        mi_tbl.setStyle(_tbl_style())
        story.append(mi_tbl)
    else:
        story.append(Paragraph(
            "No market intelligence items found in this cycle.", sWarn))

    story.append(PageBreak())

    # ── 3. Pricing Signals ────────────────────────────────────────────────────
    story.append(Paragraph("3. Active Pricing Signals", sH1))
    story.append(_hr())

    if signals:
        for s in signals:
            mag   = s.get("magnitude", 0)
            conf  = s.get("confidence", 0)
            label = f"{'▲' if mag >= 0 else '▼'} {abs(mag):.1f}% price {'increase' if mag >= 0 else 'decrease'}"
            clr   = GREEN if mag >= 0 else RED

            story.append(KeepTogether([
                Paragraph(
                    f"<b>{s.get('competitor','?')}</b> — {s.get('signal_type','?')}",
                    _style("sh", "Normal", fontSize=10, textColor=LIGHT,
                           fontName="Helvetica-Bold")),
                Table(
                    [[Paragraph(label, _style("sl", "Normal", fontSize=9,
                                              textColor=clr)),
                      Paragraph(f"Confidence: {conf*100:.0f}%",
                                _style("sc", "Normal", fontSize=9,
                                       textColor=YELLOW, alignment=TA_RIGHT)),
                      Paragraph(f"Detected: {_fmt_date(s.get('detected_at'))}",
                                _style("sd", "Normal", fontSize=8,
                                       textColor=MUTED, alignment=TA_RIGHT))]],
                    colWidths=[body_w*0.40, body_w*0.30, body_w*0.30],
                    style=TableStyle([
                        ("BACKGROUND",   (0,0),(-1,-1), ROW1_BG),
                        ("BOX",          (0,0),(-1,-1), 0.5,
                         colors.HexColor("#1e2d4a")),
                        ("TOPPADDING",   (0,0),(-1,-1), 5),
                        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                        ("LEFTPADDING",  (0,0),(-1,-1), 8),
                    ])
                ),
                Paragraph(
                    str(s.get("details", ""))[:300],
                    sSmall),
                Spacer(1, 0.3*cm),
            ]))
    else:
        story.append(Paragraph("No active pricing signals detected.", sGood))

    story.append(PageBreak())

    # ── 4. Pricing Opportunities ──────────────────────────────────────────────
    story.append(Paragraph("4. Pricing Opportunities", sH1))
    story.append(_hr())

    if opportunities:
        opp_rows = []
        for o in opportunities:
            pg = o.get("potential_gain", 0)
            pg_s = _style("pg", "Normal", fontSize=9,
                          textColor=(GREEN if pg >= 0 else RED),
                          alignment=TA_RIGHT)
            opp_rows.append([
                Paragraph(str(o.get("product","—"))[:35], sBody),
                Paragraph(str(o.get("type","—")), sSmall),
                Paragraph(f"${o.get('current_price', 0):,.2f}", sRight),
                Paragraph(f"${o.get('market_price', 0):,.2f}", sRight),
                Paragraph(f"{pg:+.1f}%", pg_s),
                Paragraph(f"{o.get('confidence', 0):.0%}", sSmall),
            ])
        opp_tbl = Table(
            [["Product", "Type", "Current Price", "Market Price",
              "Potential Gain", "Confidence"]] + opp_rows,
            colWidths=[body_w*0.28, body_w*0.14, body_w*0.14,
                       body_w*0.14, body_w*0.16, body_w*0.14],
        )
        opp_tbl.setStyle(_tbl_style())
        story.append(opp_tbl)
    else:
        story.append(Paragraph(
            "No specific pricing opportunities detected. Ensure competitor price "
            "data is available for opportunity analysis.", sSmall))

    story.append(PageBreak())

    # ── 5. Dataset Summary ────────────────────────────────────────────────────
    story.append(Paragraph("5. Dataset Summary", sH1))
    story.append(_hr())

    if df is not None and not df.empty:
        total_rev  = float(df["revenue"].sum()) if "revenue" in df.columns else 0
        total_prof = float((df["revenue"] - df["cost"]).sum()) \
                     if ("revenue" in df.columns and "cost" in df.columns) else 0
        avg_margin = float(df["margin_pct"].mean()) \
                     if "margin_pct" in df.columns else 0
        win_rate   = float(df["deal_won"].mean() * 100) \
                     if "deal_won" in df.columns else 0
        avg_deal   = float(df["revenue"].mean()) if "revenue" in df.columns else 0
        n_products = int(df["product"].nunique()) if "product" in df.columns else 0
        n_segments = int(df["segment"].nunique()) if "segment" in df.columns else 0
        n_regions  = int(df["region"].nunique()) if "region" in df.columns else 0

        story.append(_kpi_row([
            ("Total Revenue",   f"${total_rev/1e6:,.2f}M", "#4f9eff"),
            ("Gross Profit",    f"${total_prof/1e6:,.2f}M", "#2ee89a"),
            ("Avg Margin",      f"{avg_margin:.1f}%",       "#b388ff"),
            ("Win Rate",        f"{win_rate:.1f}%",         "#f5a623"),
            ("Avg Deal Size",   f"${avg_deal:,.0f}",        "#00c6c6"),
            ("Active Products", str(n_products),            "#6b7a99"),
        ]))
        story.append(Spacer(1, 0.5*cm))

        # Segment breakdown table
        if "segment" in df.columns:
            story.append(Paragraph("Revenue by Segment", sH2))
            seg_df = df.groupby("segment").agg(
                revenue=("revenue", "sum"),
                margin=("margin_pct", "mean"),
                win_rate=("deal_won", "mean"),
                deals=("deal_won", "count"),
            ).reset_index().sort_values("revenue", ascending=False)
            seg_rows = []
            for _, row in seg_df.iterrows():
                seg_rows.append([
                    Paragraph(str(row["segment"]), sBody),
                    Paragraph(f"${row['revenue']/1e3:,.1f}K", sRight),
                    Paragraph(f"{row['margin']:.1f}%",
                              _style("m", "Normal", fontSize=9,
                                     textColor=(GREEN if row['margin'] >= 30 else YELLOW),
                                     alignment=TA_RIGHT)),
                    Paragraph(f"{row['win_rate']*100:.1f}%", sRight),
                    Paragraph(f"{int(row['deals']):,}", sSmall),
                ])
            seg_tbl = Table(
                [["Segment", "Revenue", "Margin %", "Win Rate", "Deals"]] + seg_rows,
                colWidths=[body_w*0.36, body_w*0.18, body_w*0.16,
                           body_w*0.16, body_w*0.14],
            )
            seg_tbl.setStyle(_tbl_style())
            story.append(seg_tbl)
    else:
        story.append(Paragraph(
            "No dataset loaded. Build a model first to include dataset metrics.", sWarn))

    story.append(PageBreak())

    # ── 6. Recommendations ───────────────────────────────────────────────────
    story.append(Paragraph("6. Strategic Recommendations", sH1))
    story.append(_hr())

    recs = report.get("recommendations", [])
    if recs:
        for i, rec in enumerate(recs, 1):
            story.append(Paragraph(f"{i}.  {rec}", sBullet))
            story.append(Spacer(1, 0.15*cm))
    else:
        story.append(Paragraph("No specific recommendations at this time.", sSmall))

    story.append(Spacer(1, 1*cm))
    story.append(_hr())
    story.append(Paragraph(
        f"Report generated on {datetime.now().strftime('%d %B %Y at %H:%M')} by the "
        "STRAIVE Dynamic Pricing &amp; Revenue Intelligence Platform v4.1.0. "
        "This document is confidential and intended solely for internal STRAIVE use.",
        sSmall))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ============================================================================
# CALLBACK: GENERATE INTELLIGENCE REPORT
# ============================================================================
@app.callback(
    Output("intel-report-output", "children"),
    Output("intel-report-output", "style"),
    Output("download-intel-pdf", "data"),
    Input("intel-report-btn", "n_clicks"),
    [State("data-store", "data"),
     State("competitor-data-store", "data"),
     State("market-intel-store", "data"),
     State("pricing-signals-store", "data")],
    prevent_initial_call=True,
)
def generate_intelligence_report(n_clicks, df_json, competitor_json, intel_json, signals_json):
    """Generate pricing intelligence report (on-screen + PDF download)."""
    if not n_clicks:
        raise PreventUpdate

    try:
        # ── Parse data ────────────────────────────────────────────────────────
        df = pd.read_json(df_json, orient="split") if df_json else None
        competitor_data = json.loads(competitor_json) if competitor_json else []
        market_intel    = json.loads(intel_json)     if intel_json     else []
        signals         = json.loads(signals_json)   if signals_json   else []

        report = data_orchestrator.generate_pricing_intelligence_report()
        opportunities = (
            market_integrator.detect_pricing_opportunities(df, competitor_data)
            if df is not None and competitor_data else []
        )

        # ── Build PDF ─────────────────────────────────────────────────────────
        pdf_bytes = _build_intelligence_pdf(
            df, competitor_data, market_intel, signals, report, opportunities
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_download = dcc.send_bytes(pdf_bytes, f"straive_intel_report_{ts}.pdf")

        # ── On-screen panel ───────────────────────────────────────────────────
        from config import COMPETITORS as _COMP
        screen_ui = html.Div([
            _section("📊 Pricing Intelligence Report", color=ACCENT),

            # ── Success banner with download prompt ──────────────────────────
            dbc.Alert([
                html.Span("✅ ", style={"fontSize": "1.1rem"}),
                html.Strong("PDF report generated and downloading now. "),
                html.Span(
                    f"File: straive_intel_report_{ts}.pdf",
                    style={"color": MUTED, "fontSize": "0.82rem"},
                ),
            ], color="success",
               style={"background": "rgba(46,232,154,0.1)",
                      "border": f"1px solid {GREEN}",
                      "color": "#c8d8f0", "marginBottom": "1.2rem"}),

            # ── Summary KPIs ─────────────────────────────────────────────────
            dbc.Row([
                _card("Competitors Tracked", str(len(_COMP)),              ACCENT),
                _card("Price Points",        str(len(competitor_data)),    CYAN),
                _card("Intel Items",         str(len(market_intel)),       PURPLE),
                _card("Active Signals",      str(len(signals)),
                      YELLOW if signals else MUTED),
                _card("Opportunities",       str(len(opportunities)),
                      GREEN if opportunities else MUTED),
                _card("Report Generated",    datetime.now().strftime("%H:%M"), ORANGE),
            ], className="g-3", style={"marginBottom": "1.5rem"}),

            # ── Competitor table ─────────────────────────────────────────────
            html.Div([
                html.H6("💰 Competitor Price Summary",
                        style={"color": ACCENT, "marginBottom": "1rem"}),
                dbc.Table.from_dataframe(
                    pd.DataFrame([
                        {
                            "Competitor":   comp,
                            "Avg Price":    f"${report['competitor_prices']['avg_price_by_competitor'].get(comp, 0):.2f}",
                            "Market Share": f"{_COMP.get(comp, {}).get('market_share', 0)*100:.0f}%",
                            "Win Rate vs":  f"{_COMP.get(comp, {}).get('win_rate_vs', 0.5)*100:.0f}%",
                            "Quality":      f"{_COMP.get(comp, {}).get('quality_score', 7):.1f}/10",
                        }
                        for comp in _COMP.keys()
                    ]),
                    striped=True, bordered=True, hover=True,
                ),
            ], style={"marginBottom": "2rem"}),

            # ── Pricing Signals ──────────────────────────────────────────────
            *(
                [html.Div([
                    html.H6("🚨 Recent Pricing Signals",
                            style={"color": YELLOW, "marginBottom": "1rem"}),
                    *[dbc.Card(
                        dbc.CardBody([
                            html.Span(f"{s['competitor']}: ",
                                      style={"fontWeight": "600"}),
                            html.Span(
                                f"{s['magnitude']:.0f}% price "
                                f"{'decrease' if s['magnitude'] < 0 else 'increase'}",
                                style={"color": RED if s['magnitude'] < 0 else GREEN},
                            ),
                            html.Div(
                                f"Confidence: {s['confidence']*100:.0f}%  |  "
                                f"Detected: {_fmt_date(s.get('detected_at'))}",
                                style={"color": MUTED, "fontSize": "0.8rem",
                                       "marginTop": "0.3rem"},
                            ),
                        ]),
                        style={"background": CARD_BG,
                               "border": f"1px solid {BORDER2}",
                               "marginBottom": "0.5rem"},
                    ) for s in signals[:5]],
                ], style={"marginBottom": "2rem"})]
                if signals else []
            ),

            # ── Opportunities ────────────────────────────────────────────────
            *(
                [html.Div([
                    html.H6("🎯 Pricing Opportunities",
                            style={"color": GREEN, "marginBottom": "1rem"}),
                    dbc.Table.from_dataframe(
                        pd.DataFrame(opportunities)[[
                            "product", "type", "current_price",
                            "market_price", "potential_gain", "confidence",
                        ]],
                        striped=True, bordered=True, hover=True,
                    ),
                ], style={"marginBottom": "2rem"})]
                if opportunities else []
            ),

            # ── Recommendations ──────────────────────────────────────────────
            html.Div([
                html.H6("💡 Strategic Recommendations",
                        style={"color": ACCENT3, "marginBottom": "1rem"}),
                html.Ul([
                    html.Li(rec, style={"color": "#fff", "marginBottom": "0.5rem"})
                    for rec in report["recommendations"]
                ]),
            ]),

        ], style={"padding": "1.5rem", "background": CARD_BG,
                  "borderRadius": "8px", "margin": "1rem 0"})

        return screen_ui, {"display": "block", "padding": "1rem 2rem"}, pdf_download

    except Exception as exc:
        err_ui = html.Div([
            html.Span("✗ ", style={"color": RED}),
            html.Span(f"Report generation failed: {exc}"),
        ], style={"padding": "1rem", "background": CARD_BG,
                  "borderRadius": "8px", "color": RED})
        return err_ui, {"display": "block", "padding": "1rem 2rem"}, None

# ============================================================================
# CALLBACK: TAB CONTENT ROUTER (Enhanced)
# ============================================================================
@app.callback(
    Output("tab-content", "children"),
    [Input("active-tab-store", "data"),
     Input("data-store", "data"),
     Input("elasticity-store", "data"),
     Input("win-model-store", "data"),
     Input("competitor-data-store", "data"),
     ],
)
def render_tab(active_tab, df_json, elast_json, win_json, competitor_json, scored_deals_json=None):
    """Render tab content with enhanced features."""
    if not df_json:
        return _empty()
    
    df = pd.read_json(df_json, orient="split")
    df["date"] = pd.to_datetime(df["date"])
    elast = json.loads(elast_json) if elast_json else {}
    win_meta = json.loads(win_json) if win_json else {}

    # Parse competitor data if available
    competitor_data = []
    if competitor_json:
        try:
            competitor_data = json.loads(competitor_json)
        except Exception:
            pass

    # ── 1. Executive Dashboard ─────────────────────────────────────────────
    if active_tab == "📊 Executive Dashboard":
        seg_rev = df.groupby("segment")[["revenue", "cost"]].sum().reset_index()
        seg_rev["profit"] = seg_rev["revenue"] - seg_rev["cost"]
        seg_rev["margin"] = seg_rev["profit"] / seg_rev["revenue"] * 100

        fig_rev = px.bar(
            seg_rev, x="revenue", y="segment", orientation="h",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Revenue by Segment", text_auto=".2s",
        )
        fig_rev.update_layout(**PLOTLY_DARK, height=360, showlegend=False)
        fig_rev.update_traces(marker_line_width=0)

        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby(["month", "segment"])["revenue"].sum().reset_index()
        fig_trend = px.area(
            monthly, x="month", y="revenue", color="segment",
            color_discrete_map=SEGMENT_COLORS, title="Monthly Revenue Trend",
        )
        fig_trend.update_layout(**PLOTLY_DARK, height=360)

        reg_rev = df.groupby("region")["revenue"].sum().reset_index().sort_values("revenue")
        fig_reg = px.bar(
            reg_rev, x="revenue", y="region", orientation="h",
            title="Revenue by Region", color_discrete_sequence=[CYAN],
        )
        fig_reg.update_layout(**PLOTLY_DARK, height=360, showlegend=False)
        fig_reg.update_traces(marker_line_width=0)

        cust_rev = df.groupby("customer_type")["revenue"].sum().reset_index()
        fig_cust = px.pie(
            cust_rev, values="revenue", names="customer_type",
            title="Revenue Mix by Customer Segment", hole=0.5,
        )
        fig_cust.update_layout(**PLOTLY_DARK, height=360)
        fig_cust.update_traces(textposition="outside", textinfo="percent+label")

        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_rev), md=6),
                     dbc.Col(dcc.Graph(figure=fig_trend), md=6)], className="g-3"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_reg), md=6),
                     dbc.Col(dcc.Graph(figure=fig_cust), md=6)], className="g-3 mt-1"),

        ])

    # ── 2. Elasticity Analysis ──────────────────────────────────────────────
    elif active_tab == "🔍 Elasticity Analysis":
        if not elast:
            return _empty("No elasticity results available.")
        
        e_df = pd.DataFrame([{"Segment": k, **v} for k, v in elast.items()])
        e_df = e_df.sort_values("elasticity")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=e_df["elasticity"], y=e_df["Segment"], orientation="h",
            marker=dict(color=e_df["elasticity"], colorscale="RdYlGn_r",
                        showscale=True, colorbar=dict(title="Elasticity")),
            error_x=dict(
                type="data",
                array=(e_df.get("elasticity_ci_hi", e_df["elasticity"]) - e_df["elasticity"]).tolist(),
                arrayminus=(e_df["elasticity"] - e_df.get("elasticity_ci_lo", e_df["elasticity"])).tolist(),
                color=MUTED, thickness=2, width=6,
            ),
            text=[f"{v:.3f}" for v in e_df["elasticity"]], textposition="outside",
        ))
        fig.add_vline(x=-1.0, line_dash="dash", line_color=YELLOW,
                      annotation_text="Unit Elastic", annotation_font_color=YELLOW)
        fig.update_layout(**PLOTLY_DARK, title="Price Elasticity by Segment (with 95% CI)",
                          height=420, xaxis_title="Elasticity Coefficient")

        disp_cols = ["Segment", "elasticity", "elasticity_ci_lo", "elasticity_ci_hi",
                     "p_value", "r_squared", "n_obs", "best_model", "cv_rmse",
                     "mean_price", "mean_margin", "confidence"]
        disp_cols = [c for c in disp_cols if c in e_df.columns]

        fig_r2 = px.bar(
            e_df.sort_values("r_squared"), x="r_squared", y="Segment",
            orientation="h", color="r_squared", color_continuous_scale="Blues",
            title="Model R² by Segment", text_auto=".3f",
        )
        fig_r2.update_layout(**PLOTLY_DARK, height=360, showlegend=False)
        fig_r2.update_traces(marker_line_width=0)

        return html.Div([
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig_r2),
            _section("Detailed Elasticity Table"),
            dbc.Table.from_dataframe(e_df[disp_cols].round(4),
                                     striped=True, bordered=True, hover=True, responsive=True),
        ])

    # ── 3. Optimal Pricing (Enhanced with competitor data) ─────────────────
    elif active_tab == "💡 Optimal Pricing":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Product", style={"color": MUTED, "fontSize": "0.78rem",
                                               "fontWeight": "500", "display": "block",
                                               "marginBottom": "0.3rem"}),
                    dcc.Dropdown(id="opt-product",
                                 options=[{"label": k, "value": k} for k in PRODUCT_CATALOG],
                                 value=list(PRODUCT_CATALOG)[0], clearable=False,
                                 style=_dd_style())
                ], md=4), 
                dbc.Col([
                    html.Label("Optimise For", style={"color": MUTED, "fontSize": "0.78rem",
                                                    "fontWeight": "500", "display": "block",
                                                    "marginBottom": "0.3rem"}),
                    dcc.Dropdown(id="opt-target",
                                 options=[{"label": x, "value": x.lower()}
                                          for x in ["Revenue", "Profit", "Margin"]],
                                 value="revenue", clearable=False, style=_dd_style()),
                ], md=3),
                dbc.Col([
                    html.Label("Strategy", style={"color": MUTED, "fontSize": "0.78rem",
                                                "fontWeight": "500", "display": "block",
                                                "marginBottom": "0.3rem"}),
                    dcc.Dropdown(id="opt-strategy",
                                 options=[{"label": k, "value": k} for k in PRICING_STRATEGIES],
                                 value="Neutral / Market-Rate", clearable=False, style=_dd_style()),
                ], md=3),
                dbc.Col([
                    html.Label("\u00a0", style={"display": "block", "marginBottom": "0.3rem"}),
                    html.Button("Calculate", id="opt-calculate-btn",
                                className="btn btn-primary w-100",
                                style={"height": "38px", "fontFamily": "'Space Grotesk'",
                                        "fontWeight": "600"}),
                ], md=2),
            ], className="g-3"),
            html.Div(id="optimal-result", style={"marginTop": "2rem"}),
        ])

    # ── 4. Revenue Simulator ────────────────────────────────────────────────
    elif active_tab == "📈 Revenue Simulator":
        return html.Div([
            html.P("Monte-Carlo simulation using Sobol low-discrepancy sequences.",
                   style={"color": MUTED, "marginBottom": "1.5rem", "fontSize": "0.88rem"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Product", style={"color": MUTED, "fontSize": "0.78rem",
                                               "fontWeight": "500", "display": "block",
                                               "marginBottom": "0.3rem"}),
                    dcc.Dropdown(id="sim-product",
                                 options=[{"label": k, "value": k} for k in PRODUCT_CATALOG],
                                 value=list(PRODUCT_CATALOG)[0], clearable=False,
                                 style=_dd_style()),
                ], md=4),
                dbc.Col([
                    html.Label("Price Range (% vs base)", style={"color": MUTED, "fontSize": "0.78rem",
                                                                "fontWeight": "500", "display": "block",
                                                                "marginBottom": "0.3rem"}),
                    dcc.RangeSlider(id="sim-price-range", min=-70, max=150, step=5,
                                    marks={i: f"{i}%" for i in range(-70, 151, 35)},
                                    value=[-40, 80], tooltip={"placement": "bottom"}),
                ], md=5),
                dbc.Col([
                    html.Label("Simulations", style={"color": MUTED, "fontSize": "0.78rem",
                                                   "fontWeight": "500", "display": "block",
                                                   "marginBottom": "0.3rem"}),
                    dcc.Dropdown(id="sim-count", options=[512, 1024, 2048, 4096, 8192],
                                 value=2048, style=_dd_style()),
                ], md=3),
            ], className="g-3"),
            html.Div(
                html.Button("▶  Run Simulation", id="sim-run-btn",
                            className="btn btn-success px-5 mt-3",
                            style={"fontFamily": "'Space Grotesk'", "fontWeight": "600"}),
                style={"textAlign": "center", "marginTop": "0.5rem"},
            ),
            html.Div(id="simulation-result", style={"marginTop": "2.5rem"}),
        ])

    # ── 5. Price-Volume Curves ──────────────────────────────────────────────
    elif active_tab == "🎯 Price-Volume Curves":
        prod_opts = [{"label": k, "value": k} for k in PRODUCT_CATALOG]
        first_prod = list(PRODUCT_CATALOG.keys())[0]
        info = PRODUCT_CATALOG[first_prod]
        seg = info["segment"]
        elast_val = elast.get(seg, {}).get("elasticity", -1.2)
        base = info["base_price"]
        cost = info["cost"]
        prices = np.linspace(base * 0.4, base * 2.2, 200)
        volumes = np.maximum(0.01, (prices / base) ** elast_val)
        revenues = prices * volumes
        profits = (prices - cost) * volumes

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue",
                                  line=dict(color=ACCENT, width=2.5)))
        fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit",
                                  line=dict(color=GREEN, width=2.5)))
        fig.add_trace(go.Scatter(x=prices, y=volumes * base / 2, name="Volume (scaled)",
                                  line=dict(color=YELLOW, width=1.5, dash="dot")))
        fig.add_vline(x=base, line_dash="dash", line_color=MUTED,
                      annotation_text="Base Price")
        fig.update_layout(**PLOTLY_DARK, title=f"Price-Volume-Revenue – {first_prod}",
                          height=480, xaxis_title="Price ($)", yaxis_title="Value ($)")

        return html.Div([
            html.Div([
                html.Label("Product", style={"color": MUTED, "fontSize": "0.78rem",
                                           "marginRight": "0.8rem", "fontWeight": "500"}),
                dcc.Dropdown(id="pvc-product", options=prod_opts, value=first_prod,
                             clearable=False, style={**_dd_style(), "width": "380px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "1.5rem"}),
            dcc.Graph(figure=fig, id="pvc-graph"),
        ])

    # ── 6. Competitive Positioning ─────────────────────────────────────────
    elif active_tab == "⚔️ Competitive Positioning":

        # ── Controls row (product selector + strategy selector) ────────────
        return html.Div([
            # ── Filter bar ──────────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Label("Benchmark Product", style={
                        "color": MUTED, "fontSize": "0.75rem", "fontWeight": "500",
                        "display": "block", "marginBottom": "0.3rem",
                    }),
                    dcc.Dropdown(
                        id="cp-product-dd",
                        options=[{"label": k, "value": k} for k in PRODUCT_CATALOG],
                        value=list(PRODUCT_CATALOG.keys())[0],
                        clearable=False, style=_dd_style(),
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("View Mode", style={
                        "color": MUTED, "fontSize": "0.75rem", "fontWeight": "500",
                        "display": "block", "marginBottom": "0.3rem",
                    }),
                    dcc.Dropdown(
                        id="cp-view-dd",
                        options=[
                            {"label": "All Competitors", "value": "all"},
                            {"label": "Direct Rivals Only (±15% price band)", "value": "direct"},
                            {"label": "Premium Segment", "value": "premium"},
                            {"label": "Budget Segment", "value": "budget"},
                        ],
                        value="all", clearable=False, style=_dd_style(),
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Price Scenario", style={
                        "color": MUTED, "fontSize": "0.75rem", "fontWeight": "500",
                        "display": "block", "marginBottom": "0.3rem",
                    }),
                    dcc.Slider(
                        id="cp-price-slider",
                        min=-20, max=30, step=1, value=0,
                        marks={i: f"{i:+d}%" for i in [-20, -10, 0, 10, 20, 30]},
                        tooltip={"placement": "bottom"},
                    ),
                ], md=4),
            ], className="g-3", style={"marginBottom": "1.5rem"}),

            # Placeholder populated by callback
            html.Div(id="cp-content"),
        ])

    # ── 7. Regional Pricing ─────────────────────────────────────────────────
    elif active_tab == "🌍 Regional Pricing":
        reg_df = df.groupby("region").agg(
            revenue=("revenue", "sum"),
            margin=("margin_pct", "mean"),
            win_rate=("deal_won", "mean"),
            volume=("volume", "sum"),
            avg_price=("actual_price", "mean"),
            confidence=("confidence_score", "mean") if "confidence_score" in df.columns else ("revenue", lambda x: 1.0),
        ).reset_index()
        reg_df["win_rate"] *= 100

        fig_rev = px.bar(
            reg_df.sort_values("revenue"), x="revenue", y="region",
            orientation="h", color="margin", color_continuous_scale="RdYlGn",
            title="Revenue & Margin by Region", text_auto=".2s",
        )
        fig_rev.update_layout(**PLOTLY_DARK, height=400, xaxis_title="Revenue ($)")
        fig_rev.update_traces(marker_line_width=0)

        reg_info = pd.DataFrame([
            {"region": r, **{k: v for k, v in d.items() if k != "color"}}
            for r, d in REGIONS.items()
        ])
        fig_growth = px.scatter(
            reg_info, x="competitor_pressure", y="growth_rate",
            size="demand_index", color="region",
            title="Market Attractiveness: Growth vs Competitor Pressure",
            labels={"competitor_pressure": "Competitor Pressure",
                     "growth_rate": "Market Growth Rate"},
        )
        fig_growth.update_layout(**PLOTLY_DARK, height=400)

        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_rev), md=6),
                     dbc.Col(dcc.Graph(figure=fig_growth), md=6)], className="g-3"),
            _section("Regional Performance Summary"),
            dbc.Table.from_dataframe(reg_df.round(2), striped=True, bordered=True,
                                     hover=True, responsive=True),
        ])

    # ── 8. Segment Intelligence ─────────────────────────────────────────────
    elif active_tab == "👥 Segment Intelligence":
        seg_df = df.groupby("customer_type").agg(
            revenue=("revenue", "sum"),
            margin=("margin_pct", "mean"),
            win_rate=("deal_won", "mean"),
            avg_price=("actual_price", "mean"),
            avg_discount=("discount_pct", "mean"),
            deals=("deal_won", "count"),
            confidence=("confidence_score", "mean") if "confidence_score" in df.columns else ("revenue", lambda x: 1.0),
        ).reset_index()
        seg_df["win_rate"] *= 100

        cust_meta = pd.DataFrame([
            {"customer_type": k,
             "loyalty": round(v["loyalty"] * 100, 1),
             "price_sensitivity": round(v["price_sensitivity"] * 100, 1),
             "nps_proxy": v["nps_proxy"],
             "avg_contract_months": v["avg_contract_months"]}
            for k, v in CUSTOMER_SEGMENTS.items()
        ])
        seg_merged = seg_df.merge(cust_meta, on="customer_type", how="left")

        fig_radar_data = []
        for _, row in seg_merged.iterrows():
            fig_radar_data.append(go.Scatterpolar(
                r=[row.get("loyalty", 50), row.get("win_rate", 50),
                   100 - row.get("price_sensitivity", 50),
                   row.get("margin", 30), row.get("nps_proxy", 30)],
                theta=["Loyalty", "Win Rate", "Price Insensitivity",
                        "Margin %", "NPS Proxy"],
                fill="toself", name=row["customer_type"],
            ))
        fig_radar = go.Figure(data=fig_radar_data)
        fig_radar.update_layout(**PLOTLY_DARK, height=480,
                                title="Segment Characteristics Radar",
                                polar=dict(bgcolor=CARD_BG,
                                           radialaxis=dict(visible=True, range=[0, 100])))

        fig_bubble = px.scatter(
            seg_merged, x="avg_discount", y="margin",
            size="revenue", color="customer_type",
            title="Discount vs Margin by Segment (bubble = revenue)",
            labels={"avg_discount": "Avg Discount %", "margin": "Avg Margin %"},
            size_max=60,
        )
        fig_bubble.update_layout(**PLOTLY_DARK, height=420)

        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_radar), md=6),
                      dbc.Col(dcc.Graph(figure=fig_bubble), md=6)], className="g-3"),
            _section("Segment KPI Table"),
            dbc.Table.from_dataframe(seg_merged.round(2), striped=True, bordered=True,
                                     hover=True, responsive=True),
        ])

    # ── 9. What-If Scenarios ────────────────────────────────────────────────
    elif active_tab == "🔧 What-If Scenarios":
        prod_list = list(PRODUCT_CATALOG.keys())
        segments  = sorted({PRODUCT_CATALOG[p]["segment"] for p in prod_list})

        # ── Segment-level global adjustment controls ──────────────────────
        seg_controls = []
        for seg in segments:
            seg_controls.append(
                dbc.Col([
                    html.Div([
                        html.Span(seg, style={"color": ACCENT, "fontSize": "0.78rem",
                                              "fontWeight": "600", "flex": "1"}),
                        html.Span("Segment-wide %Δ",
                                  style={"color": MUTED, "fontSize": "0.68rem"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                               "marginBottom": "0.2rem"}),
                    dcc.Slider(
                        id={"type": "whatif-seg-slider", "index": seg},
                        min=-30, max=30, step=1, value=0,
                        marks={-30: "-30%", -15: "-15%", 0: "0", 15: "+15%", 30: "+30%"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], md=4, style={"marginBottom": "1.2rem"})
            )

        # ── Per-product price sliders ──────────────────────────────────────
        prod_sliders = []
        for prod in prod_list:
            info  = PRODUCT_CATALOG[prod]
            base  = info["base_price"]
            cost  = info["cost"]
            seg   = info["segment"]
            unit  = info.get("unit", "unit")
            base_margin = round((base - cost) / base * 100, 1)

            prod_sliders.append(
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span(prod, style={"color": "#dde5f5", "fontSize": "0.82rem",
                                                   "fontWeight": "600"}),
                            html.Span(f" / {unit}", style={"color": MUTED, "fontSize": "0.72rem"}),
                        ]),
                        html.Div([
                            _badge(seg, SEGMENT_COLORS.get(seg, ACCENT)),
                            html.Span(f"  base ${base:,}",
                                      style={"color": MUTED, "fontSize": "0.72rem",
                                             "marginLeft": "6px"}),
                            html.Span(f"  margin {base_margin}%",
                                      style={"color": CYAN, "fontSize": "0.72rem",
                                             "marginLeft": "6px"}),
                        ], style={"marginTop": "2px"}),
                    ], style={"marginBottom": "0.35rem"}),
                    dcc.Slider(
                        id={"type": "whatif-slider", "index": prod},
                        min=int(info.get("min_price", base * 0.60)),
                        max=int(info.get("max_price", base * 1.60)),
                        step=max(1, int(base * 0.01)),
                        value=int(base),
                        marks={
                            int(info.get("min_price", base * 0.60)): "Min",
                            int(base): "Base",
                            int(info.get("max_price", base * 1.60)): "Max",
                        },
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], md=6, style={
                    "marginBottom": "1.4rem",
                    "padding": "0.9rem 1rem",
                    "background": f"linear-gradient(135deg, {CARD_BG} 0%, #0c1428 100%)",
                    "borderRadius": "10px",
                    "border": f"1px solid {BORDER}",
                })
            )

        # ── Scenario presets ──────────────────────────────────────────────
        preset_btns = html.Div([
            html.Span("Quick presets: ", style={"color": MUTED, "fontSize": "0.78rem",
                                                 "marginRight": "8px", "lineHeight": "2"}),
            html.Button("↑ +10% All",    id="whatif-preset-up10",   n_clicks=0,
                        style={"fontSize": "0.72rem", "padding": "3px 10px",
                               "background": "transparent", "border": f"1px solid {GREEN}",
                               "color": GREEN, "borderRadius": "6px",
                               "cursor": "pointer", "marginRight": "6px"}),
            html.Button("↓ –10% All",    id="whatif-preset-down10", n_clicks=0,
                        style={"fontSize": "0.72rem", "padding": "3px 10px",
                               "background": "transparent", "border": f"1px solid {RED}",
                               "color": RED, "borderRadius": "6px",
                               "cursor": "pointer", "marginRight": "6px"}),
            html.Button("⟳ Reset",       id="whatif-preset-reset",  n_clicks=0,
                        style={"fontSize": "0.72rem", "padding": "3px 10px",
                               "background": "transparent", "border": f"1px solid {MUTED}",
                               "color": MUTED, "borderRadius": "6px",
                               "cursor": "pointer"}),
        ], style={"marginBottom": "1.2rem", "display": "flex", "alignItems": "center",
                  "flexWrap": "wrap", "gap": "4px"})

        # ── Scenario name input ───────────────────────────────────────────
        scenario_input = dbc.Row([
            dbc.Col([
                html.Label("Scenario Name", style={"color": MUTED, "fontSize": "0.78rem",
                                                    "marginBottom": "4px", "display": "block"}),
                dcc.Input(
                    id="whatif-scenario-name",
                    type="text",
                    placeholder="e.g. Q3 Price Hike, Competitive Response…",
                    debounce=False,
                    style={**_input_style(), "fontSize": "0.82rem"},
                ),
            ], md=5),
            dbc.Col([
                html.Label("Objective Focus", style={"color": MUTED, "fontSize": "0.78rem",
                                                      "marginBottom": "4px", "display": "block"}),
                dcc.Dropdown(
                    id="whatif-objective",
                    options=[
                        {"label": "Revenue Maximisation", "value": "revenue"},
                        {"label": "Profit / Margin",      "value": "profit"},
                        {"label": "Win-Rate Defence",     "value": "win_rate"},
                        {"label": "Market Share Gain",    "value": "market_share"},
                    ],
                    value="revenue",
                    clearable=False,
                    style=_dd_style(),
                ),
            ], md=4),
            dbc.Col([
                html.Label("Segment Filter", style={"color": MUTED, "fontSize": "0.78rem",
                                                     "marginBottom": "4px", "display": "block"}),
                dcc.Dropdown(
                    id="whatif-seg-filter",
                    options=[{"label": "All Segments", "value": "all"}]
                            + [{"label": s, "value": s} for s in segments],
                    value="all",
                    clearable=False,
                    style=_dd_style(),
                ),
            ], md=3),
        ], className="g-2", style={"marginBottom": "1.2rem"})

        return html.Div([
            # Header
            html.Div([
                html.H5("What-If Scenario Modeller",
                        style={"color": "#e8edf5", "fontFamily": "'Space Grotesk'",
                               "fontWeight": "700", "margin": 0}),
                html.P(
                    "Adjust prices per product or per segment, pick an objective, "
                    "then run the scenario to see full portfolio impact with break-even "
                    "analysis and waterfall visualisation.",
                    style={"color": MUTED, "fontSize": "0.84rem",
                           "margin": "0.4rem 0 0"},
                ),
            ], style={"marginBottom": "1.5rem"}),

            # Scenario configuration row
            scenario_input,

            # Segment-wide adjusters
            _section("① Segment-Wide Price Adjustment (apply to all products in segment)"),
            dbc.Row(seg_controls, className="g-3"),

            # Per-product sliders
            _section("② Individual Product Price Sliders"),
            preset_btns,
            dbc.Row(prod_sliders, className="g-3"),

            # Apply button
            html.Div([
                html.Button(
                    [html.Span("▶ ", style={"marginRight": "6px"}), "Run Scenario Analysis"],
                    id="whatif-apply-btn", n_clicks=0,
                    style={
                        "padding": "0.75rem 2.5rem",
                        "background": f"linear-gradient(135deg, {ACCENT} 0%, #2a6dd6 100%)",
                        "border": "none", "borderRadius": "8px",
                        "color": "#fff", "fontWeight": "700", "fontSize": "0.9rem",
                        "cursor": "pointer", "letterSpacing": "0.4px",
                        "boxShadow": f"0 4px 16px rgba(79,158,255,0.35)",
                        "fontFamily": "'Space Grotesk', sans-serif",
                        "transition": "all 0.2s",
                    },
                ),
                html.Span("  Changes are not applied to the model until you click Run",
                          style={"color": MUTED, "fontSize": "0.75rem", "marginLeft": "14px"}),
            ], style={"margin": "1.5rem 0 0.5rem"}),

            dcc.Loading(
                type="circle", color=ACCENT,
                children=html.Div(id="whatif-result", style={"marginTop": "1.5rem"}),
            ),
        ])

    # ── 10. Win-Rate Analysis ───────────────────────────────────────────────
    elif active_tab == "🤝 Win-Rate Analysis":
        fig_heat = px.density_heatmap(
            df, x="discount_pct", y="actual_price", z="deal_won",
            nbinsx=20, nbinsy=20, histfunc="avg",
            title="Win Rate Heat Map: Discount % vs Price",
            color_continuous_scale="RdYlGn", labels={"deal_won": "Win Rate"},
        )
        fig_heat.update_layout(**PLOTLY_DARK, height=440)

        win_by_seg = (
            df.groupby("segment")["deal_won"].mean() * 100
        ).reset_index().sort_values("deal_won")
        win_by_seg.columns = ["segment", "win_rate"]

        fig_seg = px.bar(
            win_by_seg, x="win_rate", y="segment", orientation="h",
            title="Average Win Rate by Product Segment",
            color="win_rate", color_continuous_scale="Greens", text_auto=".1f",
        )
        fig_seg.update_layout(**PLOTLY_DARK, height=360, showlegend=False,
                              xaxis_title="Win Rate (%)")
        fig_seg.update_traces(marker_line_width=0)

        fi = win_meta.get("feature_importance", {})
        fi_children = []
        if fi:
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
            fig_fi = px.bar(
                fi_df.sort_values("Importance"), x="Importance", y="Feature",
                orientation="h",
                title=f"Feature Importance ({win_meta.get('model_name','')})",
                color="Importance", color_continuous_scale="Blues",
            )
            fig_fi.update_layout(**PLOTLY_DARK, height=340)
            fi_children = [dcc.Graph(figure=fig_fi)]

        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_heat), md=7),
                     dbc.Col(dcc.Graph(figure=fig_seg), md=5)], className="g-3"),
            *fi_children,
            _section(f"Win-Rate Model (AUC: {win_meta.get('auc','—')})"),
            html.P(f"Best model: {win_meta.get('model_name','—')} | "
                   f"All AUCs: {win_meta.get('all_auc',{})}",
                   style={"color": MUTED, "fontSize": "0.85rem"}),
        ])

    # ── 11. Margin Waterfall ────────────────────────────────────────────────
    elif active_tab == "📉 Margin Waterfall":
        wf_df = MarginWaterfallBuilder.build(df)
        vals = wf_df["value"].fillna(0).tolist()
        labels = wf_df["label"].tolist()
        measure = wf_df["measure"].tolist()

        fig_wf = go.Figure(go.Waterfall(
            name="Margin", orientation="v",
            measure=measure, x=labels, y=vals,
            connector={"line": {"color": BORDER2}},
            decreasing={"marker": {"color": RED}},
            increasing={"marker": {"color": GREEN}},
            totals={"marker": {"color": ACCENT}},
        ))
        fig_wf.update_layout(**PLOTLY_DARK, height=500,
                             title="Portfolio Margin Waterfall", yaxis_title="USD ($)")

        seg_margin = df.groupby("segment").agg(
            revenue=("revenue", "sum"), cost=("cost", "sum"),
        ).reset_index()
        seg_margin["gross_profit"] = seg_margin["revenue"] - seg_margin["cost"]
        seg_margin["margin_pct"] = seg_margin["gross_profit"] / seg_margin["revenue"] * 100

        fig_seg_m = px.bar(
            seg_margin, x="segment", y="margin_pct",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Gross Margin % by Segment", text_auto=".1f",
        )
        fig_seg_m.update_layout(**PLOTLY_DARK, height=360, showlegend=False,
                                yaxis_title="Margin (%)")
        fig_seg_m.update_traces(marker_line_width=0)

        return html.Div([
            dcc.Graph(figure=fig_wf),
            dcc.Graph(figure=fig_seg_m),
        ])

    # ── 12. Product Portfolio ───────────────────────────────────────────────
    elif active_tab == "📦 Product Portfolio":
        scored = PortfolioScorer.score_products(df, elast, competitor_data)
        if scored.empty:
            return _empty("Insufficient data for portfolio scoring.")

        fig_matrix = px.scatter(
            scored, x="margin_pct", y="growth_pct",
            size="revenue", color="adjusted_score", color_continuous_scale="Viridis",
            hover_name="product",
            title="Portfolio Matrix: Margin vs Growth (bubble = revenue, color = score)",
            labels={"margin_pct": "Gross Margin %", "growth_pct": "Revenue Growth %"},
            size_max=50,
        )
        fig_matrix.add_hline(y=0, line_dash="dot", line_color=MUTED)
        fig_matrix.add_vline(x=scored["margin_pct"].median(), line_dash="dot", line_color=MUTED,
                             annotation_text="Median Margin")
        fig_matrix.update_layout(**PLOTLY_DARK, height=520)

        fig_score = px.bar(
            scored.sort_values("adjusted_score"), x="adjusted_score", y="product",
            orientation="h", color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Product Health Score (0–100)", text_auto=".1f",
        )
        fig_score.update_layout(**PLOTLY_DARK, height=max(380, len(scored) * 28))
        fig_score.update_traces(marker_line_width=0)

        return html.Div([
            dcc.Graph(figure=fig_matrix),
            dcc.Graph(figure=fig_score),
            _section("Detailed Portfolio Table"),
            dbc.Table.from_dataframe(scored.round(2), striped=True, bordered=True,
                                     hover=True, responsive=True),
        ])

    # ── 13. Risk & Sensitivity ──────────────────────────────────────────────
    elif active_tab == "⚠️ Risk & Sensitivity":
        base_profit = (df["revenue"] - df["cost"]).sum()
        tornado = []
        for label, mult in {
            "Price +10%": 1.10, "Price −10%": 0.90,
            "Volume +10%": 1.10, "Volume −10%": 0.90,
            "Cost +10%": 1.00, "Cost −10%": 1.00,
        }.items():
            if "Price" in label or "Volume" in label:
                p = base_profit * mult
            else:
                cost_adj = df["cost"].sum() * (1.10 if "+10%" in label else 0.90)
                p = df["revenue"].sum() - cost_adj
            tornado.append({"Driver": label, "Profit Impact": round(p - base_profit, 2)})

        t_df = pd.DataFrame(tornado).sort_values("Profit Impact")
        fig_tornado = px.bar(
            t_df, x="Profit Impact", y="Driver", orientation="h",
            title="Tornado Sensitivity (±10% on key drivers)",
            color="Profit Impact", color_continuous_scale="RdYlGn",
        )
        fig_tornado.add_vline(x=0, line_color=MUTED)
        fig_tornado.update_layout(**PLOTLY_DARK, height=400, showlegend=False)
        fig_tornado.update_traces(marker_line_width=0)

        fig_sens = go.Figure()
        if elast:
            e_sens = pd.DataFrame([
                {"Segment": k, "Elasticity": v["elasticity"],
                 "R²": v["r_squared"], "P-value": v.get("p_value", np.nan),
                 "Observations": v["n_obs"], "Confidence": v.get("confidence", 0.5)}
                for k, v in elast.items()
            ])
            fig_sens = px.scatter(
                e_sens, x="Observations", y="R²",
                size=e_sens["Confidence"] * 50, color="Segment",
                title="Elasticity Reliability: R² vs Sample Size",
                hover_data=["P-value", "Elasticity", "Confidence"],
            )
        fig_sens.update_layout(**PLOTLY_DARK, height=400)

        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_tornado), md=6),
                     dbc.Col(dcc.Graph(figure=fig_sens), md=6)], className="g-3"),
        ])

    # ── 14. Seasonality & Trends ────────────────────────────────────────────
    elif active_tab == "🗓️ Seasonality & Trends":
        try:
            df2 = df.copy()
            df2["month_num"] = df2["date"].dt.month
            df2["year"] = df2["date"].dt.year
            monthly_agg = df2.groupby(["month_num", "segment"])["revenue"].mean().reset_index()
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                           7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            monthly_agg["month_name"] = monthly_agg["month_num"].map(month_names)
            month_order = list(month_names.values())

            fig_season = px.line(
                monthly_agg, x="month_name", y="revenue",
                color="segment", color_discrete_map=SEGMENT_COLORS,
                title="Average Monthly Revenue by Segment (Seasonality Profile)",
                markers=True,
                category_orders={"month_name": month_order},
            )
            _season_layout = {**PLOTLY_DARK}
            _season_layout["xaxis"] = {**_season_layout.get("xaxis", {}),
                                         "categoryorder": "array",
                                         "categoryarray": month_order}
            fig_season.update_layout(**_season_layout, height=440)

            season_df = pd.DataFrame([
                {"month": month_names[m], "index": v, "month_num": m}
                for m, v in MONTHLY_SEASONALITY.items()
            ]).sort_values("month_num")
            fig_s2 = px.bar(
                season_df, x="month", y="index", color="index",
                color_continuous_scale="RdYlGn",
                title="Demand Seasonality Index (catalog)", text_auto=".2f",
                category_orders={"month": month_order},
            )
            fig_s2.add_hline(y=1.0, line_dash="dash", line_color=MUTED)
            _s2_layout = {**PLOTLY_DARK}
            _s2_layout["xaxis"] = {**_s2_layout.get("xaxis", {}),
                                     "categoryorder": "array",
                                     "categoryarray": month_order}
            fig_s2.update_layout(**_s2_layout, height=360, showlegend=False,
                                  yaxis_title="Demand Index")
            fig_s2.update_traces(marker_line_width=0)

            return html.Div([
                dcc.Graph(figure=fig_season),
                dcc.Graph(figure=fig_s2),
            ])
        except Exception as exc:
            return html.Div([
                html.P(f"⚠ Error rendering Seasonality & Trends: {exc}",
                       style={"color": RED, "padding": "2rem"}),
            ])

    # ── 15. Revenue Forecast ────────────────────────────────────────────────
    elif active_tab == "🔮 Revenue Forecast":
        try:
            forecast_df = RevenueForecaster.forecast(df, horizon_months=12)
            fig_fc = go.Figure()
            for seg in forecast_df["segment"].unique():
                sub_h = forecast_df[(forecast_df["segment"] == seg) &
                                      (forecast_df["type"] == "historical")]
                sub_f = forecast_df[(forecast_df["segment"] == seg) &
                                      (forecast_df["type"] == "forecast")]
                color = SEGMENT_COLORS.get(seg, ACCENT)
                fig_fc.add_trace(go.Scatter(
                    x=sub_h["month"], y=sub_h["forecast"],
                    name=f"{seg} (actual)", line=dict(color=color, width=2),
                ))
                if not sub_f.empty:
                    fig_fc.add_trace(go.Scatter(
                        x=sub_f["month"], y=sub_f["forecast"],
                        name=f"{seg} (forecast)",
                        line=dict(color=color, width=2, dash="dash"),
                    ))
                    # Build fill color safely
                    if color.startswith("rgb"):
                        fill_color = color.replace(")", ",0.10)").replace("rgb", "rgba")
                    elif color.startswith("#"):
                        h = color.lstrip("#")
                        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                        fill_color = f"rgba({r},{g},{b},0.125)"
                    else:
                        fill_color = color
                    fig_fc.add_trace(go.Scatter(
                        x=pd.concat([sub_f["month"], sub_f["month"].iloc[::-1]]),
                        y=pd.concat([sub_f["ci_hi"], sub_f["ci_lo"].iloc[::-1]]),
                        fill="toself",
                        fillcolor=fill_color,
                        line=dict(color="rgba(0,0,0,0)"),
                        name=f"{seg} CI", showlegend=False,
                    ))
            fig_fc.update_layout(**PLOTLY_DARK, height=540,
                                 title="12-Month Revenue Forecast by Segment (90% CI)",
                                 xaxis_title="Month", yaxis_title="Revenue ($)")

            forecast_only = forecast_df[forecast_df["type"] == "forecast"]
            next3 = forecast_only.groupby("segment").head(3) if not forecast_only.empty else forecast_only
            preview_cols = ["segment", "month", "forecast", "ci_lo", "ci_hi", "method"]
            table_df = next3[preview_cols].round(2) if not next3.empty else pd.DataFrame(columns=preview_cols)

            return html.Div([
                dcc.Graph(figure=fig_fc),
                _section("Next-3-Month Forecast Preview"),
                dbc.Table.from_dataframe(
                    table_df,
                    striped=True, bordered=True, hover=True, responsive=True,
                ),
            ])
        except Exception as exc:
            return html.Div([
                html.P(f"⚠ Error rendering Revenue Forecast: {exc}",
                       style={"color": RED, "padding": "2rem"}),
            ])

    # ── 17. Market Intelligence (New Tab) ───────────────────────────────────
    elif active_tab == "🕷️ Market Intelligence":
        # This tab will show the intelligence report
        if not competitor_data:
            return html.Div([
                html.P("No market intelligence data available. Click 'Scrape Competitor Prices' in the sidebar to gather data.",
                       style={"color": MUTED, "textAlign": "center", "padding": "3rem"}),
            ])
        
        # Trigger report generation
        return html.Div(id="market-intel-tab-content")

    return html.Div([
        html.H5(active_tab, style={"color": ACCENT}),
        html.P("Module loaded. Build the model to see data.", style={"color": MUTED}),
    ])

# ============================================================================
# CALLBACK: COMPETITIVE POSITIONING PANEL
# ============================================================================
@app.callback(
    Output("cp-content", "children"),
    [Input("cp-product-dd", "value"),
     Input("cp-view-dd", "value"),
     Input("cp-price-slider", "value")],
    [State("data-store", "data"),
     State("competitor-data-store", "data"),
     State("elasticity-store", "data")],
)
def render_competitive_positioning(product, view_mode, price_delta_pct,
                                   df_json, competitor_json, elast_json):
    if not product:
        raise PreventUpdate

    info       = PRODUCT_CATALOG[product]
    base_price = info["base_price"]
    cost       = info["cost"]
    seg        = info["segment"]
    straive_q  = 8.2  # internal quality benchmark

    # Apply price scenario
    scenario_price = base_price * (1 + price_delta_pct / 100)

    # Competitor data
    competitor_data = json.loads(competitor_json) if competitor_json else []
    comp_result     = CompetitiveAnalyzer.get_price_score(
        scenario_price, base_price, competitor_data
    )
    c_df = comp_result["df"].copy()
    market_pos = comp_result.get("market_position", {})

    # Elasticity
    elast     = json.loads(elast_json) if elast_json else {}
    elast_val = elast.get(seg, {}).get("elasticity", -1.35)

    # Filter by view mode
    if view_mode == "direct":
        c_df = c_df[c_df["Comp Relative"].between(0.85, 1.15)]
    elif view_mode == "premium":
        c_df = c_df[c_df["Comp Relative"] >= 1.0]
    elif view_mode == "budget":
        c_df = c_df[c_df["Comp Relative"] < 1.0]

    if c_df.empty:
        return html.P("No competitors match the selected filter.",
                      style={"color": MUTED, "padding": "2rem"})

    # ── Derived metrics ─────────────────────────────────────────────────────
    avg_comp_price = c_df["Comp Price"].mean()
    straive_rel    = comp_result["straive_relative"]
    n_cheaper      = int((c_df["Comp Relative"] < straive_rel).sum())
    n_costlier     = int((c_df["Comp Relative"] > straive_rel).sum())
    avg_gap        = float(c_df["Gap %"].mean())
    avg_win        = float(c_df["Win Rate vs"].mean())
    value_score_straive = straive_q / straive_rel if straive_rel > 0 else 0
    value_vs_avg   = value_score_straive - c_df["Value Score"].mean()

    # ── 1. KPI bar ──────────────────────────────────────────────────────────
    kpi_color_gap  = GREEN if avg_gap > 0 else RED
    kpi_color_win  = GREEN if avg_win >= 60 else (YELLOW if avg_win >= 45 else RED)

    kpi_bar = dbc.Row([
        _card("Scenario Price",   f"${scenario_price:,.0f}",
              GREEN if price_delta_pct >= 0 else RED,
              f"{price_delta_pct:+d}% vs base"),
        _card("Avg Comp Price",   f"${avg_comp_price:,.0f}", CYAN),
        _card("Price Gap (avg)",  f"{avg_gap:+.1f}%",  kpi_color_gap,
              "vs competitors"),
        _card("Avg Win Rate",     f"{avg_win:.1f}%",   kpi_color_win),
        _card("Cheaper Rivals",   str(n_cheaper),      YELLOW if n_cheaper > 0 else GREEN),
        _card("Value Score Δ",    f"{value_vs_avg:+.2f}", GREEN if value_vs_avg > 0 else RED,
              "STRAIVE vs comp avg"),
    ], className="g-3 mb-3")

    # ── 2. Strategic Positioning Matrix ────────────────────────────────────
    fig_pos = go.Figure()

    # Quadrant shading
    avg_q   = float(c_df["Quality Score"].mean())
    fig_pos.add_shape(type="rect", x0=0, x1=1.0, y0=avg_q, y1=11,
                      fillcolor="rgba(46,232,154,0.04)", line_width=0)
    fig_pos.add_shape(type="rect", x0=1.0, x1=2.2, y0=avg_q, y1=11,
                      fillcolor="rgba(79,158,255,0.04)", line_width=0)
    fig_pos.add_shape(type="rect", x0=0, x1=1.0, y0=0, y1=avg_q,
                      fillcolor="rgba(255,64,96,0.04)", line_width=0)
    fig_pos.add_shape(type="rect", x0=1.0, x1=2.2, y0=0, y1=avg_q,
                      fillcolor="rgba(245,166,35,0.04)", line_width=0)

    # Quadrant labels
    for label, x, y, col in [
        ("Low Price · High Quality", 0.5,  10.5, GREEN),
        ("High Price · High Quality", 1.6, 10.5, ACCENT),
        ("Low Price · Low Quality",  0.5,  4.5,  RED),
        ("High Price · Low Quality", 1.6,  4.5,  YELLOW),
    ]:
        fig_pos.add_annotation(x=x, y=y, text=label,
                               font=dict(size=8, color=col), showarrow=False,
                               opacity=0.5)

    # Bubble size = market share, color by win rate
    sizes = [max(14, COMPETITORS.get(r, {}).get("market_share", 0.07) * 300)
             for r in c_df["Competitor"]]

    fig_pos.add_trace(go.Scatter(
        x=c_df["Comp Relative"],
        y=c_df["Quality Score"],
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=c_df["Win Rate vs"],
            colorscale=[[0, RED], [0.45, YELLOW], [0.7, GREEN], [1, "#00ffaa"]],
            colorbar=dict(title="Win Rate %", thickness=12, len=0.6),
            cmin=0, cmax=100,
            symbol="circle",
            line=dict(color="rgba(255,255,255,0.4)", width=1.5),
            showscale=True,
        ),
        text=c_df["Competitor"],
        textposition="top center",
        textfont=dict(size=9, color="#c8d8f0"),
        name="Competitors",
        customdata=np.stack([
            c_df["Comp Price"], c_df["Gap %"],
            c_df["Value Score"], c_df["Win Rate vs"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Price: $%{customdata[0]:,.0f}  (%{x:.2f}× base)<br>"
            "Gap vs STRAIVE: %{customdata[1]:+.1f}%<br>"
            "Quality: %{y:.1f}/10<br>"
            "Value Score: %{customdata[2]:.2f}<br>"
            "Win Rate: %{customdata[3]:.1f}%<extra></extra>"
        ),
    ))

    # STRAIVE star
    fig_pos.add_trace(go.Scatter(
        x=[straive_rel], y=[straive_q],
        mode="markers+text",
        marker=dict(size=28, color=GREEN, symbol="star",
                    line=dict(color="#fff", width=2)),
        text=["STRAIVE"], textposition="top center",
        textfont=dict(size=11, color=GREEN, family="'Space Grotesk', sans-serif"),
        name="STRAIVE",
        hovertemplate=(
            f"<b>STRAIVE</b><br>Price: ${scenario_price:,.0f}<br>"
            f"Quality: {straive_q}/10<br>"
            f"Value Score: {value_score_straive:.2f}<extra></extra>"
        ),
    ))

    # Reference lines
    fig_pos.add_hline(y=avg_q, line_dash="dot", line_color=MUTED,
                      annotation_text=f"Avg Quality ({avg_q:.1f})",
                      annotation_font_color=MUTED, annotation_font_size=9)
    fig_pos.add_vline(x=1.0, line_dash="dot", line_color=MUTED,
                      annotation_text="STRAIVE Base", annotation_font_color=MUTED,
                      annotation_font_size=9)

    fig_pos.update_layout(
        **PLOTLY_DARK,
        title=dict(
            text=f"Strategic Positioning Matrix — {product}  "
                 f"<span style='font-size:12px;color:{MUTED}'>bubble size = market share</span>",
            # NOTE: title-level font is separate from the top-level "font" key in PLOTLY_DARK
            font=dict(size=14),
        ),
        height=500,
    )
    # Override legend separately to avoid duplicate-keyword clash with PLOTLY_DARK["legend"]
    fig_pos.update_layout(legend=dict(orientation="h", y=-0.12))
    fig_pos.update_xaxes(title="Relative Price  (1.0 = STRAIVE base)", range=[0.55, 1.7])
    fig_pos.update_yaxes(title="Quality Score  (1–10)", range=[5.5, 11])

    # ── 3. Win Rate Breakdown ───────────────────────────────────────────────
    c_sorted = c_df.sort_values("Win Rate vs")
    bar_colors = [
        GREEN if v >= 65 else (YELLOW if v >= 50 else RED)
        for v in c_sorted["Win Rate vs"]
    ]

    fig_win = go.Figure()
    fig_win.add_trace(go.Bar(
        x=c_sorted["Win Rate vs"], y=c_sorted["Competitor"],
        orientation="h",
        marker=dict(color=bar_colors, line_width=0),
        text=[f"{v:.1f}%" for v in c_sorted["Win Rate vs"]],
        textposition="outside",
        textfont=dict(color="#c8d8f0", size=9),
        hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>",
    ))
    fig_win.add_vline(x=50, line_dash="dash", line_color=MUTED,
                      annotation_text="50% breakeven", annotation_font_size=9)
    fig_win.add_vline(x=avg_win, line_dash="dot", line_color=CYAN,
                      annotation_text=f"Avg {avg_win:.1f}%",
                      annotation_font_color=CYAN, annotation_font_size=9)
    fig_win.update_layout(
        **PLOTLY_DARK,
        title="Win Rate vs Each Competitor",
        height=370,
        showlegend=False,
    )
    fig_win.update_xaxes(title="Win Rate (%)", range=[0, 105])
    fig_win.update_yaxes(title="")

    # ── 4. Price Gap Waterfall ──────────────────────────────────────────────
    wf_sorted = c_df.sort_values("Gap %")
    gap_colors = [GREEN if v > 0 else RED for v in wf_sorted["Gap %"]]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        x=wf_sorted["Competitor"], y=wf_sorted["Gap %"],
        marker=dict(color=gap_colors, line_width=0),
        text=[f"{v:+.1f}%" for v in wf_sorted["Gap %"]],
        textposition="outside",
        textfont=dict(color="#c8d8f0", size=9),
        hovertemplate="<b>%{x}</b><br>STRAIVE is %{y:+.1f}% vs this competitor<extra></extra>",
    ))
    fig_gap.add_hline(y=0, line_color=MUTED, line_width=1)
    fig_gap.update_layout(
        **PLOTLY_DARK,
        title="Price Gap: STRAIVE vs Competitors  (+ve = STRAIVE is more expensive)",
        height=340,
        showlegend=False,
    )
    fig_gap.update_yaxes(title="Price Gap (%)")
    fig_gap.update_xaxes(title="")

    # ── 5. Value Map (Price vs Value Score) ─────────────────────────────────
    vs_all = pd.concat([
        c_df[["Competitor", "Comp Relative", "Value Score", "Quality Score",
              "Win Rate vs"]].rename(columns={
                  "Comp Relative": "rel_price",
                  "Value Score": "value_score",
                  "Quality Score": "quality",
                  "Win Rate vs": "win_rate",
              }).assign(entity=c_df["Competitor"]),
    ], ignore_index=True)

    fig_val = go.Figure()
    for _, row in vs_all.iterrows():
        col = COMPETITOR_COLORS.get(row["entity"], ACCENT2)
        fig_val.add_trace(go.Scatter(
            x=[row["rel_price"]], y=[row["value_score"]],
            mode="markers+text",
            marker=dict(size=14, color=col,
                        line=dict(color="rgba(255,255,255,0.3)", width=1)),
            text=[row["entity"]], textposition="top center",
            textfont=dict(size=8, color="#c8d8f0"),
            name=row["entity"],
            showlegend=True,
            hovertemplate=f"<b>{row['entity']}</b><br>Rel Price: {row['rel_price']:.2f}<br>"
                          f"Value Score: {row['value_score']:.2f}<br>"
                          f"Win Rate: {row['win_rate']:.1f}%<extra></extra>",
        ))

    # STRAIVE on value map
    fig_val.add_trace(go.Scatter(
        x=[straive_rel], y=[value_score_straive],
        mode="markers+text",
        marker=dict(size=22, color=GREEN, symbol="star",
                    line=dict(color="#fff", width=2)),
        text=["STRAIVE"], textposition="top center",
        textfont=dict(size=10, color=GREEN),
        name="STRAIVE", showlegend=True,
        hovertemplate=f"<b>STRAIVE</b><br>Rel Price: {straive_rel:.2f}<br>"
                      f"Value Score: {value_score_straive:.2f}<extra></extra>",
    ))

    # Iso-value line (value = quality/price = constant)
    iso_x = np.linspace(0.6, 1.8, 100)
    iso_y = straive_q / iso_x
    fig_val.add_trace(go.Scatter(
        x=iso_x, y=iso_y, mode="lines",
        line=dict(color=MUTED, dash="dot", width=1),
        name="Iso-value (STRAIVE quality)",
        showlegend=True,
        hoverinfo="skip",
    ))

    fig_val.update_layout(
        **PLOTLY_DARK,
        title="Value Map  (Value Score = Quality ÷ Relative Price)",
        height=400,
    )
    fig_val.update_layout(legend=dict(orientation="h", y=-0.18, font=dict(size=8)))
    fig_val.update_xaxes(title="Relative Price", range=[0.55, 1.8])
    fig_val.update_yaxes(title="Value Score")

    # ── 6. Radar: STRAIVE vs top 3 rivals ──────────────────────────────────
    radar_metrics = ["Win Rate vs", "Quality Score", "Value Score", "Confidence"]
    # Normalise each metric 0→100 across all competitors + STRAIVE
    norm_rows = []
    for _, row in c_df.iterrows():
        norm_rows.append({
            "entity": row["Competitor"],
            "Win Rate vs":  row["Win Rate vs"],
            "Quality Score": row["Quality Score"] * 10,   # scale to 100
            "Value Score":   min(row["Value Score"] * 10, 100),
            "Confidence":    row.get("Confidence", 0.75) * 100,
        })
    norm_rows.append({
        "entity": "STRAIVE",
        "Win Rate vs":  avg_win,
        "Quality Score": straive_q * 10,
        "Value Score":   min(value_score_straive * 10, 100),
        "Confidence":    85.0,
    })
    radar_df = pd.DataFrame(norm_rows)

    # Pick STRAIVE + top 3 by market share to keep chart legible
    top3 = sorted(COMPETITORS, key=lambda c: COMPETITORS[c].get("market_share", 0), reverse=True)[:3]
    radar_entities = ["STRAIVE"] + [t for t in top3 if t in radar_df["entity"].values]
    radar_colors   = [GREEN] + [COMPETITOR_COLORS.get(t, ACCENT2) for t in radar_entities[1:]]
    radar_dims     = ["Win Rate vs", "Quality Score", "Value Score", "Confidence"]

    fig_radar = go.Figure()
    for ent, col in zip(radar_entities, radar_colors):
        row = radar_df[radar_df["entity"] == ent]
        if row.empty:
            continue
        vals = [float(row.iloc[0][d]) for d in radar_dims] + [float(row.iloc[0][radar_dims[0]])]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=radar_dims + [radar_dims[0]],
            fill="toself",
            fillcolor=(f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.08)"
                       if col.startswith("#") and len(col) == 7
                       else "rgba(79,158,255,0.08)"),
            line=dict(color=col, width=2),
            name=ent,
        ))
    fig_radar.update_layout(
        **PLOTLY_DARK,
        title="Multi-Dimension Radar: STRAIVE vs Top 3 Rivals",
        height=400,
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=8, color=MUTED),
                            gridcolor=BORDER2),
            angularaxis=dict(tickfont=dict(size=9, color="#c8d8f0"),
                             gridcolor=BORDER2),
        ),
    )
    fig_radar.update_layout(legend=dict(orientation="h", y=-0.12, font=dict(size=9)))

    # ── 7. Price sensitivity: STRAIVE revenue curve with comp overlays ──────
    prices_arr = np.linspace(base_price * 0.5, base_price * 1.8, 200)
    volumes    = np.maximum(0.01, (prices_arr / base_price) ** elast_val)
    rev_arr    = prices_arr * volumes
    prof_arr   = (prices_arr - cost) * volumes

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=prices_arr, y=rev_arr, name="STRAIVE Revenue",
        line=dict(color=ACCENT, width=2.5),
    ))
    fig_sens.add_trace(go.Scatter(
        x=prices_arr, y=prof_arr, name="STRAIVE Profit",
        line=dict(color=GREEN, width=2, dash="dash"),
    ))

    # Competitor price verticals
    for _, row in c_df.iterrows():
        col = COMPETITOR_COLORS.get(row["Competitor"], MUTED)
        fig_sens.add_vline(
            x=row["Comp Price"],
            line_dash="dot", line_color=col, line_width=1,
            annotation_text=row["Competitor"][:8],
            annotation_font=dict(size=8, color=col),
            annotation_position="top",
        )

    # Scenario price marker
    fig_sens.add_vline(
        x=scenario_price, line_dash="solid", line_color=YELLOW, line_width=2,
        annotation_text=f"Scenario ${scenario_price:,.0f}",
        annotation_font=dict(size=9, color=YELLOW),
    )
    fig_sens.add_vline(
        x=base_price, line_dash="dot", line_color=MUTED, line_width=1,
        annotation_text="Base",
        annotation_font=dict(size=9, color=MUTED),
    )

    fig_sens.update_layout(
        **PLOTLY_DARK,
        title=f"Price Sensitivity Curve with Competitor Price Anchors — {product}",
        height=400,
    )
    fig_sens.update_layout(legend=dict(orientation="h", y=-0.15, font=dict(size=9)))
    fig_sens.update_xaxes(title="Price ($)")
    fig_sens.update_yaxes(title="Revenue / Profit ($)")

    # ── 8. Detailed benchmarks table ────────────────────────────────────────
    disp_df = c_df[["Competitor", "Comp Price", "STRAIVE Price", "Gap %",
                    "Quality Score", "Value Score", "Win Rate vs",
                    "Confidence", "Source"]].copy()
    disp_df["Comp Price"]    = disp_df["Comp Price"].map("${:,.0f}".format)
    disp_df["STRAIVE Price"] = disp_df["STRAIVE Price"].map("${:,.0f}".format)
    disp_df["Gap %"]         = disp_df["Gap %"].map("{:+.1f}%".format)
    disp_df["Win Rate vs"]   = disp_df["Win Rate vs"].map("{:.1f}%".format)
    disp_df["Confidence"]    = disp_df["Confidence"].map("{:.0%}".format)

    # Colour-code strengths/weaknesses inline
    def _strength_badges(comp_name):
        info = COMPETITORS.get(comp_name, {})
        s = info.get("strengths", [])
        w = info.get("weaknesses", [])
        parts = [html.Span(x, style={
            "background": "rgba(46,232,154,0.12)", "color": GREEN,
            "borderRadius": "10px", "padding": "1px 6px",
            "fontSize": "0.68rem", "marginRight": "3px",
        }) for x in s[:2]]
        parts += [html.Span(x, style={
            "background": "rgba(255,64,96,0.12)", "color": RED,
            "borderRadius": "10px", "padding": "1px 6px",
            "fontSize": "0.68rem", "marginRight": "3px",
        }) for x in w[:2]]
        return html.Div(parts, style={"display": "flex", "flexWrap": "wrap", "gap": "2px"})

    detail_rows = []
    for row_idx, (_, row) in enumerate(c_df.iterrows()):
        gap_val = float(c_df.loc[c_df["Competitor"] == row["Competitor"], "Gap %"].iloc[0])
        win_val = float(c_df.loc[c_df["Competitor"] == row["Competitor"], "Win Rate vs"].iloc[0])
        gap_col = GREEN if gap_val > 0 else RED
        win_col = GREEN if win_val >= 65 else (YELLOW if win_val >= 50 else RED)
        detail_rows.append(
            html.Tr([
                html.Td(html.Div([
                    html.Span("●", style={"color": COMPETITOR_COLORS.get(row["Competitor"], MUTED),
                                          "marginRight": "6px"}),
                    html.Span(row["Competitor"], style={"fontWeight": "500"}),
                ]), style={"padding": "0.55rem 0.8rem", "whiteSpace": "nowrap"}),
                html.Td(f"${row['Comp Price']:,.0f}",
                        style={"padding": "0.55rem 0.8rem", "textAlign": "right",
                               "fontFamily": "'DM Mono', monospace", "fontSize": "0.85rem"}),
                html.Td(f"{gap_val:+.1f}%",
                        style={"padding": "0.55rem 0.8rem", "color": gap_col,
                               "textAlign": "right", "fontWeight": "600"}),
                html.Td(f"{row['Quality Score']:.1f}",
                        style={"padding": "0.55rem 0.8rem", "textAlign": "right"}),
                html.Td(f"{row['Value Score']:.2f}",
                        style={"padding": "0.55rem 0.8rem", "textAlign": "right"}),
                html.Td(f"{win_val:.1f}%",
                        style={"padding": "0.55rem 0.8rem", "color": win_col,
                               "textAlign": "right", "fontWeight": "600"}),
                html.Td(f"{COMPETITORS.get(row['Competitor'], {}).get('market_share', 0)*100:.0f}%",
                        style={"padding": "0.55rem 0.8rem", "textAlign": "right",
                               "color": CYAN}),
                html.Td(_strength_badges(row["Competitor"]),
                        style={"padding": "0.55rem 0.8rem"}),
            ], style={"borderBottom": f"1px solid {BORDER}",
                      "background": CARD_BG if row_idx % 2 == 0 else CARD_BG2})
        )

    thead = html.Thead(html.Tr([
        html.Th(h, style={
            "padding": "0.55rem 0.8rem", "background": CARD_BG2,
            "color": ACCENT, "fontSize": "0.72rem", "textTransform": "uppercase",
            "letterSpacing": "0.6px", "fontFamily": "'Space Grotesk', sans-serif",
            "whiteSpace": "nowrap",
        })
        for h in ["Competitor", "Price", "Gap %", "Quality", "Value",
                  "Win Rate", "Mkt Share", "Strengths / Weaknesses"]
    ]))
    benchmark_table = html.Div(
        html.Table([thead, html.Tbody(detail_rows)],
                   style={"width": "100%", "borderCollapse": "collapse",
                          "fontSize": "0.85rem", "color": "#c8d8f0"}),
        style={"overflowX": "auto", "borderRadius": "8px",
               "border": f"1px solid {BORDER2}"},
    )

    # ── 9. Strategic insights text ──────────────────────────────────────────
    cheapest_rival   = c_df.loc[c_df["Comp Price"].idxmin(), "Competitor"]
    cheapest_price   = c_df["Comp Price"].min()
    premium_rival    = c_df.loc[c_df["Comp Price"].idxmax(), "Competitor"]
    best_win_rival   = c_df.loc[c_df["Win Rate vs"].idxmax(), "Competitor"]
    worst_win_rival  = c_df.loc[c_df["Win Rate vs"].idxmin(), "Competitor"]
    best_value_rival = c_df.loc[c_df["Value Score"].idxmax(), "Competitor"]

    price_position = (
        "below-market (penetration stance)" if avg_gap < -5 else
        "at-market (neutral stance)" if abs(avg_gap) <= 5 else
        "above-market (premium stance)"
    )
    insight_text = (
        f"At the scenario price of **${scenario_price:,.0f}** ({price_delta_pct:+d}% vs base), "
        f"STRAIVE is priced **{price_position}** with an average gap of {avg_gap:+.1f}% "
        f"across {len(c_df)} tracked competitors. "
        f"The cheapest rival is **{cheapest_rival}** at ${cheapest_price:,.0f}. "
        f"STRAIVE wins most often against **{best_win_rival}** and faces its toughest "
        f"competition from **{worst_win_rival}**. "
        f"**{best_value_rival}** offers the highest perceived value among competitors, "
        f"making it the primary differentiator risk."
    )
    # Simple markdown → Dash render
    insight_parts = []
    for chunk in insight_text.split("**"):
        if insight_parts and len(insight_parts) % 2 == 1:
            insight_parts.append(html.Strong(chunk, style={"color": ACCENT}))
        else:
            insight_parts.append(chunk)

    # ── Assemble layout ─────────────────────────────────────────────────────
    return html.Div([

        kpi_bar,

        # Insight callout
        html.Div(
            html.Div([
                html.Span("💡 ", style={"fontSize": "1rem"}),
                html.Span("Strategic Insight: ", style={
                    "fontWeight": "600", "color": ACCENT,
                    "fontFamily": "'Space Grotesk', sans-serif",
                }),
                *[html.Span(p) for p in insight_parts],
            ], style={"fontSize": "0.87rem", "lineHeight": "1.7", "color": "#c8d8f0"}),
            style={
                "background": f"linear-gradient(135deg, {CARD_BG} 0%, #0f1a2e 100%)",
                "border": f"1px solid {BORDER2}",
                "borderLeft": f"3px solid {ACCENT}",
                "borderRadius": "8px",
                "padding": "1rem 1.2rem",
                "marginBottom": "1.5rem",
            },
        ),

        # Row 1: Positioning matrix (large) + Win rate bar
        _section("Strategic Positioning Matrix", color=ACCENT),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_pos, config={"displayModeBar": False}), md=7),
            dbc.Col(dcc.Graph(figure=fig_win, config={"displayModeBar": False}), md=5),
        ], className="g-3"),

        # Row 2: Price gap waterfall + Value map
        _section("Price Gap & Value Analysis", color=PURPLE),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_gap, config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_val, config={"displayModeBar": False}), md=6),
        ], className="g-3"),

        # Row 3: Radar + Sensitivity curve
        _section("Capability Radar & Price Sensitivity", color=CYAN),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_radar, config={"displayModeBar": False}), md=5),
            dbc.Col(dcc.Graph(figure=fig_sens, config={"displayModeBar": False}), md=7),
        ], className="g-3"),

        # Detailed benchmark table
        _section("Detailed Competitor Benchmarks", color=GREEN),
        benchmark_table,

        html.Div(style={"height": "2rem"}),
    ])


# ============================================================================
# CALLBACK: MARKET INTELLIGENCE TAB
# ============================================================================
@app.callback(
    Output("market-intel-tab-content", "children"),
    Input("active-tab-store", "data"),
    [State("data-store", "data"),
     State("competitor-data-store", "data"),
     State("market-intel-store", "data"),
     State("pricing-signals-store", "data")],
)
def update_market_intel_tab(active_tab, df_json, competitor_json, intel_json, signals_json):
    """Update market intelligence tab content."""
    if active_tab != "🕷️ Market Intelligence":
        raise PreventUpdate

    # generate_intelligence_report returns (children, style, pdf_download) — we only need children
    result = generate_intelligence_report(1, df_json, competitor_json, intel_json, signals_json)
    return result[0] if isinstance(result, (tuple, list)) else result

# ============================================================================
# CALLBACK: OPTIMAL PRICING RESULT (Enhanced with competitor data)
# ============================================================================
@app.callback(
    Output("optimal-result", "children"),
    Input("opt-calculate-btn", "n_clicks"),
    [State("opt-product", "value"),
     State("opt-target", "value"),
     State("opt-strategy", "value"),
     State("elasticity-store", "data"),
     State("competitor-data-store", "data")],
    prevent_initial_call=True,
)
def calculate_optimal(n, product, target, strategy, elast_json, competitor_json):
    if not n or not product:
        raise PreventUpdate
    
    info = PRODUCT_CATALOG[product]
    base = info["base_price"]
    cost = info["cost"]
    seg = info["segment"]
    elast = json.loads(elast_json) if elast_json else {}
    elasticity = elast.get(seg, {}).get("elasticity", -1.35)

    # Parse competitor data
    competitor_prices = {}
    if competitor_json:
        try:
            comp_data = json.loads(competitor_json)
            for item in comp_data:
                if item.get("competitor_name") in COMPETITORS:
                    competitor_prices[item["competitor_name"]] = item.get("price", base)
        except Exception:
            pass

    strat_cfg = PRICING_STRATEGIES.get(strategy, {})
    constraints = {
        "min_price": base * strat_cfg.get("price_floor_pct", 0.5),
        "max_price": base * strat_cfg.get("price_ceil_pct", 2.0),
    }
    
    result = PricingOptimizer.calculate(
        base_price=base, cost=cost, elasticity=elasticity,
        target=target, constraints=constraints,
        complexity=info.get("complexity", 3),
        competitor_prices=competitor_prices if competitor_prices else None,
    )

    pct = result["vs_base_pct"]
    pct_color = GREEN if pct >= 0 else RED

    prices = np.linspace(base * 0.5, base * 1.8, 150)
    d = np.maximum(0.01, (prices / base) ** elasticity)
    rev_arr = prices * d
    prf_arr = (prices - cost) * d
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=rev_arr, name="Revenue",
                              line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=prices, y=prf_arr, name="Profit",
                              line=dict(color=GREEN, width=2.5)))
    fig.add_vline(x=result["optimal_price"], line_dash="dash", line_color=YELLOW,
                  annotation_text=f"Optimal ${result['optimal_price']:,.0f}")
    fig.add_vline(x=base, line_dash="dot", line_color=MUTED, annotation_text="Base")
    
    # Add competitor markers if available
    if competitor_prices:
        for comp_name, comp_price in competitor_prices.items():
            if comp_price > 0:
                fig.add_vline(x=comp_price, line_dash="dot", line_color=COMPETITOR_COLORS.get(comp_name, MUTED),
                              annotation_text=comp_name, annotation_position="bottom")
    
    fig.update_layout(**PLOTLY_DARK, height=340,
                      title=f"Revenue & Profit Curve – {product}",
                      xaxis_title="Price ($)", yaxis_title="Value ($)")

    # Add competitor comparison to result
    comp_text = ""
    if competitor_prices and "vs_competitor" in result:
        vs_comp = result["vs_competitor"]
        comp_text = f" | Price vs competitor avg: {vs_comp:.2f}x"

    return html.Div([
        dbc.Row([
            _card("Optimal Price", f"${result['optimal_price']:,.2f}", ACCENT),
            _card("Expected Revenue", f"${result['expected_revenue']:,.0f}", PURPLE),
            _card("Expected Profit", f"${result['expected_profit']:,.0f}", GREEN),
            _card("Margin", f"{result['margin_pct']:.1f}%", YELLOW,
                  f"{pct:+.1f}% vs base"),
            _card("Win Probability", f"{result['win_probability']*100:.1f}%", CYAN),
            _card("Demand Index", f"{result['demand_index']:.4f}", ORANGE),
        ], className="g-3"),
        dcc.Graph(figure=fig, style={"marginTop": "1.5rem"}),
        html.Div([
            html.Span("Recommendation:  ", style={"fontWeight": "600", "color": "#c8d8f0"}),
            html.Span(f"Set price at ${result['optimal_price']:,.2f}  "),
            html.Span(f"({pct:+.1f}% vs base)", style={"color": pct_color}),
            html.Span(f" to maximise {target}. Strategy: {strategy}.{comp_text} "
                      f"Win probability: {result['win_probability']*100:.1f}%. "),
        ], style={"marginTop": "1.2rem", "color": "#a0b4cc", "fontSize": "0.9rem",
                   "padding": "0.9rem 1.1rem",
                   "background": CARD_BG, "borderRadius": "8px",
                   "border": f"1px solid {BORDER2}"}),
    ])

# ============================================================================
# CALLBACK: MONTE-CARLO SIMULATION (Enhanced)
# ============================================================================
@app.callback(
    Output("simulation-result", "children"),
    Input("sim-run-btn", "n_clicks"),
    [State("sim-product", "value"),
     State("sim-price-range", "value"),
     State("sim-count", "value"),
     State("elasticity-store", "data"),
     State("competitor-data-store", "data")],
    prevent_initial_call=True,
)
def run_simulation(n, product, pct_range, n_sim, elast_json, competitor_json):
    if not n or not product:
        raise PreventUpdate
    
    info = PRODUCT_CATALOG[product]
    base = info["base_price"]
    cost = info["cost"]
    seg = info["segment"]
    elast = json.loads(elast_json) if elast_json else {}
    elasticity = elast.get(seg, {}).get("elasticity", -1.35)
    
    # Parse competitor data
    competitor_prices = {}
    if competitor_json:
        try:
            comp_data = json.loads(competitor_json)
            for item in comp_data:
                if item.get("competitor_name") in COMPETITORS:
                    competitor_prices[item["competitor_name"]] = item.get("price", base)
        except Exception:
            pass
    
    p_low = base * (1 + pct_range[0] / 100)
    p_high = base * (1 + pct_range[1] / 100)

    sim_df, risk = SimulationEngine.simulate_revenue_scenarios(
        product=product, price_range=(p_low, p_high),
        elasticity=elasticity, cost=cost, n_sim=n_sim,
        competitor_prices=competitor_prices if competitor_prices else None,
    )

    fig_hist = px.histogram(sim_df, x="revenue", nbins=60,
                            title="Revenue Distribution",
                            color_discrete_sequence=[ACCENT])
    fig_hist.add_vline(x=risk["revenue_p50"], line_dash="dash",
                       line_color=GREEN, annotation_text="P50")
    fig_hist.add_vline(x=risk["revenue_var5"], line_dash="dash",
                       line_color=RED, annotation_text="VaR 5%")
    fig_hist.update_layout(**PLOTLY_DARK, height=400, bargap=0.03)

    fig_profit = px.histogram(sim_df, x="profit", nbins=60,
                              title="Profit Distribution",
                              color_discrete_sequence=[GREEN])
    fig_profit.add_vline(x=risk["profit_p50"], line_dash="dash",
                          line_color=ACCENT, annotation_text="P50")
    fig_profit.update_layout(**PLOTLY_DARK, height=400, bargap=0.03)

    fig_scatter = px.scatter(
        sim_df.sample(min(2_000, len(sim_df)), random_state=42),
        x="price", y="revenue", color="margin",
        color_continuous_scale="RdYlGn",
        title="Price vs Revenue (colour = Margin %)", opacity=0.5,
    )
    fig_scatter.update_layout(**PLOTLY_DARK, height=400)

    risk_table = pd.DataFrame([
        {"Metric": "Revenue – Mean", "Value": f"${risk['revenue_mean']:,.2f}"},
        {"Metric": "Revenue – P10", "Value": f"${risk['revenue_p10']:,.2f}"},
        {"Metric": "Revenue – P50", "Value": f"${risk['revenue_p50']:,.2f}"},
        {"Metric": "Revenue – P90", "Value": f"${risk['revenue_p90']:,.2f}"},
        {"Metric": "Revenue – VaR 5%", "Value": f"${risk['revenue_var5']:,.2f}"},
        {"Metric": "Revenue – 90% CI", "Value": f"[${risk['revenue_ci_lower']:,.0f}, ${risk['revenue_ci_upper']:,.0f}]"},
        {"Metric": "Profit – Mean", "Value": f"${risk['profit_mean']:,.2f}"},
        {"Metric": "Profit – P50", "Value": f"${risk['profit_p50']:,.2f}"},
        {"Metric": "Profit – VaR 5%", "Value": f"${risk['profit_var5']:,.2f}"},
        {"Metric": "Profit – 90% CI", "Value": f"[${risk['profit_ci_lower']:,.0f}, ${risk['profit_ci_upper']:,.0f}]"},
        {"Metric": "Simulations Run", "Value": str(risk["n_sim"])},
        {"Metric": "Confidence Level", "Value": f"{risk['confidence']*100:.0f}%"},
    ])

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_hist), md=6),
                 dbc.Col(dcc.Graph(figure=fig_profit), md=6)], className="g-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_scatter), md=12)], className="g-3 mt-1"),
        _section("Risk Summary (Percentile Table)"),
        dbc.Table.from_dataframe(risk_table, striped=True, bordered=True,
                                  hover=True, responsive=True),
    ])

# ============================================================================
# CALLBACK: WHAT-IF SCENARIO (Enhanced v2)
# ============================================================================
@app.callback(
    Output("whatif-result", "children"),
    Input("whatif-apply-btn", "n_clicks"),
    [State({"type": "whatif-slider", "index": ALL}, "value"),
     State({"type": "whatif-slider", "index": ALL}, "id"),
     State({"type": "whatif-seg-slider", "index": ALL}, "value"),
     State({"type": "whatif-seg-slider", "index": ALL}, "id"),
     State("whatif-scenario-name", "value"),
     State("whatif-objective", "value"),
     State("whatif-seg-filter", "value"),
     State("data-store", "data"),
     State("elasticity-store", "data"),
     State("competitor-data-store", "data")],
    prevent_initial_call=True,
)
def apply_whatif(
    n,
    slider_values, slider_ids,
    seg_slider_values, seg_slider_ids,
    scenario_name, objective, seg_filter,
    df_json, elast_json, competitor_json,
):
    if not n or not df_json:
        raise PreventUpdate

    df   = pd.read_json(df_json, orient="split")
    elast = json.loads(elast_json) if elast_json else {}

    # ── Build per-product price overrides ────────────────────────────────
    # Start from slider values, then apply segment-wide adjustments on top
    seg_adjustments: Dict[str, float] = {}
    if seg_slider_ids and seg_slider_values:
        for sid, val in zip(seg_slider_ids, seg_slider_values):
            if val != 0:
                seg_adjustments[sid["index"]] = val / 100.0   # convert % → ratio

    overrides: Dict[str, float] = {}
    for sid, val in zip(slider_ids, slider_values):
        prod = sid["index"]
        info = PRODUCT_CATALOG.get(prod, {})
        seg  = info.get("segment", "")

        # Apply segment-level nudge on top of per-product slider value
        if seg in seg_adjustments:
            adj = 1.0 + seg_adjustments[seg]
            val = float(np.clip(
                val * adj,
                info.get("min_price", val * 0.5),
                info.get("max_price", val * 2.0),
            ))

        # Only include products whose price actually changed
        base = info.get("base_price", val)
        if abs(val - base) > 0.5:
            overrides[prod] = float(val)

    # Optional segment filter: restrict to products in chosen segment
    if seg_filter and seg_filter != "all":
        overrides = {
            p: v for p, v in overrides.items()
            if PRODUCT_CATALOG.get(p, {}).get("segment") == seg_filter
        }

    if not overrides:
        return dbc.Alert(
            "No prices were changed from their base values. "
            "Move at least one slider and click Run again.",
            color="warning", className="mt-3",
        )

    # ── Parse competitor data ─────────────────────────────────────────────
    competitor_prices: Dict[str, float] = {}
    if competitor_json:
        try:
            for item in json.loads(competitor_json):
                if item.get("competitor_name") in COMPETITORS:
                    competitor_prices[item["competitor_name"]] = float(item.get("price", 0))
        except Exception:
            pass

    # ── Run the simulation ────────────────────────────────────────────────
    _modified, summary_df = SimulationEngine.apply_pricing_scenario(
        df, overrides, elasticities=elast,
        competitor_prices=competitor_prices if competitor_prices else None,
    )

    if summary_df.empty:
        return dbc.Alert("No matching products found in the dataset.", color="warning")

    # ── Derived metrics ───────────────────────────────────────────────────
    total_rev_before  = summary_df["revenue_before"].sum()
    total_rev_after   = summary_df["revenue_after"].sum()
    total_rev_delta   = summary_df["revenue_delta"].sum()
    total_prof_before = summary_df["profit_before"].sum()
    total_prof_after  = summary_df["profit_after"].sum()
    total_prof_delta  = summary_df["profit_delta"].sum()
    avg_vol_change    = summary_df["volume_change_pct"].mean()
    n_increased       = (summary_df["revenue_delta"] > 0).sum()
    n_decreased       = (summary_df["revenue_delta"] < 0).sum()

    # Margin before/after
    summary_df["margin_before_pct"] = (
        (summary_df["revenue_before"] - summary_df["profit_before"])
        .rsub(summary_df["revenue_before"])          # revenue_before - cost_before
        .div(summary_df["revenue_before"].replace(0, np.nan)) * 100
    ).fillna(0).round(2)
    # Simpler: use profit_before / revenue_before
    summary_df["margin_before_pct"] = (
        summary_df["profit_before"]
        .div(summary_df["revenue_before"].replace(0, np.nan)) * 100
    ).fillna(0).round(2)
    summary_df["margin_after_pct"] = (
        summary_df["profit_after"]
        .div(summary_df["revenue_after"].replace(0, np.nan)) * 100
    ).fillna(0).round(2)
    summary_df["margin_delta_pct"] = (
        summary_df["margin_after_pct"] - summary_df["margin_before_pct"]
    ).round(2)

    # Break-even volume index (volume needed at new price to match old revenue)
    summary_df["breakeven_vol_idx"] = (
        summary_df["revenue_before"]
        .div(summary_df["new_price"].replace(0, np.nan))
        .div(
            (summary_df["revenue_before"]
             .div(summary_df["old_price"].replace(0, np.nan)))
            .replace(0, np.nan)
        )
    ).fillna(1.0).round(3)

    color_r = GREEN if total_rev_delta >= 0 else RED
    color_p = GREEN if total_prof_delta >= 0 else RED
    color_v = GREEN if avg_vol_change >= 0 else RED
    scenario_label = scenario_name or "Unnamed Scenario"

    # ── KPI cards ─────────────────────────────────────────────────────────
    kpi_row = dbc.Row([
        _card("Revenue Δ",    f"${total_rev_delta:+,.0f}",  color_r,
              sub=f"${total_rev_before/1e3:,.0f}k → ${total_rev_after/1e3:,.0f}k"),
        _card("Profit Δ",     f"${total_prof_delta:+,.0f}", color_p,
              sub=f"${total_prof_before/1e3:,.0f}k → ${total_prof_after/1e3:,.0f}k"),
        _card("Avg Vol Δ",    f"{avg_vol_change:+.1f}%",    color_v),
        _card("Products ↑",   str(n_increased),              GREEN),
        _card("Products ↓",   str(n_decreased),              RED),
        _card("Scenario",     scenario_label[:18],            PURPLE),
    ], className="g-3")

    # ── Revenue impact bar chart ─────────────────────────────────────────
    fig_delta = px.bar(
        summary_df.sort_values("revenue_delta"),
        x="revenue_delta", y="product", orientation="h",
        color="revenue_delta", color_continuous_scale="RdYlGn",
        title="Revenue Impact by Product",
        text_auto=".2s",
        labels={"revenue_delta": "Revenue Δ ($)", "product": ""},
    )
    fig_delta.add_vline(x=0, line_color=MUTED, line_width=1.5)
    fig_delta.update_layout(
        **PLOTLY_DARK,
        height=max(320, len(summary_df) * 42),
        showlegend=False,
    )
    fig_delta.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig_delta.update_traces(marker_line_width=0)

    # ── Revenue waterfall chart ───────────────────────────────────────────
    wf_products = summary_df["product"].tolist()
    wf_deltas   = summary_df["revenue_delta"].tolist()
    wf_colors   = [GREEN if d >= 0 else RED for d in wf_deltas]

    fig_waterfall = go.Figure(go.Waterfall(
        name="Revenue Waterfall",
        orientation="v",
        measure=["relative"] * len(wf_products) + ["total"],
        x=wf_products + ["NET TOTAL"],
        y=wf_deltas + [total_rev_delta],
        connector={"line": {"color": BORDER2, "width": 1}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": ACCENT}},
        text=[f"${d:+,.0f}" for d in wf_deltas] + [f"${total_rev_delta:+,.0f}"],
        textposition="outside",
    ))
    fig_waterfall.update_layout(
        **PLOTLY_DARK,
        title="Revenue Waterfall: Per-Product Contribution",
        height=max(340, len(summary_df) * 28 + 120),
        showlegend=False,
        xaxis_tickangle=-35,
    )
    fig_waterfall.update_layout(margin=dict(l=10, r=10, t=40, b=60))

    # ── Margin before vs after scatter ───────────────────────────────────
    fig_margin = go.Figure()
    for _, row in summary_df.iterrows():
        clr = GREEN if row["margin_delta_pct"] >= 0 else RED
        fig_margin.add_trace(go.Scatter(
            x=[row["margin_before_pct"]], y=[row["margin_after_pct"]],
            mode="markers+text",
            marker=dict(size=14, color=clr, opacity=0.85,
                        line=dict(color="#fff", width=1)),
            text=[row["product"].split(" – ")[0]],
            textposition="top center",
            textfont=dict(size=9, color="#c0cce0"),
            name=row["product"],
            showlegend=False,
        ))
    # Diagonal reference line
    all_margins = (
        list(summary_df["margin_before_pct"]) + list(summary_df["margin_after_pct"])
    )
    m_lo, m_hi = min(all_margins) - 2, max(all_margins) + 2
    fig_margin.add_trace(go.Scatter(
        x=[m_lo, m_hi], y=[m_lo, m_hi],
        mode="lines",
        line=dict(color=MUTED, dash="dash", width=1),
        name="No change",
        showlegend=True,
    ))
    fig_margin.update_layout(
        **PLOTLY_DARK,
        title="Margin Before vs After (points above diagonal = improvement)",
        height=400,
        xaxis_title="Margin Before (%)",
        yaxis_title="Margin After (%)",
    )
    fig_margin.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    # ── Price-change percentage bars ──────────────────────────────────────
    fig_price_pct = px.bar(
        summary_df.sort_values("price_change_pct"),
        x="product", y="price_change_pct",
        color="price_change_pct", color_continuous_scale="RdYlGn",
        title="Price Change % per Product",
        text_auto=".1f",
        labels={"price_change_pct": "Price Δ (%)", "product": ""},
    )
    fig_price_pct.add_hline(y=0, line_color=MUTED, line_width=1)
    fig_price_pct.update_layout(
        **PLOTLY_DARK,
        height=320, showlegend=False,
        xaxis_tickangle=-35,
    )
    fig_price_pct.update_layout(margin=dict(l=10, r=10, t=40, b=80))
    fig_price_pct.update_traces(marker_line_width=0)

    # ── Objective-specific insight ────────────────────────────────────────
    if objective == "revenue":
        top_prod = summary_df.nlargest(1, "revenue_delta").iloc[0]
        insight_text = (
            f"Revenue objective: best performer is '{top_prod['product']}' "
            f"(+${top_prod['revenue_delta']:,.0f}). "
            f"Net portfolio revenue change: ${total_rev_delta:+,.0f}."
        )
        insight_color = "success" if total_rev_delta >= 0 else "danger"
    elif objective == "profit":
        top_prod = summary_df.nlargest(1, "profit_delta").iloc[0]
        insight_text = (
            f"Profit objective: biggest profit gain is '{top_prod['product']}' "
            f"(+${top_prod['profit_delta']:,.0f}). "
            f"Net portfolio profit change: ${total_prof_delta:+,.0f}."
        )
        insight_color = "success" if total_prof_delta >= 0 else "danger"
    elif objective == "win_rate":
        n_price_cuts = (summary_df["price_change_pct"] < 0).sum()
        insight_text = (
            f"Win-rate defence: {n_price_cuts} product(s) had price reductions. "
            f"Lower prices typically improve competitive win probability. "
            f"Monitor margin impact: ${total_prof_delta:+,.0f}."
        )
        insight_color = "info"
    else:  # market_share
        vol_gainers = (summary_df["volume_change_pct"] > 0).sum()
        insight_text = (
            f"Market-share objective: {vol_gainers}/{len(summary_df)} products "
            f"are projected to grow volume. Avg volume change: {avg_vol_change:+.1f}%."
        )
        insight_color = "primary"

    # ── Display table (clean columns) ────────────────────────────────────
    display_cols = [
        "product", "old_price", "new_price", "price_change_pct",
        "revenue_delta", "profit_delta", "volume_change_pct",
        "margin_before_pct", "margin_after_pct", "margin_delta_pct",
    ]
    col_rename = {
        "product":           "Product",
        "old_price":         "Old Price ($)",
        "new_price":         "New Price ($)",
        "price_change_pct":  "Price Δ (%)",
        "revenue_delta":     "Revenue Δ ($)",
        "profit_delta":      "Profit Δ ($)",
        "volume_change_pct": "Volume Δ (%)",
        "margin_before_pct": "Margin Before (%)",
        "margin_after_pct":  "Margin After (%)",
        "margin_delta_pct":  "Margin Δ (pp)",
    }
    table_df = summary_df[display_cols].rename(columns=col_rename).round(2)

    return html.Div([
        # Scenario header badge
        html.Div([
            html.Span("📋 ", style={"fontSize": "1rem"}),
            html.Span(f"Scenario: {scenario_label}",
                      style={"color": ACCENT, "fontWeight": "700",
                             "fontFamily": "'Space Grotesk'", "fontSize": "1rem"}),
            html.Span(f"  ·  Objective: {objective.replace('_', ' ').title()}",
                      style={"color": MUTED, "fontSize": "0.82rem", "marginLeft": "8px"}),
        ], style={"marginBottom": "1rem", "padding": "0.6rem 1rem",
                  "background": f"rgba(79,158,255,0.08)",
                  "borderRadius": "8px", "border": f"1px solid {BORDER2}"}),

        # Objective insight
        dbc.Alert(insight_text, color=insight_color, className="mb-3",
                  style={"fontSize": "0.85rem"}),

        # KPI cards
        kpi_row,

        # Charts: row 1
        _section("Revenue & Profit Impact"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_delta), md=7),
            dbc.Col(dcc.Graph(figure=fig_price_pct), md=5),
        ], className="g-3"),

        # Waterfall
        _section("Revenue Waterfall"),
        dcc.Graph(figure=fig_waterfall),

        # Margin scatter
        _section("Margin Before vs After"),
        dcc.Graph(figure=fig_margin),

        # Detail table
        _section("Scenario Impact Detail"),
        dbc.Table.from_dataframe(
            table_df, striped=True, bordered=True,
            hover=True, responsive=True,
            style={"fontSize": "0.78rem"},
        ),

        # Break-even note
        html.Div([
            html.Span("ℹ️  Break-even volume index: ",
                      style={"color": MUTED, "fontSize": "0.75rem", "fontWeight": "600"}),
            html.Span(
                "values < 1 mean you need proportionally fewer units to match previous revenue "
                "(price rise more than offsets volume loss). Values > 1 mean you need more units.",
                style={"color": MUTED, "fontSize": "0.75rem"},
            ),
        ], style={"marginTop": "0.5rem", "padding": "0.5rem",
                  "background": "rgba(0,0,0,0.2)", "borderRadius": "6px"}),
    ])

# ============================================================================
# CALLBACK: WHAT-IF PRESET BUTTONS (reset / ±10%)
# ============================================================================
@app.callback(
    Output({"type": "whatif-slider", "index": ALL}, "value"),
    [Input("whatif-preset-up10",   "n_clicks"),
     Input("whatif-preset-down10", "n_clicks"),
     Input("whatif-preset-reset",  "n_clicks")],
    State({"type": "whatif-slider", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def apply_whatif_preset(up10, down10, reset, slider_ids):
    trigger = ctx.triggered_id
    if not trigger:
        raise PreventUpdate
    new_values = []
    for sid in slider_ids:
        prod = sid["index"]
        base = PRODUCT_CATALOG.get(prod, {}).get("base_price", 100)
        if trigger == "whatif-preset-up10":
            val = base * 1.10
        elif trigger == "whatif-preset-down10":
            val = base * 0.90
        else:  # reset
            val = base
        info = PRODUCT_CATALOG.get(prod, {})
        val = float(np.clip(
            val,
            info.get("min_price", base * 0.5),
            info.get("max_price", base * 2.0),
        ))
        new_values.append(int(round(val)))
    return new_values


# ============================================================================
# CALLBACK: EXPORT (Enhanced)
# ============================================================================
@app.callback(
    Output("download-component", "data"),
    Input("download-btn", "n_clicks"),
    [State("data-store", "data"),
     State("elasticity-store", "data"),
     State("win-model-store", "data"),
     State("competitor-data-store", "data"),
     State("market-intel-store", "data"),
     State("pricing-signals-store", "data"),
     State("active-tab-store", "data")],
    prevent_initial_call=True,
)
def export_results(n_clicks, df_json, elast_json, win_json, competitor_json, intel_json, signals_json, active_tab):
    if not n_clicks:
        raise PreventUpdate
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_rows = 0
    if df_json:
        try:
            n_rows = len(pd.read_json(df_json, orient="split"))
        except Exception:
            pass

    payload = {
        "exported": datetime.now().isoformat(),
        "app_version": APP_VERSION,
        "active_tab": active_tab,
        "dataset_rows": n_rows,
        "elasticity": json.loads(elast_json) if elast_json else {},
        "win_model_meta": json.loads(win_json) if win_json else {},
        "competitor_data": json.loads(competitor_json) if competitor_json else [],
        "market_intelligence": json.loads(intel_json) if intel_json else [],
        "pricing_signals": json.loads(signals_json) if signals_json else [],
        "products": {k: {f: v for f, v in info.items() if f != "tags"}
                      for k, info in PRODUCT_CATALOG.items()},
        "note": "Full dataset excluded. Re-generate or reload CSV to restore.",
    }
    return dict(
        content=json.dumps(payload, indent=2, default=str),
        filename=f"straive_export_{ts}.json",
        type="application/json",
    )

# ============================================================================
# CALLBACK: PRICE-VOLUME CURVES
# ============================================================================
@app.callback(
    Output("pvc-graph", "figure"),
    Input("pvc-product", "value"),
    [State("elasticity-store", "data")],
)
def update_pvc_graph(product, elast_json):
    if not product:
        raise PreventUpdate
    
    info = PRODUCT_CATALOG[product]
    seg = info["segment"]
    elast = json.loads(elast_json) if elast_json else {}
    elast_val = elast.get(seg, {}).get("elasticity", -1.2)
    base = info["base_price"]
    cost = info["cost"]
    
    prices = np.linspace(base * 0.4, base * 2.2, 200)
    volumes = np.maximum(0.01, (prices / base) ** elast_val)
    revenues = prices * volumes
    profits = (prices - cost) * volumes

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue",
                              line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit",
                              line=dict(color=GREEN, width=2.5)))
    fig.add_trace(go.Scatter(x=prices, y=volumes * base / 2, name="Volume (scaled)",
                              line=dict(color=YELLOW, width=1.5, dash="dot")))
    fig.add_vline(x=base, line_dash="dash", line_color=MUTED,
                  annotation_text="Base Price")
    fig.update_layout(**PLOTLY_DARK, title=f"Price-Volume-Revenue – {product}",
                      height=480, xaxis_title="Price ($)", yaxis_title="Value ($)")
    
    return fig
  
# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"{APP_TITLE}")
    print(f"{'='*60}")
    print(f"Version: {APP_VERSION}")
    print(f"URL: http://127.0.0.1:8050")
    print(f"\nFeatures:")
    print(f" • 17 Analytics Modules")
    print(f" • Real-time competitor price scraping")
    print(f" • Market intelligence gathering")
    print(f" • Pricing signal detection")
    print(f" • Blended synthetic + real data generation")
    print(f" • Confidence-weighted modeling")
    print(f"\nStarting server...\n")

    # For local development only
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050
    )

# ────────────────────────────────────────────────
# Important line for production / hosted environments
# Render, Railway, Fly.io, etc. look for this
# ────────────────────────────────────────────────
server = app.server

