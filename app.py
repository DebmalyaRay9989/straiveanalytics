"""
STRAIVE Pricing Platform – Dash Application (v5.0 Cloud-Light)

Changes vs v4.1:
  • Removed ReportLab / PDF generation entirely (heavy C-extension dep)
  • Removed Market Intelligence tab and all scraping UI controls
  • Removed scraping-status, scrape-now-btn, intel-report-output components
  • DataGenerator default reduced to 2000 rows (configurable up to 10k)
  • Uses NAV_OPTIONS from config (Market Intelligence tab already removed there)
  • All 15 analytics tabs retained; only the scraping-dependent tab removed
  • No threading / asyncio for background tasks
  • server = app.server exported for gunicorn / Procfile
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
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import logging
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from config import (
    PRODUCT_CATALOG, PRODUCT_SEGMENTS, SEGMENT_COLORS, COMPETITOR_COLORS,
    PLOTLY_DARK, NAV_OPTIONS, NAV_GROUPS, DARK_BG, CARD_BG, CARD_BG2,
    SIDEBAR_BG, BORDER, BORDER2, ACCENT, ACCENT2, ACCENT3, GREEN, YELLOW, RED,
    PURPLE, CYAN, ORANGE, MUTED, CURRENT_DATE, DATA_COLLECTION_GUIDANCE,
    CUSTOMER_SEGMENTS, REGIONS, COMPETITORS, PRICING_STRATEGIES,
    MONTHLY_SEASONALITY, MARGIN_WATERFALL_BUCKETS, APP_TITLE, APP_VERSION,
    CSV_UPLOAD_SPEC, PREPROCESSING_OPTIONS, SCRAPING_CONFIG,
    COMPETITOR_NAMES,
)
from engine import (
    DataGenerator, ModelComparator, PricingOptimizer, SimulationEngine,
    CompetitiveAnalyzer, RevenueForecaster, DealScorer,
    MarginWaterfallBuilder, PortfolioScorer, MarketIntelligenceIntegrator,
)
from data_gathering import (
    DataGatheringOrchestrator, EnhancedDataGenerator, ScheduledDataGatherer,
    CompetitorPricePoint, MarketIntelligence, PricingSignal,
)

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# SHARED HELPERS
# ============================================================================
def _parse_upload(contents: str, filename: str) -> tuple:
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
    except Exception as exc:
        return None, f"Could not decode upload payload: {exc}"

    file_size_mb = len(decoded) / (1024 * 1024)
    max_size_mb  = CSV_UPLOAD_SPEC.get("max_file_size_mb", 20)
    if file_size_mb > max_size_mb:
        return None, f"File too large: {file_size_mb:.1f} MB (max {max_size_mb} MB)."

    file_ext = (filename or "").lower().rsplit(".", 1)[-1]
    if file_ext == "csv":
        df = None
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


def _numpy_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return str(obj)

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=_numpy_default)

def _fmt_date(value, fallback: str = "—", length: int = 10) -> str:
    if value is None:              return fallback
    if isinstance(value, datetime): return value.isoformat()[:length]
    try:    return str(value)[:length]
    except: return fallback

# ============================================================================
# INIT DATA COMPONENTS
# ============================================================================
data_orchestrator = DataGatheringOrchestrator(use_redis=False)
enhanced_generator = EnhancedDataGenerator(data_orchestrator)
market_integrator  = MarketIntelligenceIntegrator()

# ============================================================================
# UI COMPONENT HELPERS
# ============================================================================
def _card(title: str, value: str, color: str = ACCENT, sub: str = "", tooltip: str = "") -> dbc.Col:
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
            html.P(sub, style={"color": MUTED, "fontSize": "0.75rem", "margin": "0.2rem 0 0"})
            if sub else None,
        ], style={"padding": "1rem 1.2rem"}),
        style={
            "background": f"linear-gradient(135deg, {CARD_BG} 0%, #0f1a2e 100%)",
            "border": f"1px solid {BORDER2}", "borderRadius": "10px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
            "cursor": "help" if tooltip else "default",
        },
    )
    inner = html.Div(card, title=tooltip) if tooltip else card
    return dbc.Col(inner, md=2)

def _section(title: str, color: str = ACCENT) -> html.Div:
    return html.Div([
        html.H5(title, style={
            "color": color, "marginTop": "2.5rem", "marginBottom": "0.8rem",
            "fontFamily": "'Space Grotesk', sans-serif", "fontWeight": "600",
            "fontSize": "0.95rem", "textTransform": "uppercase", "letterSpacing": "1px",
        }),
        html.Hr(style={"borderColor": BORDER, "marginTop": 0, "marginBottom": "1rem"}),
    ])

def _empty(msg: str = "Build a model first — click  ▶  Build / Refresh Model  in the sidebar") -> html.Div:
    return html.Div([
        html.Div(style={
            "width": "80px", "height": "80px", "borderRadius": "50%",
            "background": f"linear-gradient(135deg, {CARD_BG}, {CARD_BG2})",
            "border": f"2px solid {BORDER2}",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "margin": "8rem auto 1.5rem", "fontSize": "2rem",
        }, children="🔒"),
        html.H5(msg, style={
            "textAlign": "center", "color": MUTED, "maxWidth": "480px",
            "margin": "0 auto", "lineHeight": "1.6", "fontWeight": "400",
        }),
    ])

def _dd_style() -> Dict:
    return {"background": CARD_BG2, "color": "#fff", "borderColor": BORDER2, "borderRadius": "7px"}

def _input_style() -> Dict:
    return {
        "width": "100%", "background": CARD_BG2,
        "border": f"1px solid {BORDER2}", "borderRadius": "7px",
        "color": "#fff", "padding": "0.5rem 0.75rem",
    }

def _badge(text: str, color: str) -> html.Span:
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        bg = f"rgba({r},{g},{b},0.18)"
    else:
        bg = "rgba(79,158,255,0.18)"
    return html.Span(text, style={
        "display": "inline-block", "padding": "2px 8px", "borderRadius": "12px",
        "background": bg, "color": color, "fontSize": "0.7rem",
        "fontWeight": "500", "marginLeft": "5px",
    })

# ============================================================================
# KPI BAR
# ============================================================================
def _build_kpi_bar(
    df_json: Optional[str],
    scored_deals: Optional[list] = None,
    win_meta: Optional[dict] = None,
) -> dbc.Row:
    total_rev = total_prof = avg_margin = avg_deal = 0.0
    win_rate = None
    n_products = 0
    avg_confidence = 1.0

    if df_json:
        try:
            df = pd.read_json(df_json, orient="split")
            total_rev      = float(df["revenue"].sum())
            total_prof     = float((df["revenue"] - df["cost"]).sum())
            avg_margin     = float(df["margin_pct"].mean()) if "margin_pct" in df.columns else 35.0
            win_rate       = float(df["deal_won"].mean() * 100) if "deal_won" in df.columns else None
            avg_deal       = float(df["revenue"].mean())
            n_products     = int(df["product"].nunique())
            avg_confidence = float(df["confidence_score"].mean()) if "confidence_score" in df.columns else 1.0
        except Exception:
            pass

    auc_text = f"AUC {win_meta.get('auc', 0):.3f}" if win_meta else ""
    conf_badge = _badge(f"Conf: {avg_confidence:.2f}", GREEN if avg_confidence > 0.8 else YELLOW)

    cards = [
        _card("Total Revenue",   f"${total_rev/1e6:,.2f}M",  ACCENT),
        _card("Gross Profit",    f"${total_prof/1e6:,.2f}M", GREEN),
        _card("Avg Margin",      f"{avg_margin:.1f}%",        PURPLE),
        _card("Win Rate",        f"{win_rate:.1f}%" if win_rate is not None else "—", YELLOW, auc_text),
        _card("Avg Deal Size",   f"${avg_deal:,.0f}",         CYAN),
        _card("Active Products", str(n_products),             ORANGE, sub=conf_badge),
    ]
    return dbc.Row(cards, className="g-3", style={"marginBottom": "0.5rem"})


# ============================================================================
# GLOBAL CSS
# ============================================================================
GLOBAL_CSS = f"""
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{
    background: {DARK_BG}; color: #e0e8f4;
    font-family: 'DM Sans', 'Segoe UI', sans-serif; min-height: 100vh;
}}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 3px; }}
.nav-tab {{ transition: all 0.18s ease !important; cursor: pointer; }}
.nav-tab:hover {{
    background: linear-gradient(90deg, rgba(79,158,255,0.12), rgba(79,158,255,0.04)) !important;
    border-color: rgba(79,158,255,0.35) !important;
    color: #c8d8f0 !important; transform: translateX(3px);
}}
.nav-tab.active {{
    background: linear-gradient(90deg, rgba(79,158,255,0.22), rgba(79,158,255,0.08)) !important;
    border-left: 3px solid {ACCENT} !important; color: #fff !important;
}}
.Select-control {{ background: {CARD_BG2} !important; border-color: {BORDER2} !important; color: #fff !important; }}
.Select-menu-outer {{ background: {CARD_BG2} !important; border-color: {BORDER2} !important; }}
.Select-option {{ background: {CARD_BG2} !important; color: #e0e8f4 !important; }}
.Select-option:hover {{ background: {BORDER2} !important; }}
.Select-value-label {{ color: #fff !important; }}
.Select-placeholder {{ color: {MUTED} !important; }}
table.dataframe th, .table th {{
    background: {CARD_BG2} !important; color: {ACCENT} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.78rem !important; text-transform: uppercase !important;
    border-color: {BORDER2} !important;
}}
table.dataframe td, .table td {{
    background: {CARD_BG} !important; color: #c8d8f0 !important;
    border-color: {BORDER} !important; font-size: 0.88rem !important;
}}
.table-hover > tbody > tr:hover > td {{ background: rgba(79,158,255,0.07) !important; }}
._dash-loading {{ color: {ACCENT} !important; }}
input[type=number]:focus, input[type=text]:focus {{
    outline: none; border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(79,158,255,0.15);
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
        "family=DM+Sans:wght@300;400;500&display=swap",
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
# SIDEBAR
# ============================================================================
def _build_sidebar() -> html.Div:
    nav_items = []
    for group_label, options in NAV_GROUPS.items():
        nav_items.append(html.Div(group_label, style={
            "color": MUTED, "fontSize": "0.65rem", "letterSpacing": "1.4px",
            "textTransform": "uppercase", "fontWeight": "600",
            "padding": "0.9rem 0.6rem 0.3rem",
            "fontFamily": "'Space Grotesk', sans-serif",
        }))
        for opt in options:
            idx = NAV_OPTIONS.index(opt)
            nav_items.append(html.Div(
                children=opt,
                id={"type": "nav-tab", "index": idx},
                className="nav-tab",
                style={
                    "padding": "0.55rem 0.9rem", "marginBottom": "2px",
                    "borderRadius": "7px", "cursor": "pointer",
                    "background": "transparent", "border": "1px solid transparent",
                    "color": "#8a9bb8", "fontSize": "0.82rem", "fontWeight": "400",
                    "whiteSpace": "nowrap", "overflow": "hidden",
                    "textOverflow": "ellipsis", "borderLeft": "3px solid transparent",
                },
            ))

    return html.Div([
        # Logo
        html.Div([
            html.Div([
                html.Span("S", style={"fontFamily": "'Space Grotesk'", "fontWeight": "700",
                                      "fontSize": "1.1rem", "color": ACCENT}),
                html.Span("TRAIVE", style={"fontFamily": "'Space Grotesk'", "fontWeight": "700",
                                           "fontSize": "1.1rem", "color": "#c8d8f0", "letterSpacing": "3px"}),
            ]),
            html.Div("Pricing Intelligence", style={"color": MUTED, "fontSize": "0.7rem", "marginTop": "2px"}),
        ], style={"marginBottom": "1.2rem", "paddingBottom": "1rem", "borderBottom": f"1px solid {BORDER}"}),

        html.Div("MODEL CONTROLS", style={
            "color": MUTED, "fontSize": "0.65rem", "letterSpacing": "1.4px",
            "fontWeight": "600", "fontFamily": "'Space Grotesk', sans-serif",
            "marginBottom": "0.8rem",
        }),

        html.Label("Record Count", style={"color": "#8a9bb8", "fontSize": "0.78rem", "marginBottom": "0.3rem", "display": "block"}),
        dcc.Input(
            id="n-records-input", type="number", value=2_000,
            min=300, max=10_000, step=500,
            style={**_input_style(), "marginBottom": "0.9rem", "fontSize": "0.88rem"},
        ),

        html.Label("Data Source", style={"color": "#8a9bb8", "fontSize": "0.78rem", "marginBottom": "0.4rem", "display": "block"}),
        dcc.RadioItems(
            id="data-mode-radio",
            options=[
                {"label": "  Synthetic Data",    "value": "synthetic"},
                {"label": "  Upload CSV / Excel", "value": "upload"},
            ],
            value="synthetic",
            labelStyle={"display": "block", "margin": "0.3rem 0", "color": "#8a9bb8", "fontSize": "0.82rem"},
        ),

        dcc.Upload(
            id="upload-csv",
            children=html.Div([
                html.Span("📁", style={"fontSize": "1.5rem", "marginRight": "8px"}),
                html.Span("Drop file or ", style={"color": MUTED}),
                html.A("browse", style={"color": ACCENT, "fontWeight": "600"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"}),
            style={
                "width": "100%", "height": "65px",
                "border": f"2px dashed {BORDER2}", "borderRadius": "10px",
                "textAlign": "center", "margin": "0.8rem 0",
                "color": MUTED, "fontSize": "0.82rem", "cursor": "pointer",
                "background": "rgba(79,158,255,0.05)",
                "display": "flex", "flexDirection": "column",
                "justifyContent": "center", "alignItems": "center",
            },
            multiple=False,
        ),

        html.Div(id="upload-feedback", style={
            "minHeight": "2rem", "fontSize": "0.8rem",
            "marginBottom": "0.5rem", "padding": "0.4rem",
            "borderRadius": "6px",
        }),

        html.Button(
            [html.Span("▶ ", style={"marginRight": "7px"}), "Build / Refresh Model"],
            id="build-model-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "0.8rem",
                "background": f"linear-gradient(135deg, {ACCENT} 0%, #2a6dd6 100%)",
                "border": "none", "borderRadius": "8px",
                "color": "#fff", "fontWeight": "600",
                "fontSize": "0.85rem", "cursor": "pointer",
                "boxShadow": f"0 4px 16px rgba(79,158,255,0.3)",
                "transition": "all 0.2s", "margin": "0.5rem 0",
                "fontFamily": "'Space Grotesk', sans-serif",
            },
        ),

        dcc.Loading(id="model-loading", type="circle", color=ACCENT,
                    children=html.Div(id="model-loading-output", style={"height": "4px"})),

        html.Hr(style={"borderColor": BORDER, "margin": "0.9rem 0"}),

        # Navigation
        html.Div(nav_items, style={"overflowY": "auto"}),

        html.Hr(style={"borderColor": BORDER, "margin": "0.9rem 0"}),

        # Export
        html.Button(
            [html.Span("⬇ ", style={"marginRight": "7px"}), "Export Results"],
            id="download-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "0.6rem", "background": "transparent",
                "border": f"1px solid {BORDER2}", "borderRadius": "7px",
                "color": MUTED, "fontSize": "0.78rem", "cursor": "pointer",
                "marginBottom": "0.5rem",
            },
        ),
        dcc.Download(id="download-component"),

        # Template download
        html.Button(
            [html.Span("📋 ", style={"marginRight": "7px"}), "Download Template"],
            id="download-template-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "0.6rem", "background": "transparent",
                "border": f"1px solid {BORDER2}", "borderRadius": "7px",
                "color": MUTED, "fontSize": "0.78rem", "cursor": "pointer",
            },
        ),
        dcc.Download(id="download-template-component"),

        html.Div([
            html.Span(f"v{APP_VERSION}", style={"color": MUTED, "fontSize": "0.7rem"}),
            html.Span(" · ", style={"color": BORDER2}),
            html.Span(CURRENT_DATE.strftime("%b %Y"), style={"color": MUTED, "fontSize": "0.7rem"}),
        ], style={"marginTop": "1rem", "textAlign": "center"}),

    ], style={
        "width": "260px", "minWidth": "260px", "background": SIDEBAR_BG,
        "padding": "1.2rem 0.9rem 1.5rem", "height": "100vh",
        "position": "fixed", "top": 0, "left": 0,
        "overflowY": "auto", "borderRight": f"1px solid {BORDER}", "zIndex": 100,
    })

# ============================================================================
# LAYOUT
# ============================================================================
app.layout = html.Div([
    dcc.Store(id="data-store"),
    dcc.Store(id="elasticity-store"),
    dcc.Store(id="win-model-store"),
    dcc.Store(id="active-tab-store", data=NAV_OPTIONS[0]),
    dcc.Store(id="competitor-data-store"),

    _build_sidebar(),

    html.Div([
        # Header
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
                html.Span(CURRENT_DATE.strftime("%A, %d %B %Y"), style={"color": MUTED, "fontSize": "0.8rem"}),
                html.Div([
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

        html.Div(id="kpi-bar", style={"padding": "1.2rem 2rem 0"}),
        html.Div(id="tab-content", style={"padding": "1.5rem 2rem 5rem"}),

    ], style={"marginLeft": "260px", "minHeight": "100vh", "background": DARK_BG}),
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
    active_idx   = triggered["index"]
    active_label = NAV_OPTIONS[active_idx]
    n = len(NAV_OPTIONS)

    base_style = {
        "padding": "0.55rem 0.9rem", "marginBottom": "2px", "borderRadius": "7px",
        "cursor": "pointer", "background": "transparent", "border": "1px solid transparent",
        "borderLeft": "3px solid transparent", "color": "#8a9bb8",
        "fontSize": "0.82rem", "fontWeight": "400",
        "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
    }
    active_style = {
        **base_style,
        "background": "linear-gradient(90deg, rgba(79,158,255,0.18), rgba(79,158,255,0.06))",
        "border": "1px solid rgba(79,158,255,0.3)",
        "borderLeft": f"3px solid {ACCENT}", "color": "#e8edf5", "fontWeight": "500",
    }

    styles  = [active_style if i == active_idx else base_style for i in range(n)]
    classes = ["nav-tab active" if i == active_idx else "nav-tab" for i in range(n)]
    page_title = html.H2(active_label, style={
        "fontFamily": "'Space Grotesk', sans-serif", "fontWeight": "700",
        "fontSize": "1.4rem", "margin": 0, "color": "#e8edf5",
    })
    return styles, classes, active_label, page_title


# ============================================================================
# CALLBACK: FILE UPLOAD
# ============================================================================
@app.callback(
    [Output("upload-feedback", "children", allow_duplicate=True),
     Output("upload-feedback", "style"),
     Output("data-store", "data", allow_duplicate=True)],
    Input("upload-csv", "contents"),
    [State("upload-csv", "filename")],
    prevent_initial_call=True,
)
def handle_file_upload(contents, filename):
    if not contents:
        raise PreventUpdate

    df, err = _parse_upload(contents, filename)
    if df is None:
        color = YELLOW if "Unsupported" in err else RED
        bg    = "rgba(245,166,35,0.15)" if "Unsupported" in err else "rgba(255,64,96,0.15)"
        return (
            html.Div([html.Span("⚠ ", style={"color": color}), html.Span(err)]),
            {"background": bg, "border": f"1px solid {color}", "borderRadius": "6px", "color": color},
            None,
        )

    try:
        min_rows = CSV_UPLOAD_SPEC.get("min_rows", 100)
        if len(df) < min_rows:
            return (
                html.Div([html.Span("⚠ ", style={"color": YELLOW}),
                          html.Span(f"Only {len(df)} rows. Minimum {min_rows} recommended.")]),
                {"background": "rgba(245,166,35,0.15)", "border": f"1px solid {YELLOW}",
                 "borderRadius": "6px", "color": YELLOW},
                None,
            )

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        if "confidence_score" not in df.columns:
            df["confidence_score"] = 1.0

        return (
            html.Div([html.Span("✓ ", style={"color": GREEN}),
                      html.Span(f"{len(df):,} rows · {len(df.columns)} cols")]),
            {"background": "rgba(46,232,154,0.1)", "border": f"1px solid {GREEN}",
             "borderRadius": "6px", "color": GREEN},
            df.to_json(date_format="iso", orient="split"),
        )
    except Exception as exc:
        return (
            html.Div([html.Span("✗ ", style={"color": RED}),
                      html.Span(f"Upload error: {str(exc)[:120]}")]),
            {"background": "rgba(255,64,96,0.15)", "border": f"1px solid {RED}",
             "borderRadius": "6px", "color": RED},
            None,
        )


# ============================================================================
# CALLBACK: DOWNLOAD TEMPLATE
# ============================================================================
@app.callback(
    Output("download-template-component", "data"),
    Input("download-template-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_template(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    example = [
        {"date": CURRENT_DATE.strftime("%Y-%m-%d"), "product": "Editorial Services – Standard",
         "customer_type": "Academic Publishers", "region": "North America",
         "actual_price": 2800, "volume": 10, "revenue": 28000, "cost": 11000,
         "deal_won": 1, "base_price": 2800, "discount_pct": 0,
         "competitor": "Aptara", "competitor_price": 2500, "source": "internal", "confidence_score": 1.0},
        {"date": (CURRENT_DATE - timedelta(days=30)).strftime("%Y-%m-%d"),
         "product": "Data Annotation – Basic", "customer_type": "STM Publishers", "region": "Europe",
         "actual_price": 420, "volume": 25, "revenue": 10500, "cost": 4500,
         "deal_won": 0, "base_price": 450, "discount_pct": 7,
         "competitor": "Innodata", "competitor_price": 400, "source": "scraped", "confidence_score": 0.85},
    ]
    buf = io.StringIO()
    pd.DataFrame(example).to_csv(buf, index=False)
    buf.seek(0)
    return dict(content=buf.getvalue(),
                filename=f"straive_template_{datetime.now().strftime('%Y%m%d')}.csv",
                type="text/csv")


# ============================================================================
# CALLBACK: BUILD MODEL
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
     State("competitor-data-store", "data")],
    prevent_initial_call=True,
)
def build_model(n_clicks, n_rows, mode, contents, filename, existing_df_json, competitor_json):
    if not n_clicks:
        raise PreventUpdate

    competitor_data = []
    if competitor_json:
        try:
            competitor_data = json.loads(competitor_json)
        except Exception:
            pass

    df       = None
    feedback = ""

    if mode == "upload" and contents:
        df, err = _parse_upload(contents, filename)
        if df is None:
            return [None, None, None, [],
                    html.Span(f"⚠ {err}", style={"color": YELLOW}), ""]
        if "confidence_score" not in df.columns:
            df["confidence_score"] = 1.0
        feedback = html.Span(f"✓ Uploaded {len(df):,} rows", style={"color": GREEN})
    else:
        n = min(int(n_rows or 2000), 10_000)
        df = DataGenerator().generate(n=n)
        feedback = html.Span(f"✓ Synthetic dataset ready ({n:,} rows)", style={"color": GREEN})

    if df is None or df.empty:
        return [None, None, None, [], feedback, ""]

    elast = ModelComparator.fit_elasticity_models(df)
    if competitor_data:
        elast = market_integrator.enhance_elasticity_with_market_data(elast, competitor_data)

    win_bundle = ModelComparator.fit_win_probability_model(df)
    win_meta   = {k: v for k, v in win_bundle.items() if k != "model"}

    df_json_str = df.to_json(date_format="iso", orient="split")
    kpi_bar     = _build_kpi_bar(df_json_str, win_meta=win_meta)

    return (df_json_str, _json_dumps(elast), _json_dumps(win_meta), kpi_bar, feedback, "")


# ============================================================================
# CALLBACK: EXPORT
# ============================================================================
@app.callback(
    Output("download-component", "data"),
    Input("download-btn", "n_clicks"),
    [State("data-store", "data"),
     State("elasticity-store", "data"),
     State("win-model-store", "data"),
     State("active-tab-store", "data")],
    prevent_initial_call=True,
)
def export_results(n_clicks, df_json, elast_json, win_json, active_tab):
    if not n_clicks:
        raise PreventUpdate
    n_rows = 0
    if df_json:
        try:
            n_rows = len(pd.read_json(df_json, orient="split"))
        except Exception:
            pass
    payload = {
        "exported":       datetime.now().isoformat(),
        "app_version":    APP_VERSION,
        "active_tab":     active_tab,
        "dataset_rows":   n_rows,
        "elasticity":     json.loads(elast_json) if elast_json else {},
        "win_model_meta": json.loads(win_json)   if win_json   else {},
        "products":       {k: {f: v for f, v in info.items() if f != "tags"}
                           for k, info in PRODUCT_CATALOG.items()},
    }
    return dict(
        content=json.dumps(payload, indent=2, default=str),
        filename=f"straive_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        type="application/json",
    )


# ============================================================================
# CALLBACK: TAB CONTENT ROUTER
# ============================================================================
@app.callback(
    Output("tab-content", "children"),
    [Input("active-tab-store", "data"),
     Input("data-store", "data"),
     Input("elasticity-store", "data"),
     Input("win-model-store", "data")],
)
def render_tab(active_tab, df_json, elast_json, win_json):
    if not active_tab:
        return _empty()

    df   = None
    elast = {}
    win_meta = {}

    if df_json:
        try:
            df = pd.read_json(df_json, orient="split")
        except Exception:
            pass
    if elast_json:
        try:
            elast = json.loads(elast_json)
        except Exception:
            pass
    if win_json:
        try:
            win_meta = json.loads(win_json)
        except Exception:
            pass

    tab = active_tab

    # ── 1. Executive Dashboard ───────────────────────────────────────────────
    if tab == "📊 Executive Dashboard":
        if df is None:
            return _empty()
        rev_by_seg = df.groupby("segment")["revenue"].sum().reset_index()
        fig_seg = px.pie(rev_by_seg, names="segment", values="revenue",
                         color="segment", color_discrete_map=SEGMENT_COLORS,
                         hole=0.45, title="Revenue by Segment")
        fig_seg.update_layout(**PLOTLY_DARK, height=340)

        monthly = df.copy()
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
        monthly_rev = monthly.groupby("month")["revenue"].sum().reset_index()
        fig_trend = px.area(monthly_rev, x="month", y="revenue", title="Monthly Revenue Trend",
                            color_discrete_sequence=[ACCENT])
        fig_trend.update_layout(**PLOTLY_DARK, height=260)

        win_data = df.groupby("segment")["deal_won"].mean().reset_index()
        win_data.columns = ["segment", "win_rate"]
        fig_win = px.bar(win_data, x="segment", y="win_rate", color="segment",
                         color_discrete_map=SEGMENT_COLORS,
                         title="Win Rate by Segment")
        fig_win.update_layout(**PLOTLY_DARK, height=300, yaxis_tickformat=".0%")

        margin_data = df.groupby("segment")["margin_pct"].mean().reset_index()
        fig_margin = px.bar(margin_data, x="segment", y="margin_pct", color="segment",
                            color_discrete_map=SEGMENT_COLORS, title="Avg Margin % by Segment")
        fig_margin.update_layout(**PLOTLY_DARK, height=300)

        return html.Div([
            _section("Revenue Overview"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_seg), md=5),
                dbc.Col(dcc.Graph(figure=fig_trend), md=7),
            ], className="g-3"),
            _section("Performance Metrics"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_win), md=6),
                dbc.Col(dcc.Graph(figure=fig_margin), md=6),
            ], className="g-3"),
        ])

    # ── 2. Elasticity Analysis ───────────────────────────────────────────────
    elif tab == "🔍 Elasticity Analysis":
        if not elast:
            return _empty("Build a model first to see elasticity estimates.")
        rows = []
        for seg, data in elast.items():
            rows.append({
                "Segment":    seg,
                "Elasticity": round(data.get("elasticity", 0), 4),
                "CI Low":     round(data.get("elasticity_ci_lo", 0), 4),
                "CI High":    round(data.get("elasticity_ci_hi", 0), 4),
                "R²":         round(data.get("r_squared", 0), 4),
                "p-value":    round(data.get("p_value", 1), 4),
                "Obs":        data.get("n_obs", 0),
                "Best Model": data.get("best_model", "—"),
                "Confidence": round(data.get("confidence", 0), 3),
            })
        elast_df = pd.DataFrame(rows)
        fig = px.bar(elast_df, x="Segment", y="Elasticity", color="Segment",
                     color_discrete_map=SEGMENT_COLORS,
                     error_y=elast_df["Elasticity"] - elast_df["CI Low"],
                     error_y_minus=elast_df["CI High"] - elast_df["Elasticity"],
                     title="Price Elasticity by Segment (with 95% CI)")
        fig.update_layout(**PLOTLY_DARK, height=420)
        return html.Div([
            _section("Price Elasticity Estimates"),
            dcc.Graph(figure=fig),
            _section("Model Details"),
            dbc.Table.from_dataframe(
                elast_df, striped=True, bordered=True, hover=True, responsive=True,
                style={"fontSize": "0.78rem"},
            ),
            dbc.Alert(
                "Elasticity < -1 means demand is elastic (price-sensitive). "
                "Higher confidence = more reliable estimate.",
                color="info", className="mt-3", style={"fontSize": "0.82rem"},
            ),
        ])

    # ── 3. Optimal Pricing ───────────────────────────────────────────────────
    elif tab == "💡 Optimal Pricing":
        if df is None or not elast:
            return _empty()
        objectives = ["max_revenue", "max_profit", "max_margin", "vs_competitor"]
        results = []
        for prod in list(PRODUCT_CATALOG.keys())[:12]:
            seg = PRODUCT_CATALOG[prod]["segment"]
            for obj in objectives[:2]:  # keep it lean — show revenue + profit
                res = PricingOptimizer.calculate(
                    product=prod, segment=seg,
                    elasticity_results=elast, objective=obj,
                )
                results.append({
                    "Product":       prod[:30],
                    "Segment":       seg,
                    "Objective":     obj.replace("_", " ").title(),
                    "Base Price":    f"${res['base_price']:,.0f}",
                    "Optimal Price": f"${res['optimal_price']:,.0f}",
                    "Change %":      f"{res['price_change_pct']:+.1f}%",
                    "Revenue Lift":  f"{res['revenue_lift']:+.1f}%",
                    "Profit Lift":   f"{res['profit_lift']:+.1f}%",
                    "Margin %":      f"{res['optimal_margin']:.1f}%",
                })
        res_df = pd.DataFrame(results)
        pivot = res_df[res_df["Objective"] == "Max Revenue"][["Product", "Optimal Price", "Change %", "Revenue Lift"]].copy()
        fig = px.bar(
            res_df[res_df["Objective"] == "Max Revenue"],
            x="Product", y="Revenue Lift",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            title="Revenue Lift at Optimal Price (Max Revenue objective)",
        )
        fig.update_layout(**PLOTLY_DARK, height=400, xaxis_tickangle=-45)
        return html.Div([
            _section("Optimal Pricing Results"),
            dcc.Graph(figure=fig),
            _section("Optimization Table"),
            dbc.Table.from_dataframe(
                res_df, striped=True, bordered=True, hover=True,
                responsive=True, style={"fontSize": "0.75rem"},
            ),
        ])

    # ── 4. Revenue Simulator ─────────────────────────────────────────────────
    elif tab == "📈 Revenue Simulator":
        if df is None or not elast:
            return _empty()
        sim_df = SimulationEngine.simulate_revenue_scenarios(df, elast, n_scenarios=200)
        fig_scatter = px.scatter(
            sim_df, x="price_change_pct", y="revenue_delta",
            color="margin_pct", color_continuous_scale="RdYlGn",
            title="Price Change → Revenue Delta",
            labels={"price_change_pct": "Price Change %", "revenue_delta": "Revenue Δ ($)"},
        )
        fig_scatter.update_layout(**PLOTLY_DARK, height=420)
        fig_profit = px.scatter(
            sim_df, x="revenue_delta", y="profit_delta",
            color="price_change_pct", color_continuous_scale="Blues",
            title="Revenue Δ vs Profit Δ",
        )
        fig_profit.update_layout(**PLOTLY_DARK, height=400)
        return html.Div([
            _section("Monte Carlo Revenue Simulation"),
            dbc.Alert(f"Ran {len(sim_df)} price-volume scenarios.", color="info",
                      style={"fontSize": "0.82rem"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_scatter), md=7),
                dbc.Col(dcc.Graph(figure=fig_profit), md=5),
            ], className="g-3"),
        ])

    # ── 5. Price-Volume Curves ───────────────────────────────────────────────
    elif tab == "🎯 Price-Volume Curves":
        products = list(PRODUCT_CATALOG.keys())
        return html.Div([
            _section("Price-Volume-Revenue Curves"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Product", style={"color": MUTED, "fontSize": "0.82rem"}),
                    dcc.Dropdown(
                        id="pvc-product", options=[{"label": p, "value": p} for p in products],
                        value=products[0], clearable=False, style=_dd_style(),
                    ),
                ], md=4),
            ], className="mb-3"),
            dcc.Graph(id="pvc-graph"),
        ])

    # ── 6. Competitive Positioning ───────────────────────────────────────────
    elif tab == "⚔️ Competitive Positioning":
        if df is None:
            return _empty()
        scores = []
        for prod in df["product"].unique():
            s = CompetitiveAnalyzer.get_price_score(df, prod)
            if s:
                scores.append(s)
        if not scores:
            return _empty("Not enough data for competitive analysis.")
        score_df = pd.DataFrame(scores)
        fig = px.scatter(
            score_df, x="market_avg", y="our_avg_price",
            size=[30] * len(score_df),
            color="price_index", color_continuous_scale="RdYlGn_r",
            hover_name="product", title="Our Price vs Market Average",
            labels={"market_avg": "Market Avg ($)", "our_avg_price": "Our Price ($)"},
        )
        fig.add_shape(type="line", x0=score_df["market_avg"].min(),
                      y0=score_df["market_avg"].min(),
                      x1=score_df["market_avg"].max(),
                      y1=score_df["market_avg"].max(),
                      line=dict(color=MUTED, dash="dash"))
        fig.update_layout(**PLOTLY_DARK, height=460)

        trends = CompetitiveAnalyzer.track_price_trends(df)
        trend_rows = [{"Competitor": k, "Avg Price": f"${v['avg_price']:,.0f}",
                       "Latest Price": f"${v['latest_price']:,.0f}",
                       "Trend": "↑" if v["price_trend"] > 0 else "↓", "Obs": v["n_obs"]}
                      for k, v in trends.items()]
        return html.Div([
            _section("Price Positioning"),
            dcc.Graph(figure=fig),
            _section("Competitor Price Trends"),
            dbc.Table.from_dataframe(pd.DataFrame(trend_rows), striped=True, bordered=True,
                                     hover=True, responsive=True, style={"fontSize": "0.78rem"}),
        ])

    # ── 7. Regional Pricing ──────────────────────────────────────────────────
    elif tab == "🌍 Regional Pricing":
        if df is None:
            return _empty()
        reg_df = df.groupby("region").agg(
            avg_price=("actual_price", "mean"),
            revenue=("revenue", "sum"),
            win_rate=("deal_won", "mean"),
            margin=("margin_pct", "mean"),
        ).reset_index()
        fig = px.bar(reg_df, x="region", y="avg_price", color="region",
                     title="Average Price by Region")
        fig.update_layout(**PLOTLY_DARK, height=380)
        fig2 = px.scatter(reg_df, x="avg_price", y="win_rate", size="revenue",
                          color="region", hover_name="region",
                          title="Price vs Win Rate by Region")
        fig2.update_layout(**PLOTLY_DARK, height=380)
        return html.Div([
            _section("Regional Pricing Overview"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig), md=6),
                dbc.Col(dcc.Graph(figure=fig2), md=6),
            ], className="g-3"),
        ])

    # ── 8. Segment Intelligence ──────────────────────────────────────────────
    elif tab == "👥 Segment Intelligence":
        if df is None:
            return _empty()
        seg_df = df.groupby("customer_type").agg(
            avg_price=("actual_price", "mean"),
            revenue=("revenue", "sum"),
            win_rate=("deal_won", "mean"),
            margin=("margin_pct", "mean"),
            deals=("deal_won", "count"),
        ).reset_index()
        fig = px.bar(seg_df, x="customer_type", y="revenue", color="customer_type",
                     title="Revenue by Customer Segment")
        fig.update_layout(**PLOTLY_DARK, height=380, xaxis_tickangle=-30)
        fig2 = px.scatter(seg_df, x="avg_price", y="win_rate", size="revenue",
                          color="customer_type", hover_name="customer_type",
                          title="Price Sensitivity vs Win Rate")
        fig2.update_layout(**PLOTLY_DARK, height=380)
        return html.Div([
            _section("Customer Segment Intelligence"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig), md=6),
                dbc.Col(dcc.Graph(figure=fig2), md=6),
            ], className="g-3"),
        ])

    # ── 9. What-If Scenarios ─────────────────────────────────────────────────
    elif tab == "🔧 What-If Scenarios":
        if df is None or not elast:
            return _empty()
        products = list(PRODUCT_CATALOG.keys())[:10]
        sliders = []
        for prod in products:
            info  = PRODUCT_CATALOG[prod]
            base  = info["base_price"]
            lo    = info.get("min_price", base * 0.5)
            hi    = info.get("max_price", base * 2.0)
            sliders.append(html.Div([
                html.Div([
                    html.Span(prod[:32], style={"color": "#c8d8f0", "fontSize": "0.8rem"}),
                    html.Span(f"  Base: ${base:,.0f}", style={"color": MUTED, "fontSize": "0.72rem"}),
                ]),
                dcc.Slider(
                    id={"type": "whatif-slider", "index": prod},
                    min=int(lo), max=int(hi), step=max(1, int((hi - lo) / 100)),
                    value=int(base), marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={"marginBottom": "1rem"}))

        return html.Div([
            _section("What-If Price Scenarios"),
            dbc.Row([
                dbc.Col([
                    html.Div(sliders),
                    dbc.Row([
                        dbc.Col(html.Button("↑ +10%", id="whatif-preset-up10", n_clicks=0,
                                            style={"width": "100%", "background": "transparent",
                                                   "border": f"1px solid {GREEN}", "color": GREEN,
                                                   "borderRadius": "6px", "padding": "0.4rem"}), md=4),
                        dbc.Col(html.Button("↓ −10%", id="whatif-preset-down10", n_clicks=0,
                                            style={"width": "100%", "background": "transparent",
                                                   "border": f"1px solid {RED}", "color": RED,
                                                   "borderRadius": "6px", "padding": "0.4rem"}), md=4),
                        dbc.Col(html.Button("⟳ Reset", id="whatif-preset-reset", n_clicks=0,
                                            style={"width": "100%", "background": "transparent",
                                                   "border": f"1px solid {MUTED}", "color": MUTED,
                                                   "borderRadius": "6px", "padding": "0.4rem"}), md=4),
                    ], className="g-2 mt-2"),
                ], md=5),
                dbc.Col(dcc.Graph(id="whatif-output-graph"), md=7),
            ], className="g-3"),
        ])

    # ── 10. Win-Rate Analysis ────────────────────────────────────────────────
    elif tab == "🤝 Win-Rate Analysis":
        if df is None:
            return _empty()
        wr_df = df.groupby("product").agg(
            win_rate=("deal_won", "mean"),
            avg_price=("actual_price", "mean"),
            avg_discount=("discount_pct", "mean"),
            deals=("deal_won", "count"),
        ).reset_index()
        wr_df["segment"] = wr_df["product"].map(lambda p: PRODUCT_CATALOG.get(p, {}).get("segment", ""))
        fig = px.scatter(
            wr_df, x="avg_discount", y="win_rate", color="segment",
            size="deals", hover_name="product",
            color_discrete_map=SEGMENT_COLORS,
            title="Discount % vs Win Rate",
            labels={"avg_discount": "Avg Discount %", "win_rate": "Win Rate"},
        )
        fig.update_layout(**PLOTLY_DARK, height=460)
        return html.Div([
            _section("Win-Rate Analysis"),
            dcc.Graph(figure=fig),
            dbc.Alert(
                f"Overall win rate: {df['deal_won'].mean()*100:.1f}%  |  "
                f"Avg discount: {df['discount_pct'].mean():.1f}%",
                color="info", style={"fontSize": "0.82rem"},
            ),
        ])

    # ── 11. Margin Waterfall ─────────────────────────────────────────────────
    elif tab == "📉 Margin Waterfall":
        if df is None:
            return _empty()
        wf_df = MarginWaterfallBuilder.build(df)
        if wf_df.empty:
            return _empty("Not enough data for waterfall.")
        fig = go.Figure(go.Waterfall(
            name="Margin", orientation="v",
            x=wf_df["label"].tolist(),
            y=wf_df["value"].tolist(),
            connector={"line": {"color": BORDER2}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker":    {"color": ACCENT}},
        ))
        fig.update_layout(**PLOTLY_DARK, title="Revenue & Margin Waterfall", height=450)
        return html.Div([
            _section("Margin Waterfall"),
            dcc.Graph(figure=fig),
            dbc.Table.from_dataframe(
                wf_df.assign(value=wf_df["value"].apply(lambda v: f"${v:,.0f}")),
                striped=True, bordered=True, hover=True,
                responsive=True, style={"fontSize": "0.8rem"},
            ),
        ])

    # ── 12. Product Portfolio ────────────────────────────────────────────────
    elif tab == "📦 Product Portfolio":
        if df is None or not elast:
            return _empty()
        port_df = PortfolioScorer.score_products(df, elast)
        if port_df.empty:
            return _empty("No portfolio data available.")
        fig = px.scatter(
            port_df, x="margin_pct", y="win_rate",
            size="revenue", color="segment",
            color_discrete_map=SEGMENT_COLORS,
            hover_name="product",
            text="product",
            title="Portfolio Matrix: Margin vs Win Rate",
        )
        fig.update_traces(textposition="top center", textfont_size=8)
        fig.update_layout(**PLOTLY_DARK, height=520)

        show_cols = ["product", "segment", "revenue", "margin_pct", "win_rate",
                     "growth_pct", "score", "adjusted_score"]
        show_cols = [c for c in show_cols if c in port_df.columns]
        return html.Div([
            _section("Portfolio Scoring"),
            dcc.Graph(figure=fig),
            _section("Product Scores"),
            dbc.Table.from_dataframe(
                port_df[show_cols].head(20), striped=True, bordered=True,
                hover=True, responsive=True, style={"fontSize": "0.75rem"},
            ),
        ])

    # ── 13. Risk & Sensitivity ───────────────────────────────────────────────
    elif tab == "⚠️ Risk & Sensitivity":
        if df is None or not elast:
            return _empty()
        sensitivity_rows = []
        for seg, data in elast.items():
            el = data.get("elasticity", -1.2)
            for delta in [-0.20, -0.10, 0.0, 0.10, 0.20]:
                vol_change = (1 + delta) ** el - 1
                rev_change = (1 + delta) * (1 + vol_change) - 1
                sensitivity_rows.append({
                    "Segment": seg,
                    "Price Δ": f"{delta*100:+.0f}%",
                    "Volume Δ": f"{vol_change*100:+.1f}%",
                    "Revenue Δ": f"{rev_change*100:+.1f}%",
                    "Risk":     "🔴 High" if abs(rev_change) > 0.10 else ("🟡 Med" if abs(rev_change) > 0.05 else "🟢 Low"),
                })
        sens_df = pd.DataFrame(sensitivity_rows)
        pivot = sens_df[sens_df["Segment"].isin(list(elast.keys())[:4])]
        return html.Div([
            _section("Price Sensitivity Analysis"),
            dbc.Alert("Revenue response to ±10–20% price changes across segments.",
                      color="warning", style={"fontSize": "0.82rem"}),
            dbc.Table.from_dataframe(
                sens_df, striped=True, bordered=True, hover=True,
                responsive=True, style={"fontSize": "0.78rem"},
            ),
        ])

    # ── 14. Seasonality & Trends ─────────────────────────────────────────────
    elif tab == "🗓️ Seasonality & Trends":
        if df is None:
            return _empty()
        df2 = df.copy()
        df2["month_num"] = pd.to_datetime(df2["date"]).dt.month
        monthly_avg = df2.groupby("month_num")["revenue"].mean().reset_index()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly_avg["month"] = monthly_avg["month_num"].map(month_names)

        fig = px.bar(monthly_avg, x="month", y="revenue",
                     title="Average Revenue by Month (Seasonality)",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(**PLOTLY_DARK, height=380)

        season_known = pd.DataFrame([
            {"Month": month_names[m], "Multiplier": v,
             "Effect": "Peak" if v > 1.05 else ("Trough" if v < 0.92 else "Normal")}
            for m, v in MONTHLY_SEASONALITY.items()
        ])
        return html.Div([
            _section("Seasonality Analysis"),
            dcc.Graph(figure=fig),
            _section("Known Seasonal Factors"),
            dbc.Table.from_dataframe(
                season_known, striped=True, bordered=True, hover=True,
                responsive=True, style={"fontSize": "0.78rem"},
            ),
        ])

    # ── 15. Revenue Forecast ─────────────────────────────────────────────────
    elif tab == "🔮 Revenue Forecast":
        if df is None:
            return _empty()
        forecast = RevenueForecaster.forecast(df, horizon=12)
        hist   = forecast.get("historical", {})
        fcast  = forecast.get("forecast", {})

        hist_vals  = list(hist.values())
        hist_dates = list(hist.keys())
        fcast_vals  = list(fcast.values())
        fcast_dates = list(fcast.keys())

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_vals, mode="lines+markers",
            name="Historical", line=dict(color=ACCENT, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=fcast_dates, y=fcast_vals, mode="lines+markers",
            name="Forecast", line=dict(color=GREEN, width=2.5, dash="dot"),
        ))
        fig.update_layout(**PLOTLY_DARK, title="12-Month Revenue Forecast",
                          height=480, xaxis_title="Month", yaxis_title="Revenue ($)")

        return html.Div([
            _section("Revenue Forecast"),
            dbc.Alert(
                f"Method: {forecast.get('method', 'ensemble')}  |  "
                f"History: {forecast.get('n_history', 0)} months  |  "
                f"Trend slope: ${forecast.get('trend_slope', 0):+,.0f}/month",
                color="info", style={"fontSize": "0.82rem"},
            ),
            dcc.Graph(figure=fig),
        ])

    return _empty(f"Tab not found: {tab}")


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
    info     = PRODUCT_CATALOG[product]
    seg      = info["segment"]
    elast    = json.loads(elast_json) if elast_json else {}
    elast_val = elast.get(seg, {}).get("elasticity", -1.2)
    base     = info["base_price"]
    cost     = info["cost"]
    prices   = np.linspace(base * 0.4, base * 2.2, 200)
    volumes  = np.maximum(0.01, (prices / base) ** elast_val)
    revenues = prices * volumes
    profits  = (prices - cost) * volumes

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue",
                              line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit",
                              line=dict(color=GREEN, width=2.5)))
    fig.add_trace(go.Scatter(x=prices, y=volumes * base / 2, name="Volume (scaled)",
                              line=dict(color=YELLOW, width=1.5, dash="dot")))
    fig.add_vline(x=base, line_dash="dash", line_color=MUTED, annotation_text="Base Price")
    fig.update_layout(**PLOTLY_DARK, title=f"Price-Volume-Revenue – {product}",
                      height=480, xaxis_title="Price ($)", yaxis_title="Value ($)")
    return fig


# ============================================================================
# CALLBACK: WHAT-IF PRESETS
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
        else:
            val = base
        info = PRODUCT_CATALOG.get(prod, {})
        val = float(np.clip(val, info.get("min_price", base * 0.5), info.get("max_price", base * 2.0)))
        new_values.append(int(round(val)))
    return new_values


# ============================================================================
# CALLBACK: WHAT-IF GRAPH
# ============================================================================
@app.callback(
    Output("whatif-output-graph", "figure"),
    Input({"type": "whatif-slider", "index": ALL}, "value"),
    [State({"type": "whatif-slider", "index": ALL}, "id"),
     State("elasticity-store", "data")],
)
def update_whatif_graph(values, slider_ids, elast_json):
    if not values or not slider_ids:
        raise PreventUpdate
    elast = json.loads(elast_json) if elast_json else {}
    rows = []
    for sid, val in zip(slider_ids, values):
        if val is None:
            continue
        prod = sid["index"]
        info = PRODUCT_CATALOG.get(prod, {})
        base = info.get("base_price", 1)
        cost = info.get("cost", base * 0.4)
        seg  = info.get("segment", "")
        el   = elast.get(seg, {}).get("elasticity", -1.2)
        price_ratio = val / base
        demand = max(0.01, price_ratio ** el)
        revenue = val * demand
        profit  = (val - cost) * demand
        rows.append({
            "product": prod[:20],
            "price":   val,
            "revenue": revenue,
            "profit":  profit,
            "pct_change": (val - base) / base * 100,
        })
    if not rows:
        raise PreventUpdate
    rdf = pd.DataFrame(rows)
    fig = px.bar(rdf, x="product", y="revenue", color="pct_change",
                 color_continuous_scale="RdYlGn",
                 title="Projected Revenue at Adjusted Prices")
    fig.update_layout(**PLOTLY_DARK, height=380, xaxis_tickangle=-30)
    return fig


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"{APP_TITLE}")
    print(f"Version: {APP_VERSION}  (cloud-light)")
    print(f"URL: http://127.0.0.1:8050")
    print(f"{'='*55}\n")
    app.run(debug=True, host="0.0.0.0", port=8050)

# Gunicorn / Procfile entry point
server = app.server
