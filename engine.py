"""
STRAIVE Pricing Platform – Engine Module  (v4.1)

Improvements over v4.0
-----------------------
DataGenerator
  • Extracted magic multipliers to named class constants (BASE_DEMAND, etc.)
  • Clamped `confidence` computation to a clearly documented formula
  • `contract_months` uses np.array so rng.choice p-weights broadcast correctly

ModelComparator.fit_elasticity_models
  • CV loop re-uses a *clone* of the model per fold (avoids fitting on full data
    leaking into subsequent fold fits for stateful tree models)
  • sample_weights branching extracted to `_supports_sample_weight` helper
  • Avoids double import of pandas inside predict_win_probability

ModelComparator.fit_win_probability_model
  • Same clone-per-fold pattern
  • Guards scale_pos_weight divide-by-zero when all labels are the same class
  • Early-exits with empty dict when n_obs < MIN_WIN_RATE_OBS (constant)

PricingOptimizer.calculate
  • Logs a warning (not a silent clamp) when lower >= upper
  • `margin` objective guards p <= 0 explictly
  • `vs_competitor` safe when competitor_prices is empty/None

SimulationEngine.simulate_revenue_scenarios
  • Uses only 3 Sobol dimensions (d=3); comp_noise removed (was unused scalar)
  • Docstring clarifies dimension mapping

CompetitiveAnalyzer.get_price_score
  • Guards division by zero when comp_df is empty before pct_rank computation
  • `weighted_confidence` safe when Confidence sum is 0

CompetitiveAnalyzer.track_price_trends
  • Uses pd.to_datetime with utc=False and explicit errors="coerce"
  • Slope gracefully absent when all prices identical (polyfit guard)

RevenueForecaster.forecast
  • ExponentialSmoothing fitted once; forecasts pre-computed outside h-loop
    (was already partially done – now consistently applied to single-method path)
  • Fallback single-method path also uses pre-computed values
  • f-string in log.warning replaced with %-style (lazy evaluation)

DealScorer
  • SCORE_WEIGHTS extracted to class constant (single source of truth)
  • `rating` lookup extracted to `get_rating` classmethod
  • Validates `proposed_price > 0` before margin computation

MarginWaterfallBuilder.build
  • Guard for empty sub (returns empty DataFrame early)
  • `discount_amount` capped to total_revenue to prevent negative net_revenue

PortfolioScorer.score_products
  • SCORING_WEIGHTS lookup uses config import (no magic dict)
  • `market_position` sigmoid documented with explicit formula comment

MarketIntelligenceIntegrator.enhance_elasticity_with_market_data
  • Uses per-segment base_price from PRODUCT_CATALOG instead of first-key proxy

General
  • `__all__` sorted and complete
  • Consistent `%`-style log messages (avoid string interpolation at call site)
  • All bare `except Exception` narrowed where possible
"""

from __future__ import annotations

import logging
import sys
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge,
)
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

from config import (
    CHANNEL_LIST, CHANNEL_PRICE_ADJ, CHANNEL_WEIGHTS,
    COMPETITORS, COMPETITOR_COLORS,
    CURRENT_DATE, CUSTOMER_SEGMENTS,
    DATA_COLLECTION_GUIDANCE, FORECAST_CONFIG,
    MARGIN_WATERFALL_BUCKETS, MONTHLY_SEASONALITY,
    PRODUCT_CATALOG, REGIONS, SCORING_WEIGHTS,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_TREE_MODELS = frozenset({"GradientBoosting", "RandomForest", "XGBoost"})


def _supports_sample_weight(name: str) -> bool:
    """Return True for models whose fit() accepts sample_weight."""
    return name in _TREE_MODELS


def _clone(model: Any) -> Any:
    """Shallow-clone a sklearn estimator or Pipeline via deepcopy."""
    return deepcopy(model)


# ============================================================================
# DATA GENERATOR
# ============================================================================

class DataGenerator:
    """
    Generates realistic synthetic transaction data with confidence scoring.

    Constants
    ---------
    BASE_DEMAND_SCALE : float
        Baseline demand units per row before elasticity / seasonality adjustments.
    CONFIDENCE_DECAY  : float
        Per-day decay applied to confidence_score (older data = less confident).
    MAX_HISTORY_DAYS  : int
        Horizon used for exponential day-lag sampling.
    """

    BASE_DEMAND_SCALE: float = 8.0
    CONFIDENCE_DECAY: float = 0.3 / 2_000   # 0.3 spread over ~2 000 days
    CONFIDENCE_MIN: float = 0.30
    CONFIDENCE_OUTLIER_PENALTY: float = 0.80  # multiplier when |price_ratio - 1| > 0.5
    PRICE_OUTLIER_THRESHOLD: float = 0.50
    MAX_HISTORY_DAYS: int = 730
    CONTRACT_PROBS: Tuple[float, float, float] = (0.25, 0.55, 0.20)

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, n: int = 3_600, include_confidence: bool = True) -> pd.DataFrame:
        """
        Generate *n* synthetic transaction rows.

        Parameters
        ----------
        n                 : number of rows to generate
        include_confidence: whether to compute and store confidence_score

        Returns
        -------
        pd.DataFrame sorted by date, reset index.
        """
        rng = self._rng
        products   = list(PRODUCT_CATALOG.keys())
        segments   = list(CUSTOMER_SEGMENTS.keys())
        regions    = list(REGIONS.keys())
        competitors = list(COMPETITORS.keys())
        start      = CURRENT_DATE - timedelta(days=self.MAX_HISTORY_DAYS)
        rows: List[Dict[str, Any]] = []

        for _ in range(n):
            prod  = rng.choice(products)
            seg   = rng.choice(segments)
            reg   = rng.choice(regions)
            comp  = rng.choice(competitors)
            info     = PRODUCT_CATALOG[prod]
            seg_inf  = CUSTOMER_SEGMENTS[seg]
            reg_inf  = REGIONS[reg]
            comp_inf = COMPETITORS[comp]
            channel  = rng.choice(CHANNEL_LIST, p=CHANNEL_WEIGHTS)

            base  = info["base_price"]
            cost  = info["cost"]
            min_p = info.get("min_price", base * 0.50)
            max_p = info.get("max_price", base * 2.00)

            # Competitor-influenced price
            comp_price = base * comp_inf["relative_price"] * (
                1.0 + float(rng.normal(0, 0.05))
            )
            noise      = float(rng.normal(0, 0.10))
            channel_adj = CHANNEL_PRICE_ADJ.get(channel, 0.0)
            price = base * (1.0 + noise + channel_adj) * (0.85 + 0.30 * float(rng.random()))
            price = float(np.clip(price, min_p, max_p))

            # Demand
            elasticity   = -1.2 - 0.5 * (1.0 - seg_inf["price_sensitivity"])
            days_ago     = int(rng.exponential(200))
            date         = start + timedelta(days=min(days_ago, self.MAX_HISTORY_DAYS))
            season_mult  = MONTHLY_SEASONALITY.get(date.month, 1.0)
            demand_base  = (
                seg_inf["volume_multiplier"]
                * reg_inf["demand_index"]
                * self.BASE_DEMAND_SCALE
                * season_mult
            )
            price_ratio  = price / base
            demand_raw   = demand_base * (price_ratio ** elasticity) + float(rng.normal(0, 1.5))
            demand       = max(1, int(demand_raw))

            # Financials
            revenue   = price * demand
            cost_tot  = cost * demand
            margin    = (revenue - cost_tot) / revenue if revenue > 0.0 else 0.0

            # Win probability
            win_base = 0.62 - 0.25 * max(0.0, price_ratio - 1.0)
            win_base += 0.05 * (1.0 - seg_inf["price_sensitivity"])
            win_base += 0.03 * seg_inf["loyalty"]
            if price < comp_price:
                win_base += 0.08 * (comp_price - price) / comp_price
            deal_won = int(float(rng.random()) < min(0.95, max(0.05, win_base)))

            # Contract metadata
            avg_mo = seg_inf["avg_contract_months"]
            contract_months = int(rng.choice(
                np.array([max(1, avg_mo // 2), avg_mo, avg_mo * 2]),
                p=list(self.CONTRACT_PROBS),
            ))
            renewal_flag = int(float(rng.random()) < seg_inf["loyalty"])
            cycle_days   = max(1, int(rng.normal(45 + 10 * info["complexity"], 15)))
            discount_pct = max(0.0, (base - price) / base * 100)
            churn_prob   = (1.0 - seg_inf["loyalty"]) * (1.0 - margin) * 1.4
            churn_flag   = int(float(rng.random()) < min(0.8, max(0.0, churn_prob)))

            # Confidence
            confidence = 0.9 - days_ago * self.CONFIDENCE_DECAY
            if abs(price_ratio - 1.0) > self.PRICE_OUTLIER_THRESHOLD:
                confidence *= self.CONFIDENCE_OUTLIER_PENALTY
            confidence = max(self.CONFIDENCE_MIN, min(1.0, confidence))

            rows.append({
                "date":             date,
                "product":          prod,
                "segment":          info["segment"],
                "customer_type":    seg,
                "region":           reg,
                "channel":          channel,
                "base_price":       round(base, 2),
                "actual_price":     round(price, 2),
                "discount_pct":     round(discount_pct, 1),
                "volume":           demand,
                "revenue":          round(revenue, 2),
                "cost":             round(cost_tot, 2),
                "margin_pct":       round(margin * 100, 2),
                "deal_won":         deal_won,
                "competitor":       comp,
                "competitor_price": round(comp_price, 2),
                "contract_months":  contract_months,
                "renewal_flag":     renewal_flag,
                "sales_cycle_days": cycle_days,
                "churn_flag":       churn_flag,
                "complexity":       info["complexity"],
                "confidence_score": round(confidence, 3) if include_confidence else 1.0,
                "source":           "synthetic",
            })

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        log.info("DataGenerator: %d rows generated", len(df))
        return df


# ============================================================================
# MODEL COMPARATOR
# ============================================================================

class ModelComparator:
    """
    Fits multiple regression / classification models with confidence weighting.

    Key design decisions
    --------------------
    * Each CV fold uses deepcopy(model) so stateful tree models never leak
      training data from earlier folds into later ones.
    * sample_weight is passed only to models that declare support
      (_TREE_MODELS) to avoid silent misuse with Pipeline wrappers.
    """

    # Minimum observations per segment before we skip elasticity modelling
    MIN_ELASTICITY_OBS: int = 30
    # Minimum observations before we skip win-rate modelling
    MIN_WIN_RATE_OBS: int = 50

    # ------------------------------------------------------------------ #
    #  Elasticity                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def fit_elasticity_models(
        df: pd.DataFrame,
        use_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit elasticity models per segment.

        Returns a dict keyed by segment name with elasticity estimates and
        cross-validated RMSE for each candidate model.
        """
        results: Dict[str, Any] = {}

        for seg in df["segment"].unique():
            sub = df[(df["segment"] == seg) & (df["volume"] > 0)].copy()
            if len(sub) < ModelComparator.MIN_ELASTICITY_OBS:
                log.warning(
                    "Segment %s: only %d rows — skipping elasticity fitting",
                    seg, len(sub),
                )
                continue

            try:
                sub["log_price"]    = np.log(sub["actual_price"])
                sub["log_discount"] = np.log1p(sub["discount_pct"])
                sub["log_volume"]   = np.log(sub["volume"])

                feat_cols = ["log_price", "log_discount"]
                X         = sub[feat_cols].values
                y         = sub["log_volume"].values
                X_df      = sub[feat_cols].copy()   # for XGBoost 

                sample_weights = (
                    sub["confidence_score"].values
                    if use_confidence and "confidence_score" in sub.columns
                    else None
                )

                candidate_models: Dict[str, Any] = {
                    "OLS":             LinearRegression(),
                    "Ridge":           Ridge(alpha=1.0),
                    "Lasso":           Lasso(alpha=0.05, max_iter=5_000),
                    "ElasticNet":      ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5_000),
                    "BayesianRidge":   BayesianRidge(),
                    "GradientBoosting": GradientBoostingRegressor(
                        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42,
                    ),
                    "RandomForest":    RandomForestRegressor(
                        n_estimators=80, max_depth=5, random_state=42, n_jobs=-1,
                    ),
                    "XGBoost":         xgb.XGBRegressor(
                        n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42,
                    ),
                   
                }

                best_rmse  = np.inf
                best_name  = "OLS"
                model_scores: Dict[str, float] = {}
                tscv = TimeSeriesSplit(n_splits=5)

                for name, base_mdl in candidate_models.items():
                    try:
                        cv_scores: List[float] = []
                        for train_idx, val_idx in tscv.split(X):
                            fold_mdl = _clone(base_mdl)
                            X_tr = X_df.iloc[train_idx] if name in _TREE_MODELS else X[train_idx]
                            X_vl = X_df.iloc[val_idx]   if name in _TREE_MODELS else X[val_idx]
                            y_tr, y_vl = y[train_idx], y[val_idx]

                            if sample_weights is not None and _supports_sample_weight(name):
                                fold_mdl.fit(X_tr, y_tr, sample_weight=sample_weights[train_idx])
                            else:
                                fold_mdl.fit(X_tr, y_tr)

                            y_pred = fold_mdl.predict(X_vl)
                            cv_scores.append(np.sqrt(mean_squared_error(y_vl, y_pred)))

                        avg_rmse = float(np.mean(cv_scores))
                        model_scores[name] = round(avg_rmse, 5)

                        if avg_rmse < best_rmse:
                            best_rmse = avg_rmse
                            best_name = name

                    except Exception as exc:
                        log.debug("CV failed for segment=%s model=%s: %s", seg, name, exc)

                # Statsmodels OLS for statistical inference
                try:
                    sm_mod = smf.ols(
                        "log_volume ~ log_price + log_discount", data=sub
                    ).fit()
                    elasticity           = float(sm_mod.params.get("log_price", -1.2))
                    disc_lift            = float(sm_mod.params.get("log_discount", 0.1))
                    intercept            = float(sm_mod.params.get("Intercept", 0.0))
                    r_squared            = float(sm_mod.rsquared)
                    p_elasticity         = float(sm_mod.pvalues.get("log_price", 1.0))
                    ci                   = sm_mod.conf_int()
                    ci_lo                = float(ci.loc["log_price", 0])
                    ci_hi                = float(ci.loc["log_price", 1])
                    elasticity_conf      = min(
                        1.0, (len(sub) / 500) * 0.5 + r_squared * 0.5
                    )
                except Exception as exc:
                    log.warning("statsmodels OLS failed for segment=%s: %s", seg, exc)
                    elasticity = -1.2
                    disc_lift  = 0.15
                    intercept  = 0.0
                    r_squared  = 0.0
                    p_elasticity = 1.0
                    ci_lo = ci_hi = elasticity
                    elasticity_conf = 0.3

                results[seg] = {
                    "elasticity":      round(elasticity, 4),
                    "elasticity_ci_lo": round(ci_lo, 4),
                    "elasticity_ci_hi": round(ci_hi, 4),
                    "p_value":         round(p_elasticity, 4),
                    "disc_lift":       round(disc_lift, 4),
                    "intercept":       round(intercept, 4),
                    "r_squared":       round(r_squared, 4),
                    "n_obs":           len(sub),
                    "mean_price":      round(float(sub["actual_price"].mean()), 2),
                    "mean_volume":     round(float(sub["volume"].mean()), 2),
                    "mean_margin":     round(float(sub["margin_pct"].mean()), 2),
                    "best_model":      best_name,
                    "cv_rmse":         round(best_rmse, 5),
                    "all_model_rmse":  model_scores,
                    "confidence":      round(elasticity_conf, 3),
                }

            except Exception as exc:
                log.error("Elasticity fitting failed for segment=%s: %s", seg, exc)

        return results

    # ------------------------------------------------------------------ #
    #  Win-rate model                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def fit_win_probability_model(
        df: pd.DataFrame,
        use_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit a binary classifier to predict P(deal_won).

        Returns a dict with keys: model, feature_cols, auc, model_name, …
        Returns an empty dict when there is insufficient data.
        """
        sub = df.copy()
        sub["price_ratio"]       = sub["actual_price"] / sub["base_price"]
        sub["margin_norm"]       = sub["margin_pct"] / 100.0
        sc_max = float(sub["sales_cycle_days"].max()) if sub["sales_cycle_days"].max() > 0 else 90.0
        sub["sales_cycle_norm"]  = sub["sales_cycle_days"] / sc_max
        sub["complexity_norm"]   = sub["complexity"] / 5.0
        sub["price_vs_competitor"] = (
            sub["actual_price"] / sub["competitor_price"]
            if "competitor_price" in sub.columns
            else 1.0
        )

        feature_cols = [
            "price_ratio", "discount_pct", "margin_norm",
            "sales_cycle_norm", "complexity_norm", "price_vs_competitor",
        ]
        sub = sub.dropna(subset=feature_cols + ["deal_won"])

        if len(sub) < ModelComparator.MIN_WIN_RATE_OBS:
            log.warning("Win-rate model: insufficient data (%d rows)", len(sub))
            return {}

        X    = sub[feature_cols].values
        y    = sub["deal_won"].values
        X_df = sub[feature_cols].copy()

        sample_weights = (
            sub["confidence_score"].values
            if use_confidence and "confidence_score" in sub.columns
            else None
        )

        # Guard against degenerate label distribution
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos = (n_neg / n_pos) if n_pos > 0 else 1.0

        candidates: Dict[str, Any] = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1_000, C=1.0, random_state=42, class_weight="balanced"
                )),
            ]),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42,
                n_jobs=-1, class_weight="balanced",
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, scale_pos_weight=scale_pos,
            ),
           
        }

        cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_auc = 0.0
        best_name = "LogisticRegression"
        best_mdl  = candidates["LogisticRegression"]
        auc_scores: Dict[str, float] = {}

        for name, base_mdl in candidates.items():
            try:
                X_use = X_df if name in _TREE_MODELS else X
                cv_scores: List[float] = []

                for train_idx, val_idx in cv.split(X, y):
                    fold_mdl = _clone(base_mdl)
                    X_tr = X_df.iloc[train_idx] if name in _TREE_MODELS else X[train_idx]
                    X_vl = X_df.iloc[val_idx]   if name in _TREE_MODELS else X[val_idx]
                    y_tr, y_vl = y[train_idx], y[val_idx]

                    if sample_weights is not None and _supports_sample_weight(name):
                        fold_mdl.fit(X_tr, y_tr, sample_weight=sample_weights[train_idx])
                    else:
                        fold_mdl.fit(X_tr, y_tr)

                    y_prob = fold_mdl.predict_proba(X_vl)[:, 1]
                    # Skip fold if only one class present in validation set
                    if len(np.unique(y_vl)) < 2:
                        continue
                    cv_scores.append(roc_auc_score(y_vl, y_prob))

                if not cv_scores:
                    continue
                avg_auc = float(np.mean(cv_scores))
                auc_scores[name] = round(avg_auc, 4)

                if avg_auc > best_auc:
                    best_auc  = avg_auc
                    best_name = name
                    best_mdl  = base_mdl

            except Exception as exc:
                log.debug("Win-rate CV failed for model=%s: %s", name, exc)

        # Final fit on full dataset
        final_mdl = _clone(best_mdl)
        X_best = X_df if best_name in _TREE_MODELS else X
        if sample_weights is not None and _supports_sample_weight(best_name):
            final_mdl.fit(X_best, y, sample_weight=sample_weights)
        else:
            final_mdl.fit(X_best, y)

        # Feature importance
        # `final_mdl` may be a plain estimator (e.g. GradientBoostingClassifier)
        # or a Pipeline.  Only Pipelines have `named_steps`; unwrap safely.
        fi: Dict[str, float] = {}
        inner = (
            final_mdl.named_steps.get("clf", final_mdl)
            if isinstance(final_mdl, Pipeline)
            else final_mdl
        )
        if hasattr(inner, "feature_importances_"):
            fi = dict(zip(feature_cols, inner.feature_importances_.round(4)))
        elif hasattr(inner, "coef_"):
            coef = inner.coef_
            # coef_ is 2-D for multi-class, 1-D for binary
            coef_arr = coef[0] if coef.ndim == 2 else coef
            fi = dict(zip(feature_cols, np.abs(coef_arr).round(4)))

        log.info("Win-rate model: %s  AUC=%.3f  n=%d", best_name, best_auc, len(sub))
        return {
            "model":             final_mdl,
            "feature_cols":      feature_cols,
            "sales_cycle_max":   sc_max,
            "auc":               round(best_auc, 4),
            "all_auc":           auc_scores,
            "model_name":        best_name,
            "feature_importance": fi,
            "n_obs":             len(sub),
            "confidence":        round(best_auc, 3),
        }

    @staticmethod
    def predict_win_probability(
        model_bundle: Dict[str, Any],
        price_ratio: float,
        discount_pct: float,
        margin_pct: float = 35.0,
        sales_cycle_days: float = 45.0,
        complexity: float = 3.0,
        price_vs_competitor: float = 1.0,
    ) -> float:
        """Return P(deal_won) ∈ [0.05, 0.95] given feature values."""
        if not model_bundle or "model" not in model_bundle:
            base_prob = max(0.05, min(0.95, 0.62 - 0.25 * max(0.0, price_ratio - 1.0)))
            if price_vs_competitor < 1.0:
                base_prob = min(0.95, base_prob * 1.2)
            return base_prob

        cols    = model_bundle["feature_cols"]
        sc_max  = max(1.0, model_bundle.get("sales_cycle_max", 90.0))
        row_val = {
            "price_ratio":        price_ratio,
            "discount_pct":       discount_pct,
            "margin_norm":        margin_pct / 100.0,
            "sales_cycle_norm":   sales_cycle_days / sc_max,
            "complexity_norm":    complexity / 5.0,
            "price_vs_competitor": price_vs_competitor,
        }
        X   = np.array([[row_val.get(c, 1.0) for c in cols]])
        mdl = model_bundle["model"]

        try:
            if model_bundle.get("model_name", "") in _TREE_MODELS:
                X = pd.DataFrame(X, columns=cols)
            return float(mdl.predict_proba(X)[0, 1])
        except Exception:
            return 0.5


# ============================================================================
# PRICING OPTIMIZER
# ============================================================================

class PricingOptimizer:
    """Computes the optimal price given elasticity, cost, and win-rate model."""

    @staticmethod
    def calculate(
        base_price: float,
        cost: float,
        elasticity: float,
        target: str = "revenue",
        constraints: Optional[Dict[str, float]] = None,
        win_model: Optional[Dict[str, Any]] = None,
        discount_pct_at_base: float = 0.0,
        complexity: float = 3.0,
        competitor_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        target : "revenue" | "profit" | "margin"
        """
        lower = base_price * 0.70
        upper = base_price * 2.00

        if constraints:
            lower = max(lower, constraints.get("min_price", lower))
            upper = min(upper, constraints.get("max_price", upper))

        if lower >= upper:
            log.warning(
                "PricingOptimizer: lower (%.2f) >= upper (%.2f) — nudging upper",
                lower, upper,
            )
            upper = lower + max(1.0, lower * 0.01)

        avg_comp_price: Optional[float] = (
            float(np.mean(list(competitor_prices.values())))
            if competitor_prices
            else None
        )

        def demand(p: float) -> float:
            return max(0.01, (p / base_price) ** elasticity)

        def win_prob(p: float) -> float:
            pr   = p / base_price
            disc = max(0.0, (base_price - p) / base_price * 100.0)
            marg = (p - cost) / p * 100.0 if p > 0.0 else 0.0
            pvc  = (p / avg_comp_price) if avg_comp_price else 1.0

            if win_model:
                return ModelComparator.predict_win_probability(
                    win_model, price_ratio=pr, discount_pct=disc,
                    margin_pct=marg, complexity=complexity,
                    price_vs_competitor=pvc,
                )

            base_prob = max(0.05, min(0.95, 0.62 - 0.25 * max(0.0, pr - 1.0)))
            if avg_comp_price:
                if p < avg_comp_price:
                    base_prob = min(0.95, base_prob * 1.2)
                elif p > avg_comp_price * 1.2:
                    base_prob = max(0.05, base_prob * 0.8)
            return base_prob

        def neg_obj(p: float) -> float:
            d  = demand(p)
            wp = win_prob(p)
            if target == "revenue":
                return -(p * d * wp)
            if target == "profit":
                return -((p - cost) * d * wp)
            # margin
            rev = p * d * wp
            if rev <= 0.0 or p <= 0.0:
                return 0.0
            return -((rev - cost * d * wp) / rev)

        try:
            res   = minimize_scalar(neg_obj, bounds=(lower, upper), method="bounded")
            p_opt = float(np.clip(res.x, lower, upper))
        except Exception:
            p_opt = base_price

        d_opt  = demand(p_opt)
        wp_opt = win_prob(p_opt)
        rev    = p_opt * d_opt
        profit = (p_opt - cost) * d_opt

        # Honour minimum win-rate constraint by scanning price reductions
        min_wr = constraints.get("min_win_rate", 0.0) if constraints else 0.0
        if wp_opt < min_wr:
            for pct_cut in np.linspace(0.01, 0.40, 40):
                p_trial = base_price * (1.0 - pct_cut)
                if p_trial < lower:
                    break
                if win_prob(p_trial) >= min_wr:
                    p_opt  = p_trial
                    d_opt  = demand(p_opt)
                    wp_opt = win_prob(p_opt)
                    rev    = p_opt * d_opt
                    profit = (p_opt - cost) * d_opt
                    break

        vs_comp = (
            round(p_opt / avg_comp_price, 3) if avg_comp_price else 1.0
        )
        margin_pct = (
            round((p_opt - cost) / p_opt * 100.0, 2) if p_opt > 0.0 else 0.0
        )

        return {
            "optimal_price":    round(p_opt, 2),
            "demand_index":     round(d_opt, 4),
            "win_probability":  round(wp_opt, 4),
            "expected_revenue": round(rev, 2),
            "expected_profit":  round(profit, 2),
            "margin_pct":       margin_pct,
            "vs_base_pct":      round((p_opt / base_price - 1.0) * 100.0, 2),
            "target":           target,
            "vs_competitor":    vs_comp,
        }

    @staticmethod
    def optimize_portfolio(
        products: List[str],
        elasticities: Dict[str, Any],
        target: str = "profit",
        constraints: Optional[Dict[str, Any]] = None,
        competitor_prices: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Jointly optimise prices for a list of products."""
        rows = []
        for prod in products:
            info  = PRODUCT_CATALOG.get(prod)
            if not info:
                log.warning("optimize_portfolio: unknown product '%s' — skipping", prod)
                continue
            seg   = info["segment"]
            elast = elasticities.get(seg, {}).get("elasticity", -1.2)
            result = PricingOptimizer.calculate(
                base_price=info["base_price"],
                cost=info["cost"],
                elasticity=elast,
                target=target,
                constraints=constraints,
                competitor_prices=competitor_prices,
            )
            rows.append({"product": prod, "segment": seg,
                         "base_price": info["base_price"], **result})
        return pd.DataFrame(rows)


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """
    Quasi-random (Sobol) Monte-Carlo simulation with confidence intervals.

    Sobol dimension mapping (d=3)
    -----------------------------
    0 : price             uniform in [p_lo, p_hi]
    1 : elasticity noise  ±15 % of the point estimate
    2 : demand noise      ±8 %
    """

    @staticmethod
    def simulate_revenue_scenarios(
        product: str,
        price_range: Tuple[float, float],
        elasticity: float,
        cost: float,
        n_sim: int = 2_048,
        confidence: float = 0.95,
        competitor_prices: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Returns (simulation_df, risk_summary_dict).

        The simulation_df has columns: price, demand, revenue, profit, margin.
        risk_summary_dict includes percentile / VaR statistics at the requested
        confidence level.
        """
        n_pow2 = int(2 ** np.floor(np.log2(max(n_sim, 64))))
        sobol   = Sobol(d=3, scramble=True)
        samples = sobol.random(n_pow2)

        p_lo, p_hi = price_range
        prices  = p_lo + (p_hi - p_lo) * samples[:, 0]
        e_noise = elasticity * (1.0 + 0.15 * (samples[:, 1] * 2.0 - 1.0))
        d_noise = 1.0 + 0.08 * (samples[:, 2] * 2.0 - 1.0)

        base   = PRODUCT_CATALOG[product]["base_price"]
        demand = np.maximum(0.01, (prices / base) ** e_noise) * d_noise

        if competitor_prices:
            avg_comp = float(np.mean(list(competitor_prices.values())))
            win_factor = np.where(
                prices < avg_comp,
                1.0 + 0.05 * (avg_comp - prices) / avg_comp,
                np.where(prices > avg_comp * 1.2, 0.95, 1.0),
            )
            demand = demand * win_factor

        revenue = prices * demand
        profit  = (prices - cost) * demand
        margin  = np.where(revenue > 0, (revenue - cost * demand) / revenue * 100.0, 0.0)

        sim_df = pd.DataFrame({
            "price":   np.round(prices, 2),
            "demand":  np.round(demand, 4),
            "revenue": np.round(revenue, 2),
            "profit":  np.round(profit, 2),
            "margin":  np.round(margin, 2),
        })

        rev_arr = sim_df["revenue"].values
        prf_arr = sim_df["profit"].values
        z       = norm.ppf((1.0 + confidence) / 2.0)

        risk_summary = {
            "revenue_mean":     round(float(rev_arr.mean()), 2),
            "revenue_std":      round(float(rev_arr.std()), 2),
            "revenue_p10":      round(float(np.percentile(rev_arr, 10)), 2),
            "revenue_p50":      round(float(np.percentile(rev_arr, 50)), 2),
            "revenue_p90":      round(float(np.percentile(rev_arr, 90)), 2),
            "revenue_var5":     round(float(np.percentile(rev_arr, 5)), 2),
            "revenue_ci_lower": round(float(rev_arr.mean() - z * rev_arr.std()), 2),
            "revenue_ci_upper": round(float(rev_arr.mean() + z * rev_arr.std()), 2),
            "profit_mean":      round(float(prf_arr.mean()), 2),
            "profit_p10":       round(float(np.percentile(prf_arr, 10)), 2),
            "profit_p50":       round(float(np.percentile(prf_arr, 50)), 2),
            "profit_p90":       round(float(np.percentile(prf_arr, 90)), 2),
            "profit_var5":      round(float(np.percentile(prf_arr, 5)), 2),
            "profit_ci_lower":  round(float(prf_arr.mean() - z * prf_arr.std()), 2),
            "profit_ci_upper":  round(float(prf_arr.mean() + z * prf_arr.std()), 2),
            "n_sim":            n_pow2,
            "confidence":       confidence,
        }

        return sim_df, risk_summary

    @staticmethod
    def apply_pricing_scenario(
        df: pd.DataFrame,
        price_changes: Dict[str, float],
        elasticities: Optional[Dict[str, Any]] = None,
        competitor_prices: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply a set of price changes to the dataset and return
        (modified_df, summary_df).
        """
        df2 = df.copy()
        summary_rows: List[Dict[str, Any]] = []

        avg_comp: Optional[float] = (
            float(np.mean(list(competitor_prices.values())))
            if competitor_prices
            else None
        )

        for prod, new_price in price_changes.items():
            info = PRODUCT_CATALOG.get(prod)
            if info is None:
                log.warning("apply_pricing_scenario: unknown product '%s'", prod)
                continue

            old_bp = info["base_price"]
            cost_u = info["cost"]
            new_price = float(np.clip(
                new_price,
                info.get("min_price", old_bp * 0.5),
                info.get("max_price", old_bp * 2.0),
            ))

            seg        = info["segment"]
            elast_raw  = (
                float(elasticities[seg].get("elasticity", -1.2))
                if elasticities and seg in elasticities
                else -1.2
            )

            mask        = df2["product"] == prod
            before_rev  = df2.loc[mask, "revenue"].sum()
            before_prof = (df2.loc[mask, "revenue"] - df2.loc[mask, "cost"]).sum()

            ratio               = new_price / old_bp
            volume_multiplier   = ratio ** elast_raw

            if avg_comp:
                comp_ratio = new_price / avg_comp
                if comp_ratio < 1.0:
                    volume_multiplier *= 1.0 + 0.1 * (1.0 - comp_ratio)
                elif comp_ratio > 1.2:
                    volume_multiplier *= max(0.7, 1.0 - 0.2 * (comp_ratio - 1.0))

            df2.loc[mask, "actual_price"] = new_price
            df2.loc[mask, "discount_pct"] = max(0.0, (old_bp - new_price) / old_bp * 100.0)
            df2.loc[mask, "volume"]        = np.maximum(
                1,
                np.round(df2.loc[mask, "volume"].values * volume_multiplier),
            ).astype(int)
            df2.loc[mask, "revenue"]   = df2.loc[mask, "actual_price"] * df2.loc[mask, "volume"]
            df2.loc[mask, "cost"]      = cost_u * df2.loc[mask, "volume"]
            df2.loc[mask, "margin_pct"] = (
                (df2.loc[mask, "revenue"] - df2.loc[mask, "cost"])
                / df2.loc[mask, "revenue"].replace(0.0, np.nan) * 100.0
            ).fillna(0.0).round(2)

            after_rev  = df2.loc[mask, "revenue"].sum()
            after_prof = (df2.loc[mask, "revenue"] - df2.loc[mask, "cost"]).sum()

            summary_rows.append({
                "product":          prod,
                "old_price":        round(old_bp, 2),
                "new_price":        round(new_price, 2),
                "price_change_pct": round((new_price / old_bp - 1.0) * 100.0, 2),
                "revenue_before":   round(before_rev, 2),
                "revenue_after":    round(after_rev, 2),
                "revenue_delta":    round(after_rev - before_rev, 2),
                "profit_before":    round(before_prof, 2),
                "profit_after":     round(after_prof, 2),
                "profit_delta":     round(after_prof - before_prof, 2),
                "volume_change_pct": round((volume_multiplier - 1.0) * 100.0, 2),
            })

        return df2, pd.DataFrame(summary_rows)


# ============================================================================
# COMPETITIVE ANALYZER
# ============================================================================

class CompetitiveAnalyzer:
    """Compares STRAIVE pricing vs. competitors with optional real scraped data."""

    STRAIVE_QUALITY: float = 8.2  # internal quality benchmark (out of 10)

    @staticmethod
    def get_price_score(
        price: float,
        base_price: float,
        competitor_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        straive_rel = price / base_price
        comp_rows: List[Dict[str, Any]] = []

        if competitor_data:
            for item in competitor_data:
                comp_name  = item.get("competitor_name", "Unknown")
                comp_price = float(item.get("price", base_price))
                comp_info  = COMPETITORS.get(comp_name, {})
                gap_pct    = (price - comp_price) / comp_price * 100.0 if comp_price > 0.0 else 0.0
                quality    = comp_info.get("quality_score", 7.0)
                val_score  = quality / (comp_price / base_price) if comp_price > 0.0 else 0.0
                comp_rows.append({
                    "Competitor":   comp_name,
                    "Comp Price":   round(comp_price, 2),
                    "STRAIVE Price": round(price, 2),
                    "Gap %":        round(gap_pct, 1),
                    "Quality Score": quality,
                    "Comp Relative": round(comp_price / base_price, 3),
                    "Value Score":  round(val_score, 2),
                    "Win Rate vs":  round(comp_info.get("win_rate_vs", 0.5) * 100.0, 1),
                    "Source":       item.get("source", "scraped"),
                    "Confidence":   item.get("confidence_score", 0.7),
                })
        else:
            for name, info in COMPETITORS.items():
                comp_price = base_price * info["relative_price"]
                gap_pct    = (price - comp_price) / comp_price * 100.0 if comp_price > 0.0 else 0.0
                val_score  = info["quality_score"] / info["relative_price"]
                comp_rows.append({
                    "Competitor":   name,
                    "Comp Price":   round(comp_price, 2),
                    "STRAIVE Price": round(price, 2),
                    "Gap %":        round(gap_pct, 1),
                    "Quality Score": info["quality_score"],
                    "Comp Relative": info["relative_price"],
                    "Value Score":  round(val_score, 2),
                    "Win Rate vs":  round(info.get("win_rate_vs", 0.5) * 100.0, 1),
                    "Source":       "config",
                    "Confidence":   0.8,
                })

        comp_df = pd.DataFrame(comp_rows)
        if comp_df.empty:
            return {
                "df": comp_df, "pct_rank": 0.0, "straive_relative": straive_rel,
                "straive_quality": CompetitiveAnalyzer.STRAIVE_QUALITY,
                "avg_comp_price": 0.0, "min_comp_price": 0.0, "max_comp_price": 0.0,
                "market_position": {}, "n_competitors": 0,
            }

        pct_rank = int((comp_df["Gap %"] < 0.0).sum()) / len(comp_df) * 100.0
        comp_df["Quality Gap"] = CompetitiveAnalyzer.STRAIVE_QUALITY - comp_df["Quality Score"]

        conf_sum = comp_df["Confidence"].sum()
        weighted_price = (
            float((comp_df["Confidence"] * comp_df["Comp Price"]).sum() / conf_sum)
            if conf_sum > 0.0 else float(comp_df["Comp Price"].mean())
        )

        market_position = {
            "price_advantage":   round(float((comp_df["Gap %"] < 0).mean() * 100.0), 1),
            "value_advantage":   round(
                float((comp_df["Value Score"] > comp_df["Value Score"].median()).mean() * 100.0), 1
            ),
            "avg_price_gap":     round(float(comp_df["Gap %"].mean()), 1),
            "weighted_confidence": round(weighted_price, 2),
        }

        return {
            "df":               comp_df,
            "pct_rank":         round(pct_rank, 1),
            "straive_relative": round(straive_rel, 4),
            "straive_quality":  CompetitiveAnalyzer.STRAIVE_QUALITY,
            "avg_comp_price":   round(float(comp_df["Comp Price"].mean()), 2),
            "min_comp_price":   round(float(comp_df["Comp Price"].min()), 2),
            "max_comp_price":   round(float(comp_df["Comp Price"].max()), 2),
            "market_position":  market_position,
            "n_competitors":    len(comp_df),
        }

    @staticmethod
    def track_price_trends(
        competitor_prices: List[Dict],
        days: int = 90,
    ) -> pd.DataFrame:
        """Track competitor price trends over time from a list of scraped records."""
        if not competitor_prices:
            return pd.DataFrame()

        df = pd.DataFrame(competitor_prices)
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=False)
        df = df.dropna(subset=["scraped_at"])

        cutoff = datetime.now() - timedelta(days=days)
        df     = df[df["scraped_at"] >= cutoff]

        trends = []
        for competitor in df["competitor_name"].unique():
            comp_df = df[df["competitor_name"] == competitor].sort_values("scraped_at")
            if len(comp_df) < 2:
                continue

            prices_arr  = comp_df["price"].values.astype(float)
            first_price = float(prices_arr[0])
            last_price  = float(prices_arr[-1])
            change_pct  = (
                (last_price - first_price) / first_price * 100.0
                if first_price != 0.0 else 0.0
            )

            trend = "stable"
            if len(comp_df) >= 3:
                x = np.arange(len(comp_df), dtype=float)
                price_range = prices_arr.max() - prices_arr.min()
                if price_range > 0:
                    slope = float(np.polyfit(x, prices_arr, 1)[0])
                    trend = "up" if slope > 0 else "down"

            mean_price = float(prices_arr.mean())
            volatility = (
                round(float(prices_arr.std() / mean_price * 100.0), 1)
                if mean_price > 0.0 else 0.0
            )

            trends.append({
                "competitor":    competitor,
                "first_price":   round(first_price, 2),
                "last_price":    round(last_price, 2),
                "change_pct":    round(change_pct, 1),
                "trend":         trend,
                "n_observations": len(comp_df),
                "volatility":    volatility,
            })

        return pd.DataFrame(trends)


# ============================================================================
# REVENUE FORECASTER
# ============================================================================

class RevenueForecaster:
    """Produces revenue forecasts using ensemble and single-method approaches."""

    @staticmethod
    def forecast(
        df: pd.DataFrame,
        horizon_months: int = 12,
        ci: float = 0.90,
        method: str = "ensemble",
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        method : "linear" | "exp_smoothing" | "arima" | "ensemble"
        """
        df2 = df.copy()
        df2["month"] = pd.to_datetime(df2["date"]).dt.to_period("M").dt.to_timestamp()

        monthly = (
            df2.groupby(["month", "segment"])["revenue"]
            .sum()
            .reset_index()
            .sort_values("month")
        )

        forecast_rows: List[Dict[str, Any]] = []
        last_month = monthly["month"].max()
        z          = norm.ppf((1.0 + ci) / 2.0)

        for seg in monthly["segment"].unique():
            sub = monthly[monthly["segment"] == seg].copy().reset_index(drop=True)
            if len(sub) < 4:
                continue

            try:
                sub["t"] = np.arange(len(sub))
                X_t = sub["t"].values.reshape(-1, 1)
                y_r = sub["revenue"].values
                lr  = LinearRegression().fit(X_t, y_r)
                resid_std = float(np.std(y_r - lr.predict(X_t)))

                if method == "ensemble":
                    # Fit all sub-models once
                    try:
                        exp_model = ExponentialSmoothing(
                            sub["revenue"],
                            seasonal_periods=12 if len(sub) >= 24 else None,
                            trend="add",
                            seasonal="add" if len(sub) >= 24 else None,
                        ).fit()
                    except Exception:
                        exp_model = ExponentialSmoothing(
                            sub["revenue"], trend="add"
                        ).fit()

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            arima_model: Optional[Any] = ARIMA(
                                sub["revenue"], order=(1, 1, 1)
                            ).fit()
                    except Exception:
                        arima_model = None

                    # Pre-compute all horizon forecasts (O(1) per horizon)
                    exp_fc    = exp_model.forecast(horizon_months).values
                    arima_fc  = (
                        arima_model.forecast(horizon_months).values
                        if arima_model is not None
                        else None
                    )

                    for h in range(1, horizon_months + 1):
                        future_month  = last_month + pd.DateOffset(months=h)
                        lr_fc         = float(lr.predict([[len(sub) + h - 1]])[0])
                        e_fc          = float(exp_fc[h - 1])
                        a_fc          = float(arima_fc[h - 1]) if arima_fc is not None else lr_fc

                        ensemble_fc   = 0.30 * lr_fc + 0.40 * e_fc + 0.30 * a_fc
                        season_w      = FORECAST_CONFIG.get("seasonality_weight", 0.6)
                        season_factor = MONTHLY_SEASONALITY.get(future_month.month, 1.0)
                        final_fc      = ensemble_fc * (
                            season_w * season_factor + (1.0 - season_w)
                        )
                        band = z * resid_std * np.sqrt(h)

                        forecast_rows.append({
                            "month":    future_month,
                            "segment":  seg,
                            "forecast": round(max(0.0, final_fc), 2),
                            "ci_lo":    round(max(0.0, final_fc - band), 2),
                            "ci_hi":    round(final_fc + band, 2),
                            "type":     "forecast",
                            "method":   "ensemble",
                        })

                else:
                    # Single linear method — pre-compute & vectorise
                    h_arr         = np.arange(1, horizon_months + 1)
                    t_arr         = (len(sub) + h_arr - 1).reshape(-1, 1)
                    lr_preds      = lr.predict(t_arr)
                    season_w      = FORECAST_CONFIG.get("seasonality_weight", 0.6)

                    for h, lr_fc in zip(h_arr, lr_preds):
                        future_month  = last_month + pd.DateOffset(months=int(h))
                        season_factor = MONTHLY_SEASONALITY.get(future_month.month, 1.0)
                        final_fc      = float(lr_fc) * (
                            season_w * season_factor + (1.0 - season_w)
                        )
                        band = z * resid_std * np.sqrt(float(h))
                        forecast_rows.append({
                            "month":    future_month,
                            "segment":  seg,
                            "forecast": round(max(0.0, final_fc), 2),
                            "ci_lo":    round(max(0.0, final_fc - band), 2),
                            "ci_hi":    round(final_fc + band, 2),
                            "type":     "forecast",
                            "method":   method,
                        })

            except Exception as exc:
                log.warning("Forecast failed for segment=%s: %s", seg, exc)
                continue

        # Combine with historical actuals
        hist = monthly.rename(columns={"revenue": "forecast"}).copy()
        hist["ci_lo"]  = hist["forecast"]
        hist["ci_hi"]  = hist["forecast"]
        hist["type"]   = "historical"
        hist["method"] = "actual"

        all_rows = (
            pd.concat([hist, pd.DataFrame(forecast_rows)], ignore_index=True)
            if forecast_rows
            else hist.copy()
        )
        return all_rows.sort_values(["segment", "month"]).reset_index(drop=True)


# ============================================================================
# DEAL SCORER
# ============================================================================

class DealScorer:
    """
    Produces a 0–100 composite deal health score.

    All scoring weights are defined once as a class constant so changes
    propagate automatically to both the computation and the returned metadata.
    """

    RATING_THRESHOLDS: List[Tuple[float, str]] = [
        (85.0, "Excellent"),
        (70.0, "Good"),
        (55.0, "Fair"),
        (40.0, "Caution"),
        (25.0, "At Risk"),
        (0.0,  "Critical"),
    ]

    SCORE_WEIGHTS: Dict[str, float] = {
        "margin":     0.25,
        "win":        0.20,
        "price":      0.15,
        "discount":   0.10,
        "loyalty":    0.10,
        "term":       0.05,
        "competitor": 0.10,
        "quality":    0.05,
    }

    STRAIVE_QUALITY: float = 8.2

    @classmethod
    def get_rating(cls, composite: float) -> str:
        """Map a composite score to a human-readable rating label."""
        for threshold, label in cls.RATING_THRESHOLDS:
            if composite >= threshold:
                return label
        return "Critical"

    @classmethod
    def score(
        cls,
        base_price: float,
        proposed_price: float,
        discount_pct: float,
        customer_segment: str,
        win_probability: float,
        margin_pct: float,
        contract_months: int = 12,
        competitor_prices: Optional[Dict[str, float]] = None,
        competitor_quality: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with keys: score, rating, recommendation,
        component_scores, weights, price_ratio, competitive_position.
        """
        if proposed_price <= 0.0:
            log.warning("DealScorer.score: proposed_price=%.2f <= 0 — defaulting margin to 0",
                        proposed_price)
            margin_pct = 0.0

        price_ratio = proposed_price / base_price if base_price > 0.0 else 1.0
        loyalty     = CUSTOMER_SEGMENTS.get(customer_segment, {}).get("loyalty", 0.75)

        price_score   = max(0.0, min(100.0, (1.0 - max(0.0, 1.0 - price_ratio) * 2.0) * 100.0))
        disc_score    = max(0.0, 100.0 - discount_pct * 2.0)
        margin_score  = max(0.0, min(100.0, margin_pct * 2.0))
        win_score     = win_probability * 100.0
        loyalty_score = loyalty * 100.0
        term_score    = min(100.0, contract_months / 36.0 * 100.0)

        # Competitor score
        comp_score = 50.0
        if competitor_prices:
            avg_comp = float(np.mean(list(competitor_prices.values())))
            if avg_comp > 0.0:
                if proposed_price < avg_comp:
                    comp_score = 70.0 + 30.0 * (1.0 - proposed_price / avg_comp)
                elif proposed_price > avg_comp * 1.2:
                    comp_score = max(30.0, 50.0 - 20.0 * (proposed_price / avg_comp - 1.2))
                else:
                    comp_score = 60.0

        # Quality score
        quality_score = 50.0
        if competitor_quality:
            avg_q = float(np.mean(list(competitor_quality.values())))
            quality_score = min(100.0, 50.0 + (cls.STRAIVE_QUALITY - avg_q) * 10.0)

        component_scores = {
            "margin":     margin_score,
            "win":        win_score,
            "price":      price_score,
            "discount":   disc_score,
            "loyalty":    loyalty_score,
            "term":       term_score,
            "competitor": comp_score,
            "quality":    quality_score,
        }

        composite = sum(
            cls.SCORE_WEIGHTS[k] * component_scores[k]
            for k in cls.SCORE_WEIGHTS
        )
        rating = cls.get_rating(composite)

        if composite >= 70.0:
            recommendation = "✅ Strong deal – proceed with confidence"
        elif composite >= 55.0:
            recommendation = "⚠️ Moderate risk – review terms and discount"
        elif composite >= 40.0:
            recommendation = "⚠️ Caution – consider renegotiation or escalation"
        else:
            recommendation = "🔴 High risk – decline or major restructuring needed"

        return {
            "score":                round(composite, 1),
            "rating":               rating,
            "recommendation":       recommendation,
            "component_scores":     {k: round(v, 1) for k, v in component_scores.items()},
            "weights":              cls.SCORE_WEIGHTS,
            "price_ratio":          round(price_ratio, 4),
            "competitive_position": (
                "advantage"  if comp_score > 60.0 else
                "neutral"    if comp_score > 40.0 else
                "disadvantage"
            ),
        }


# ============================================================================
# MARGIN WATERFALL BUILDER
# ============================================================================

class MarginWaterfallBuilder:
    """Builds waterfall data from a transaction DataFrame."""

    @staticmethod
    def build(
        df: pd.DataFrame,
        product_filter: Optional[str] = None,
        segment_filter: Optional[str] = None,
        include_confidence: bool = True,
    ) -> pd.DataFrame:
        sub = df.copy()
        if product_filter:
            sub = sub[sub["product"] == product_filter]
        if segment_filter:
            sub = sub[sub["segment"] == segment_filter]

        if sub.empty:
            return pd.DataFrame(columns=["label", "value", "measure", "confidence"])

        total_revenue = float(sub["revenue"].sum())
        total_cost    = float(sub["cost"].sum())
        gross_profit  = total_revenue - total_cost

        avg_confidence = (
            float(sub["confidence_score"].mean())
            if include_confidence and "confidence_score" in sub.columns
            else 1.0
        )

        # Cap discount so net_revenue can never go negative
        raw_disc       = float(sub["discount_pct"].mean()) / 100.0 * total_revenue
        discount_amount = min(raw_disc, total_revenue)
        net_revenue    = total_revenue - discount_amount

        waterfall_rows = [
            {"label": "List Revenue", "value": round(total_revenue,  2), "measure": "absolute", "confidence": avg_confidence},
            {"label": "Discounts",    "value": round(-discount_amount, 2), "measure": "relative", "confidence": avg_confidence},
            {"label": "Net Revenue",  "value": round(net_revenue,    2), "measure": "total",    "confidence": avg_confidence},
        ]

        running = net_revenue
        for bucket, share in MARGIN_WATERFALL_BUCKETS.items():
            alloc    = total_cost * share
            running -= alloc
            waterfall_rows.append({
                "label":      bucket,
                "value":      -round(alloc, 2),
                "measure":    "relative",
                "confidence": avg_confidence,
            })

        waterfall_rows.append({
            "label":      "Net Margin",
            "value":      round(running, 2),
            "measure":    "total",
            "confidence": avg_confidence,
        })

        return pd.DataFrame(waterfall_rows)


# ============================================================================
# PORTFOLIO SCORER
# ============================================================================

class PortfolioScorer:
    """Ranks products using multi-criteria weighted score (config-driven)."""

    @staticmethod
    def score_products(
        df: pd.DataFrame,
        elasticities: Dict[str, Any],
        competitor_data: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for prod, info in PRODUCT_CATALOG.items():
            mask = df["product"] == prod
            sub  = df[mask]
            if sub.empty:
                continue

            seg      = info["segment"]
            e_dat    = elasticities.get(seg, {})
            elast    = abs(float(e_dat.get("elasticity", -1.2)))

            rev      = float(sub["revenue"].sum())
            cost_tot = float(sub["cost"].sum())
            margin   = (rev - cost_tot) / rev if rev > 0.0 else 0.0
            win_rate = float(sub["deal_won"].mean()) if "deal_won" in sub else 0.5

            # Revenue growth: recent half vs prior half
            if len(sub) >= 2:
                sub_sorted  = sub.sort_values("date")
                mid         = len(sub_sorted) // 2
                rev_prior   = float(sub_sorted.iloc[:mid]["revenue"].sum())
                rev_recent  = float(sub_sorted.iloc[mid:]["revenue"].sum())
                growth      = (rev_recent - rev_prior) / rev_prior if rev_prior > 0.0 else 0.0
            else:
                growth = 0.0

            # Competitive moat
            avg_cq = float(np.mean([c["quality_score"] for c in COMPETITORS.values()]))
            moat   = 1.0 - avg_cq / 10.0

            # Market position (sigmoid of price ratio vs. competitors)
            # Formula: σ(3 * (our_price / avg_comp - 1))  where σ = logistic
            market_position = 0.5
            if competitor_data:
                valid_prices = [
                    c.get("price") for c in competitor_data
                    if c.get("competitor_name") in COMPETITORS and c.get("price") is not None
                ]
                if valid_prices:
                    avg_comp        = float(np.mean(valid_prices))
                    pr              = info["base_price"] / avg_comp if avg_comp > 0.0 else 1.0
                    market_position = float(1.0 / (1.0 + np.exp(-3.0 * (pr - 1.0))))

            loyalty_vals = [
                CUSTOMER_SEGMENTS[c]["loyalty"]
                for c in sub["customer_type"].unique()
                if c in CUSTOMER_SEGMENTS
            ] if "customer_type" in sub.columns else []
            loyalty = float(np.mean(loyalty_vals)) if loyalty_vals else 0.75

            confidence = (
                float(sub["confidence_score"].mean())
                if "confidence_score" in sub.columns
                else 1.0
            )

            raw = {
                "margin_pct":          margin,
                "revenue_growth":      growth,
                "win_rate":            win_rate,
                "customer_loyalty":    loyalty,
                "competitive_moat":    moat,
                "market_position":     market_position,
                "price_competitiveness": 1.0 - market_position,
            }

            rows.append({
                "product":      prod,
                "segment":      seg,
                "revenue":      round(rev, 2),
                "margin_pct":   round(margin * 100.0, 2),
                "win_rate":     round(win_rate * 100.0, 2),
                "growth_pct":   round(growth * 100.0, 2),
                "loyalty":      round(loyalty * 100.0, 2),
                "elasticity":   round(-elast, 4),
                "market_position": round(market_position * 100.0, 2),
                "confidence":   round(confidence, 3),
                **{f"_raw_{k}": v for k, v in raw.items()},
            })

        scored_df = pd.DataFrame(rows)
        if scored_df.empty:
            return scored_df

        # Min-max normalise raw columns
        for key in SCORING_WEIGHTS:
            col = f"_raw_{key}"
            if col not in scored_df.columns:
                scored_df[f"_norm_{key}"] = 0.5
                continue
            mn    = scored_df[col].min()
            mx    = scored_df[col].max()
            denom = mx - mn
            scored_df[f"_norm_{key}"] = (
                (scored_df[col] - mn) / denom if denom > 1e-9 else 0.5
            )

        scored_df["score"] = round(
            sum(
                SCORING_WEIGHTS[k] * scored_df.get(
                    f"_norm_{k}", pd.Series(0.5, index=scored_df.index)
                )
                for k in SCORING_WEIGHTS
            ) * 100.0,
            4,
        )

        scored_df["adjusted_score"] = scored_df["score"] * scored_df["confidence"]

        # Drop internal helper columns
        drop_cols = [c for c in scored_df.columns if c.startswith("_")]
        return (
            scored_df.drop(columns=drop_cols)
            .sort_values("adjusted_score", ascending=False)
            .reset_index(drop=True)
        )


# ============================================================================
# MARKET INTELLIGENCE INTEGRATOR
# ============================================================================

class MarketIntelligenceIntegrator:
    """Integrates market intelligence data with pricing models."""

    @staticmethod
    def enhance_elasticity_with_market_data(
        elasticity_results: Dict[str, Any],
        market_data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Adjust per-segment elasticity estimates based on scraped market prices.

        Uses segment-specific base_price (first product in that segment) rather
        than the global first-product fallback used in v4.0.
        """
        enhanced = {k: dict(v) for k, v in elasticity_results.items()}

        # Build segment → base_price mapping from catalogue
        seg_base_price: Dict[str, float] = {}
        for info in PRODUCT_CATALOG.values():
            seg = info.get("segment")
            if seg and seg not in seg_base_price:
                seg_base_price[seg] = float(info["base_price"])

        # Group market data by segment
        market_by_seg: Dict[str, List[Dict]] = {}
        for item in market_data:
            seg = item.get("segment")
            if seg:
                market_by_seg.setdefault(seg, []).append(item)

        for seg, seg_data in enhanced.items():
            mkt_items = market_by_seg.get(seg)
            if not mkt_items:
                continue

            prices = [d.get("price") for d in mkt_items if d.get("price") is not None]
            if not prices:
                continue

            base_price   = seg_base_price.get(seg, 2_800.0)
            avg_mkt      = float(np.mean(prices))
            market_ratio = avg_mkt / base_price if base_price > 0.0 else 1.0

            old_elast = seg_data.get("elasticity", -1.2)
            if market_ratio < 0.9:
                # Competitors are cheaper → demand is more elastic
                seg_data["elasticity"] = old_elast * 1.1
            elif market_ratio > 1.1:
                # Competitors are more expensive → demand is less elastic
                seg_data["elasticity"] = old_elast * 0.9

            seg_data["market_adjusted"]    = True
            seg_data["market_price_ratio"] = round(market_ratio, 3)

        return enhanced

    @staticmethod
    def detect_pricing_opportunities(
        df: pd.DataFrame,
        competitor_data: List[Dict],
    ) -> List[Dict]:
        """Detect pricing opportunities based on competitor data."""
        if not competitor_data:
            return []

        comp_by_product: Dict[str, List[Dict]] = {}
        for item in competitor_data:
            svc = item.get("service_name")
            if svc:
                comp_by_product.setdefault(svc, []).append(item)

        opportunities: List[Dict] = []

        for prod, info in PRODUCT_CATALOG.items():
            mask = df["product"] == prod
            sub  = df[mask]
            if sub.empty:
                continue

            avg_price = float(sub["actual_price"].mean())
            margin    = float(sub["margin_pct"].mean())

            matching: List[Dict] = []
            for comp_prod, comp_items in comp_by_product.items():
                if any(
                    kw in comp_prod.lower()
                    for kw in info.get("keywords", [])
                ):
                    matching.extend(comp_items)

            if not matching:
                continue

            comp_prices = [c["price"] for c in matching if c.get("price") is not None]
            if not comp_prices:
                continue

            avg_comp  = float(np.mean(comp_prices))
            price_gap = (avg_price - avg_comp) / avg_comp * 100.0 if avg_comp > 0.0 else 0.0
            vol_total = float(sub["volume"].sum())

            if price_gap > 20.0 and margin > 40.0:
                opportunities.append({
                    "product":       prod,
                    "type":          "price_reduction_opportunity",
                    "description":   f"Price is {price_gap:.0f}% above competitors with high margin",
                    "current_price": round(avg_price, 2),
                    "market_price":  round(avg_comp, 2),
                    "potential_gain": round((avg_price - avg_comp) * vol_total, 2),
                    "confidence":    0.7,
                })
            elif price_gap < -15.0:
                opportunities.append({
                    "product":       prod,
                    "type":          "price_increase_opportunity",
                    "description":   f"Price is {abs(price_gap):.0f}% below market average",
                    "current_price": round(avg_price, 2),
                    "market_price":  round(avg_comp, 2),
                    "potential_gain": round((avg_comp - avg_price) * vol_total, 2),
                    "confidence":    0.6,
                })

        return sorted(opportunities, key=lambda x: x.get("potential_gain", 0.0), reverse=True)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "CompetitiveAnalyzer",
    "DataGenerator",
    "DealScorer",
    "MarginWaterfallBuilder",
    "MarketIntelligenceIntegrator",
    "ModelComparator",
    "PortfolioScorer",
    "PricingOptimizer",
    "RevenueForecaster",
    "SimulationEngine",
]
