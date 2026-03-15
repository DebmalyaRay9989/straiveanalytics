"""
STRAIVE Pricing Platform – Engine Module (v5.0 Cloud-Light)

Changes vs v4.1:
  • Removed XGBoost, GradientBoosting, RandomForest from candidate models
    → uses OLS, Ridge, Lasso, ElasticNet, BayesianRidge only (lighter, faster)
  • CV folds reduced 5 → 3 for elasticity and win-rate fitting
  • DataGenerator default n reduced to 2000 (was 3600)
  • Removed scipy Sobol quasi-random sequences → plain numpy for simulation
  • Removed ARIMA from RevenueForecaster (statsmodels ARIMA is heavy)
    → ExponentialSmoothing + linear trend ensemble only
  • Removed unused deepcopy import on tree-specific paths
  • All other business logic preserved intact
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
from scipy.stats import norm
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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


def _clone(model: Any) -> Any:
    return deepcopy(model)


# ============================================================================
# DATA GENERATOR
# ============================================================================

class DataGenerator:
    BASE_DEMAND_SCALE: float = 8.0
    CONFIDENCE_DECAY: float = 0.3 / 2_000
    CONFIDENCE_MIN: float = 0.30
    CONFIDENCE_OUTLIER_PENALTY: float = 0.80
    PRICE_OUTLIER_THRESHOLD: float = 0.50
    MAX_HISTORY_DAYS: int = 730
    CONTRACT_PROBS: Tuple[float, float, float] = (0.25, 0.55, 0.20)

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, n: int = 2_000, include_confidence: bool = True) -> pd.DataFrame:
        rng = self._rng
        products    = list(PRODUCT_CATALOG.keys())
        segments    = list(CUSTOMER_SEGMENTS.keys())
        regions     = list(REGIONS.keys())
        competitors = list(COMPETITORS.keys())
        start       = CURRENT_DATE - timedelta(days=self.MAX_HISTORY_DAYS)
        rows: List[Dict[str, Any]] = []

        for _ in range(n):
            prod     = rng.choice(products)
            seg      = rng.choice(segments)
            reg      = rng.choice(regions)
            comp     = rng.choice(competitors)
            info     = PRODUCT_CATALOG[prod]
            seg_inf  = CUSTOMER_SEGMENTS[seg]
            reg_inf  = REGIONS[reg]
            comp_inf = COMPETITORS[comp]
            channel  = rng.choice(CHANNEL_LIST, p=CHANNEL_WEIGHTS)

            base  = info["base_price"]
            cost  = info["cost"]
            min_p = info.get("min_price", base * 0.50)
            max_p = info.get("max_price", base * 2.00)

            comp_price  = base * comp_inf["relative_price"] * (1.0 + float(rng.normal(0, 0.05)))
            channel_adj = CHANNEL_PRICE_ADJ.get(channel, 0.0)
            price = base * (1.0 + float(rng.normal(0, 0.10)) + channel_adj) * (0.85 + 0.30 * float(rng.random()))
            price = float(np.clip(price, min_p, max_p))

            elasticity  = -1.2 - 0.5 * (1.0 - seg_inf["price_sensitivity"])
            days_ago    = int(rng.exponential(200))
            date        = start + timedelta(days=min(days_ago, self.MAX_HISTORY_DAYS))
            season_mult = MONTHLY_SEASONALITY.get(date.month, 1.0)
            demand_base = seg_inf["volume_multiplier"] * reg_inf["demand_index"] * self.BASE_DEMAND_SCALE * season_mult
            price_ratio = price / base
            demand      = max(1, int(demand_base * (price_ratio ** elasticity) + float(rng.normal(0, 1.5))))

            revenue   = price * demand
            cost_tot  = cost * demand
            margin    = (revenue - cost_tot) / revenue if revenue > 0.0 else 0.0

            win_base = 0.62 - 0.25 * max(0.0, price_ratio - 1.0)
            win_base += 0.05 * (1.0 - seg_inf["price_sensitivity"]) + 0.03 * seg_inf["loyalty"]
            if price < comp_price:
                win_base += 0.08 * (comp_price - price) / comp_price
            deal_won = int(float(rng.random()) < min(0.95, max(0.05, win_base)))

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
# MODEL COMPARATOR  (light: OLS/Ridge/Lasso/ElasticNet/BayesianRidge, 3-fold CV)
# ============================================================================

class ModelComparator:
    MIN_ELASTICITY_OBS: int = 30
    MIN_WIN_RATE_OBS: int = 50

    @staticmethod
    def fit_elasticity_models(
        df: pd.DataFrame,
        use_confidence: bool = True,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for seg in df["segment"].unique():
            sub = df[(df["segment"] == seg) & (df["volume"] > 0)].copy()
            if len(sub) < ModelComparator.MIN_ELASTICITY_OBS:
                log.warning("Segment %s: only %d rows — skipping elasticity fitting", seg, len(sub))
                continue

            try:
                sub["log_price"]    = np.log(sub["actual_price"])
                sub["log_discount"] = np.log1p(sub["discount_pct"])
                sub["log_volume"]   = np.log(sub["volume"])

                feat_cols = ["log_price", "log_discount"]
                X = sub[feat_cols].values
                y = sub["log_volume"].values

                candidate_models: Dict[str, Any] = {
                    "OLS":           LinearRegression(),
                    "Ridge":         Ridge(alpha=1.0),
                    "Lasso":         Lasso(alpha=0.05, max_iter=3_000),
                    "ElasticNet":    ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=3_000),
                    "BayesianRidge": BayesianRidge(),
                }

                best_rmse = np.inf
                best_name = "OLS"
                model_scores: Dict[str, float] = {}
                tscv = TimeSeriesSplit(n_splits=3)  # was 5 — lighter

                for name, base_mdl in candidate_models.items():
                    try:
                        cv_scores: List[float] = []
                        for train_idx, val_idx in tscv.split(X):
                            fold_mdl = _clone(base_mdl)
                            fold_mdl.fit(X[train_idx], y[train_idx])
                            y_pred = fold_mdl.predict(X[val_idx])
                            cv_scores.append(np.sqrt(mean_squared_error(y[val_idx], y_pred)))
                        avg_rmse = float(np.mean(cv_scores))
                        model_scores[name] = round(avg_rmse, 5)
                        if avg_rmse < best_rmse:
                            best_rmse = avg_rmse
                            best_name = name
                    except Exception as exc:
                        log.debug("CV failed for segment=%s model=%s: %s", seg, name, exc)

                try:
                    sm_mod       = smf.ols("log_volume ~ log_price + log_discount", data=sub).fit()
                    elasticity   = float(sm_mod.params.get("log_price", -1.2))
                    disc_lift    = float(sm_mod.params.get("log_discount", 0.1))
                    intercept    = float(sm_mod.params.get("Intercept", 0.0))
                    r_squared    = float(sm_mod.rsquared)
                    p_elasticity = float(sm_mod.pvalues.get("log_price", 1.0))
                    ci           = sm_mod.conf_int()
                    ci_lo        = float(ci.loc["log_price", 0])
                    ci_hi        = float(ci.loc["log_price", 1])
                    elasticity_conf = min(1.0, (len(sub) / 500) * 0.5 + r_squared * 0.5)
                except Exception as exc:
                    log.warning("statsmodels OLS failed for segment=%s: %s", seg, exc)
                    elasticity = -1.2; disc_lift = 0.15; intercept = 0.0
                    r_squared = 0.0; p_elasticity = 1.0
                    ci_lo = ci_hi = elasticity; elasticity_conf = 0.3

                results[seg] = {
                    "elasticity":       round(elasticity, 4),
                    "elasticity_ci_lo": round(ci_lo, 4),
                    "elasticity_ci_hi": round(ci_hi, 4),
                    "p_value":          round(p_elasticity, 4),
                    "disc_lift":        round(disc_lift, 4),
                    "intercept":        round(intercept, 4),
                    "r_squared":        round(r_squared, 4),
                    "n_obs":            len(sub),
                    "mean_price":       round(float(sub["actual_price"].mean()), 2),
                    "mean_volume":      round(float(sub["volume"].mean()), 2),
                    "mean_margin":      round(float(sub["margin_pct"].mean()), 2),
                    "best_model":       best_name,
                    "cv_rmse":          round(best_rmse, 5),
                    "all_model_rmse":   model_scores,
                    "confidence":       round(elasticity_conf, 3),
                }
            except Exception as exc:
                log.error("Elasticity fitting failed for segment=%s: %s", seg, exc)

        return results

    @staticmethod
    def fit_win_probability_model(
        df: pd.DataFrame,
        use_confidence: bool = True,
    ) -> Dict[str, Any]:
        sub = df.copy()
        sub["price_ratio"]      = sub["actual_price"] / sub["base_price"]
        sub["margin_norm"]      = sub["margin_pct"] / 100.0
        sc_max = float(sub["sales_cycle_days"].max()) if sub["sales_cycle_days"].max() > 0 else 90.0
        sub["sales_cycle_norm"] = sub["sales_cycle_days"] / sc_max
        sub["complexity_norm"]  = sub["complexity"] / 5.0
        sub["price_vs_competitor"] = (
            sub["actual_price"] / sub["competitor_price"]
            if "competitor_price" in sub.columns else 1.0
        )

        feature_cols = [
            "price_ratio", "discount_pct", "margin_norm",
            "sales_cycle_norm", "complexity_norm", "price_vs_competitor",
        ]
        sub = sub.dropna(subset=feature_cols + ["deal_won"])
        if len(sub) < ModelComparator.MIN_WIN_RATE_OBS:
            log.warning("Win-rate model: insufficient data (%d rows)", len(sub))
            return {}

        X = sub[feature_cols].values
        y = sub["deal_won"].values

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos = (n_neg / n_pos) if n_pos > 0 else 1.0

        candidates: Dict[str, Any] = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
            ]),
            "Ridge Logistic": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(C=0.1, max_iter=500, random_state=42)),
            ]),
        }

        best_auc  = 0.0
        best_name = "LogisticRegression"
        best_mdl  = None
        skf = TimeSeriesSplit(n_splits=3)  # was 5

        for name, mdl in candidates.items():
            try:
                aucs: List[float] = []
                for tr, vl in skf.split(X, y):
                    m = _clone(mdl)
                    m.fit(X[tr], y[tr])
                    prob = m.predict_proba(X[vl])[:, 1]
                    if len(np.unique(y[vl])) > 1:
                        aucs.append(roc_auc_score(y[vl], prob))
                if aucs:
                    avg = float(np.mean(aucs))
                    if avg > best_auc:
                        best_auc = avg
                        best_name = name
                        best_mdl = _clone(mdl)
            except Exception as exc:
                log.debug("Win model CV failed for %s: %s", name, exc)

        if best_mdl is None:
            best_mdl = _clone(candidates["LogisticRegression"])

        best_mdl.fit(X, y)

        return {
            "model":        best_mdl,
            "feature_cols": feature_cols,
            "auc":          round(best_auc, 4),
            "model_name":   best_name,
            "n_obs":        len(sub),
            "win_rate":     round(float(y.mean()), 4),
        }

    @staticmethod
    def predict_win_probability(
        win_bundle: Dict[str, Any],
        price_ratio: float,
        discount_pct: float,
        margin_pct: float,
        sales_cycle_days: float = 45,
        complexity: float = 2,
        price_vs_competitor: float = 1.0,
    ) -> float:
        model = win_bundle.get("model")
        if model is None:
            return max(0.05, min(0.95, 0.62 - 0.25 * max(0.0, price_ratio - 1.0)))
        features = np.array([[
            price_ratio, discount_pct, margin_pct / 100.0,
            min(sales_cycle_days / 90.0, 1.0), complexity / 5.0, price_vs_competitor,
        ]])
        try:
            return float(model.predict_proba(features)[0, 1])
        except Exception:
            return 0.5


# ============================================================================
# PRICING OPTIMIZER
# ============================================================================

class PricingOptimizer:
    @staticmethod
    def calculate(
        product: str,
        segment: str,
        elasticity_results: Dict[str, Any],
        objective: str = "max_revenue",
        competitor_prices: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        info = PRODUCT_CATALOG.get(product, {})
        base = float(info.get("base_price", 1000))
        cost = float(info.get("cost", base * 0.4))
        lower = float(info.get("min_price", base * 0.5))
        upper = float(info.get("max_price", base * 2.0))

        if lower >= upper:
            log.warning("PricingOptimizer: lower=%.2f >= upper=%.2f for %s — clamping", lower, upper, product)
            upper = lower * 2.0

        seg_data  = elasticity_results.get(segment, {})
        elasticity = float(seg_data.get("elasticity", -1.2))

        def demand(p: float) -> float:
            return max(0.01, (p / base) ** elasticity)

        def neg_revenue(p: float) -> float:
            return -p * demand(p)

        def neg_profit(p: float) -> float:
            return -(p - cost) * demand(p)

        def neg_margin(p: float) -> float:
            if p <= 0:
                return 0.0
            return -((p - cost) / p) * demand(p)

        def vs_competitor(p: float) -> float:
            if not competitor_prices:
                return neg_revenue(p)
            avg_comp = float(np.mean(competitor_prices))
            return -(p * demand(p) * (1 + 0.1 * max(0.0, avg_comp - p) / avg_comp))

        obj_map = {
            "max_revenue":    neg_revenue,
            "max_profit":     neg_profit,
            "max_margin":     neg_margin,
            "vs_competitor":  vs_competitor,
        }
        fn = obj_map.get(objective, neg_revenue)

        try:
            result   = minimize_scalar(fn, bounds=(lower, upper), method="bounded")
            opt_price = float(result.x)
        except Exception:
            opt_price = base

        opt_demand   = demand(opt_price)
        opt_revenue  = opt_price * opt_demand
        opt_profit   = (opt_price - cost) * opt_demand
        opt_margin   = (opt_price - cost) / opt_price if opt_price > 0 else 0.0

        base_demand  = demand(base)
        base_revenue = base * base_demand
        base_profit  = (base - cost) * base_demand

        return {
            "product":         product,
            "segment":         segment,
            "objective":       objective,
            "base_price":      round(base, 2),
            "optimal_price":   round(opt_price, 2),
            "price_change_pct": round((opt_price - base) / base * 100, 2),
            "optimal_demand":  round(opt_demand, 4),
            "optimal_revenue": round(opt_revenue, 2),
            "optimal_profit":  round(opt_profit, 2),
            "optimal_margin":  round(opt_margin * 100, 2),
            "base_revenue":    round(base_revenue, 2),
            "base_profit":     round(base_profit, 2),
            "revenue_lift":    round((opt_revenue - base_revenue) / max(base_revenue, 1) * 100, 2),
            "profit_lift":     round((opt_profit - base_profit) / max(abs(base_profit), 1) * 100, 2),
            "elasticity":      round(elasticity, 4),
            "cost":            round(cost, 2),
        }


# ============================================================================
# SIMULATION ENGINE  (plain numpy, no scipy Sobol)
# ============================================================================

class SimulationEngine:
    @staticmethod
    def simulate_revenue_scenarios(
        df: pd.DataFrame,
        elasticity_results: Dict[str, Any],
        n_scenarios: int = 300,
        price_range: Tuple[float, float] = (-0.30, 0.30),
    ) -> pd.DataFrame:
        rng = np.random.default_rng(42)

        # Plain uniform random instead of Sobol (no scipy.stats.qmc needed)
        price_changes  = rng.uniform(price_range[0], price_range[1], n_scenarios)
        volume_changes = rng.uniform(-0.20, 0.20, n_scenarios)
        cost_changes   = rng.uniform(-0.10, 0.10, n_scenarios)

        base_rev = float(df["revenue"].sum())
        base_cost = float(df["cost"].sum()) if "cost" in df.columns else base_rev * 0.40
        base_profit = base_rev - base_cost

        seg_elast = {s: float(v.get("elasticity", -1.2)) for s, v in elasticity_results.items()}
        avg_elast = float(np.mean(list(seg_elast.values()))) if seg_elast else -1.2

        rows = []
        for pc, vc, cc in zip(price_changes, volume_changes, cost_changes):
            demand_effect = (1 + pc) ** avg_elast
            new_price_idx = 1 + pc
            new_vol_idx   = demand_effect * (1 + vc)
            new_rev       = base_rev * new_price_idx * new_vol_idx
            new_cost      = base_cost * (1 + cc)
            new_profit    = new_rev - new_cost
            margin        = (new_rev - new_cost) / new_rev if new_rev > 0 else 0.0
            rows.append({
                "price_change_pct":  round(pc * 100, 2),
                "volume_change_pct": round(vc * 100, 2),
                "cost_change_pct":   round(cc * 100, 2),
                "revenue":           round(new_rev, 2),
                "profit":            round(new_profit, 2),
                "margin_pct":        round(margin * 100, 2),
                "revenue_delta":     round(new_rev - base_rev, 2),
                "profit_delta":      round(new_profit - base_profit, 2),
            })

        return pd.DataFrame(rows)


# ============================================================================
# COMPETITIVE ANALYZER
# ============================================================================

class CompetitiveAnalyzer:
    @staticmethod
    def get_price_score(
        df: pd.DataFrame,
        product: str,
    ) -> Dict[str, Any]:
        sub = df[df["product"] == product].copy()
        if sub.empty:
            return {}

        our_price = float(sub["actual_price"].mean())
        comp_df   = sub.dropna(subset=["competitor_price"])

        if comp_df.empty:
            return {
                "product": product, "our_avg_price": round(our_price, 2),
                "market_avg": round(our_price, 2), "price_index": 1.0,
                "percentile": 50.0, "n_competitors": 0,
            }

        weighted_confidence = float(comp_df.get("confidence_score", pd.Series(1.0)).mean()) or 1.0
        market_avg = float(comp_df["competitor_price"].mean())
        price_index = our_price / market_avg if market_avg > 0 else 1.0
        all_prices = list(comp_df["competitor_price"].values) + [our_price]
        percentile = float(
            (sum(1 for p in all_prices if p <= our_price) / len(all_prices)) * 100
        )

        return {
            "product":        product,
            "our_avg_price":  round(our_price, 2),
            "market_avg":     round(market_avg, 2),
            "price_index":    round(price_index, 4),
            "percentile":     round(percentile, 1),
            "n_competitors":  int(comp_df["competitor"].nunique()),
            "confidence":     round(weighted_confidence, 3),
        }

    @staticmethod
    def track_price_trends(
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df = df.dropna(subset=["date"])

        trends: Dict[str, Any] = {}
        for comp in df["competitor"].dropna().unique():
            cdf = df[df["competitor"] == comp].sort_values("date")
            if len(cdf) < 3:
                continue
            prices = cdf["competitor_price"].values
            x      = np.arange(len(prices), dtype=float)
            try:
                slope = float(np.polyfit(x, prices, 1)[0]) if prices.std() > 0 else 0.0
            except Exception:
                slope = 0.0
            trends[str(comp)] = {
                "avg_price":    round(float(prices.mean()), 2),
                "price_trend":  round(slope, 4),
                "n_obs":        len(cdf),
                "latest_price": round(float(prices[-1]), 2),
            }
        return trends


# ============================================================================
# REVENUE FORECASTER  (exp-smoothing + linear; no ARIMA)
# ============================================================================

class RevenueForecaster:
    @staticmethod
    def forecast(
        df: pd.DataFrame,
        horizon: int = 12,
    ) -> Dict[str, Any]:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["month"] = df["date"].dt.to_period("M")

        monthly = df.groupby("month")["revenue"].sum().sort_index()
        if len(monthly) < 6:
            log.warning("RevenueForecaster: fewer than 6 monthly obs – returning flat forecast")
            last = float(monthly.iloc[-1]) if len(monthly) > 0 else 0.0
            return {
                "historical": monthly.astype(float).to_dict(),
                "forecast":   {str(i + 1): last for i in range(horizon)},
                "method":     "flat",
            }

        values = monthly.values.astype(float)

        # Linear trend component
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        trend_forecast = [intercept + slope * (len(values) + h) for h in range(horizon)]

        # Exponential smoothing component
        try:
            es_model = ExponentialSmoothing(
                values, trend="add", seasonal="add" if len(values) >= 24 else None,
                seasonal_periods=12 if len(values) >= 24 else None,
            ).fit(optimized=True)
            es_forecast = list(es_model.forecast(horizon))
        except Exception:
            es_forecast = trend_forecast[:]

        # Ensemble average
        ensemble = [(t + e) / 2 for t, e in zip(trend_forecast, es_forecast)]

        # Build period labels
        last_period = monthly.index[-1]
        forecast_periods = {}
        for h in range(horizon):
            p = last_period + (h + 1)
            forecast_periods[str(p)] = round(max(0.0, ensemble[h]), 2)

        return {
            "historical":  {str(k): round(float(v), 2) for k, v in monthly.items()},
            "forecast":    forecast_periods,
            "method":      "ensemble_linear_es",
            "trend_slope": round(float(slope), 2),
            "n_history":   len(values),
        }


# ============================================================================
# DEAL SCORER
# ============================================================================

class DealScorer:
    SCORE_WEIGHTS: Dict[str, float] = {
        "margin":      0.30,
        "win_prob":    0.30,
        "revenue":     0.20,
        "competitive": 0.20,
    }

    @classmethod
    def get_rating(cls, score: float) -> Tuple[str, str]:
        if score >= 80:   return "Excellent", "#2ee89a"
        if score >= 65:   return "Good",      "#4f9eff"
        if score >= 50:   return "Fair",       "#f5a623"
        return "Poor", "#ff4060"

    @classmethod
    def score_deal(
        cls,
        product: str,
        proposed_price: float,
        volume: int,
        customer_segment: str,
        region: str,
        elasticity_results: Dict[str, Any],
        win_bundle: Optional[Dict[str, Any]] = None,
        competitor_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        info = PRODUCT_CATALOG.get(product, {})
        base = float(info.get("base_price", proposed_price))
        cost = float(info.get("cost", proposed_price * 0.4))

        if proposed_price <= 0:
            return {"error": "Proposed price must be > 0"}

        revenue  = proposed_price * volume
        margin   = (proposed_price - cost) / proposed_price
        discount = max(0.0, (base - proposed_price) / base * 100)

        seg_elast = float(elasticity_results.get(customer_segment, {}).get("elasticity", -1.2))
        price_ratio = proposed_price / base

        win_prob = ModelComparator.predict_win_probability(
            win_bundle or {},
            price_ratio    = price_ratio,
            discount_pct   = discount,
            margin_pct     = margin * 100,
            sales_cycle_days = 45,
            complexity     = float(info.get("complexity", 2)),
            price_vs_competitor = (proposed_price / competitor_price) if competitor_price else 1.0,
        )

        comp_score = 0.5
        if competitor_price:
            comp_score = min(1.0, max(0.0, 1.0 - (proposed_price - competitor_price) / competitor_price))

        rev_norm    = min(1.0, revenue / (base * volume))
        margin_norm = min(1.0, max(0.0, margin))

        raw_score = (
            cls.SCORE_WEIGHTS["margin"]      * margin_norm +
            cls.SCORE_WEIGHTS["win_prob"]    * win_prob +
            cls.SCORE_WEIGHTS["revenue"]     * rev_norm +
            cls.SCORE_WEIGHTS["competitive"] * comp_score
        )
        score = round(raw_score * 100, 1)
        rating, rating_color = cls.get_rating(score)

        return {
            "product":          product,
            "proposed_price":   round(proposed_price, 2),
            "volume":           volume,
            "revenue":          round(revenue, 2),
            "margin_pct":       round(margin * 100, 2),
            "discount_pct":     round(discount, 2),
            "win_probability":  round(win_prob, 4),
            "score":            score,
            "rating":           rating,
            "rating_color":     rating_color,
            "competitive_score": round(comp_score * 100, 1),
            "elasticity":       round(seg_elast, 4),
            "base_price":       round(base, 2),
            "cost":             round(cost, 2),
        }


# ============================================================================
# MARGIN WATERFALL BUILDER
# ============================================================================

class MarginWaterfallBuilder:
    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        total_revenue  = float(df["revenue"].sum())
        if total_revenue <= 0:
            return pd.DataFrame()

        total_cost     = float(df["cost"].sum()) if "cost" in df.columns else total_revenue * 0.40
        discount_pct   = float(df["discount_pct"].mean()) / 100 if "discount_pct" in df.columns else 0.05
        discount_amount = min(total_revenue * discount_pct, total_revenue)
        net_revenue    = total_revenue - discount_amount
        gross_profit   = net_revenue - total_cost

        rows = [
            {"label": "Gross Revenue",   "value": total_revenue,          "type": "total"},
            {"label": "Volume Discount",  "value": -discount_amount,       "type": "decrease"},
            {"label": "Net Revenue",      "value": net_revenue,            "type": "subtotal"},
            {"label": "COGS",             "value": -total_cost,            "type": "decrease"},
            {"label": "Gross Profit",     "value": gross_profit,           "type": "total"},
        ]

        return pd.DataFrame(rows)


# ============================================================================
# PORTFOLIO SCORER
# ============================================================================

class PortfolioScorer:
    @staticmethod
    def score_products(
        df: pd.DataFrame,
        elasticity_results: Dict[str, Any],
    ) -> pd.DataFrame:
        rows = []
        for prod in df["product"].unique():
            sub = df[df["product"] == prod]
            if sub.empty:
                continue

            seg      = sub["segment"].iloc[0]
            rev      = float(sub["revenue"].sum())
            margin   = float(sub["margin_pct"].mean()) / 100 if "margin_pct" in sub.columns else 0.35
            win_rate = float(sub["deal_won"].mean()) if "deal_won" in sub.columns else 0.5
            loyalty  = float(sub.get("renewal_flag", pd.Series(0.7)).mean())

            elast    = float(elasticity_results.get(seg, {}).get("elasticity", -1.2))
            all_rev  = float(df["revenue"].sum())
            moat     = min(1.0, max(0.0, 1.0 - abs(elast) / 3.0))
            growth   = float(sub.sort_values("date").groupby(
                pd.Grouper(key="date", freq="Q"))["revenue"].sum().pct_change().mean()
            ) if "date" in sub.columns else 0.0

            market_position = 1.0 / (1.0 + np.exp(-2.0 * (margin - 0.3)))
            confidence = float(sub["confidence_score"].mean()) if "confidence_score" in sub.columns else 1.0

            raw = {
                "margin_pct":            margin,
                "revenue_growth":        growth,
                "win_rate":              win_rate,
                "customer_loyalty":      loyalty,
                "competitive_moat":      moat,
                "market_position":       market_position,
                "price_competitiveness": 1.0 - market_position,
            }
            rows.append({
                "product":         prod,
                "segment":         seg,
                "revenue":         round(rev, 2),
                "margin_pct":      round(margin * 100, 2),
                "win_rate":        round(win_rate * 100, 2),
                "growth_pct":      round(growth * 100, 2),
                "loyalty":         round(loyalty * 100, 2),
                "elasticity":      round(-elast, 4),
                "market_position": round(market_position * 100, 2),
                "confidence":      round(confidence, 3),
                **{f"_raw_{k}": v for k, v in raw.items()},
            })

        scored_df = pd.DataFrame(rows)
        if scored_df.empty:
            return scored_df

        for key in SCORING_WEIGHTS:
            col = f"_raw_{key}"
            if col not in scored_df.columns:
                scored_df[f"_norm_{key}"] = 0.5
                continue
            mn = scored_df[col].min()
            mx = scored_df[col].max()
            denom = mx - mn
            scored_df[f"_norm_{key}"] = (
                (scored_df[col] - mn) / denom if denom > 1e-9 else 0.5
            )

        scored_df["score"] = round(
            sum(SCORING_WEIGHTS[k] * scored_df.get(f"_norm_{k}", pd.Series(0.5, index=scored_df.index))
                for k in SCORING_WEIGHTS) * 100.0,
            4,
        )
        scored_df["adjusted_score"] = scored_df["score"] * scored_df["confidence"]

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
    @staticmethod
    def enhance_elasticity_with_market_data(
        elasticity_results: Dict[str, Any],
        market_data: List[Dict],
    ) -> Dict[str, Any]:
        enhanced = {k: dict(v) for k, v in elasticity_results.items()}

        seg_base_price: Dict[str, float] = {}
        for info in PRODUCT_CATALOG.values():
            seg = info.get("segment")
            if seg and seg not in seg_base_price:
                seg_base_price[seg] = float(info["base_price"])

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
                seg_data["elasticity"] = old_elast * 1.1
            elif market_ratio > 1.1:
                seg_data["elasticity"] = old_elast * 0.9
            seg_data["market_adjusted"]    = True
            seg_data["market_price_ratio"] = round(market_ratio, 3)

        return enhanced

    @staticmethod
    def detect_pricing_opportunities(
        df: pd.DataFrame,
        competitor_data: List[Dict],
    ) -> List[Dict]:
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
            matching  = [
                item
                for cp, items in comp_by_product.items()
                for item in items
                if any(kw in cp.lower() for kw in info.get("keywords", []))
            ]
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
                    "product": prod, "type": "price_reduction_opportunity",
                    "description": f"Price {price_gap:.0f}% above competitors with high margin",
                    "current_price": round(avg_price, 2), "market_price": round(avg_comp, 2),
                    "potential_gain": round((avg_price - avg_comp) * vol_total, 2), "confidence": 0.7,
                })
            elif price_gap < -15.0:
                opportunities.append({
                    "product": prod, "type": "price_increase_opportunity",
                    "description": f"Price {abs(price_gap):.0f}% below market average",
                    "current_price": round(avg_price, 2), "market_price": round(avg_comp, 2),
                    "potential_gain": round((avg_comp - avg_price) * vol_total, 2), "confidence": 0.6,
                })

        return sorted(opportunities, key=lambda x: x.get("potential_gain", 0.0), reverse=True)


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
