"""Reactive callbacks for Plotly Dash interactive control tower."""
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, html

from src.optimization.safety_stock import calculate_safety_stock_and_rop
from src.dashboard.layout import create_kpi_card

def register_callbacks(app, data_loader):
    """Binds all reactive UI interaction logic to the Dash app."""

    @app.callback(
        [
            Output("forecast-fan-chart", "figure"),
            Output("inventory-risk-matrix", "figure"),
            Output("replenishment-table", "data"),
            Output("replenishment-table", "columns"),
            Output("kpi-cards-row", "children"),
            Output("provenance-badge", "children"),
            Output("provenance-badge", "style")
        ],
        [
            Input("store-selector", "value"),
            Input("category-selector", "value"),
            Input("department-selector", "value"),
            Input("horizon-selector", "value"),
            Input("model-selector", "value"),
            Input("slider-service-level", "value"),
            Input("slider-budget", "value"),
            Input("slider-lead-time", "value"),
            Input("btn-recompute", "n_clicks")
        ]
    )
    def update_dashboard(store_id, category, department, horizon, model_choice, service_level, budget, lead_time, n_clicks):
        try:
            horizon = int(horizon) if horizon is not None else 28
        except (ValueError, TypeError):
            horizon = 28

        try:
            service_level = float(service_level) if service_level is not None else 0.95
        except (ValueError, TypeError):
            service_level = 0.95

        try:
            lead_time = int(lead_time) if lead_time is not None else 3
        except (ValueError, TypeError):
            lead_time = 3

        try:
            budget = float(budget) if budget is not None else 50000.0
        except (ValueError, TypeError):
            budget = 50000.0

        # 1. Fetch series from data loader (persisted forecast artifacts or data lake)
        history_dates, history_sales, future_dates, p10, p50, p90, is_demo = data_loader.get_forecast_series(
            store_id=store_id,
            category=category,
            department=department,
            horizon_days=horizon
        )

        badge_text = "⚠️ Mode: Synthetic / Fallback" if is_demo else "⚡ Data Source: Active M5 Parquet Lake"
        badge_style = {
            "backgroundColor": "#fee2e2" if is_demo else "#d1fae5",
            "color": "#991b1b" if is_demo else "#065f46",
            "padding": "6px 14px",
            "borderRadius": "20px",
            "fontSize": "12px",
            "fontWeight": "600"
        }

        # 2. Build Fan Chart
        fig_fan = go.Figure()

        if len(history_dates) > 0 and len(history_sales) > 0:
            fig_fan.add_trace(go.Scatter(
                x=history_dates,
                y=history_sales,
                mode="lines",
                name="Historical Sales",
                line=dict(color="#4b5563", width=2)
            ))

        # P90 upper bound
        fig_fan.add_trace(go.Scatter(
            x=future_dates,
            y=p90,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="none"
        ))

        # P10 lower bound with shaded interval
        fig_fan.add_trace(go.Scatter(
            x=future_dates,
            y=p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            name="80% Prediction Interval (P10-P90)"
        ))

        # P50 median forecast
        fig_fan.add_trace(go.Scatter(
            x=future_dates,
            y=p50,
            mode="lines+markers",
            name="P50 Expected Demand",
            line=dict(color="#2563eb", width=3, dash="solid")
        ))

        fig_fan.update_layout(
            margin=dict(l=40, r=20, t=20, b=30),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#fcfcfc",
            paper_bgcolor="#ffffff",
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", title="Daily Unit Sales")
        )

        # 3. Build Risk Matrix & Replenishment Table
        items_df = data_loader.get_inventory_snapshot(store_id, category, department)

        # Apply Safety stock & ROP calculation
        ss_metrics = calculate_safety_stock_and_rop(
            p10_demand=items_df["p10_demand"].values,
            p50_demand=items_df["p50_demand"].values,
            p90_demand=items_df["p90_demand"].values,
            lead_time_days=lead_time,
            service_level=service_level
        )

        items_df["safety_stock"] = ss_metrics["safety_stock"]
        items_df["reorder_point"] = ss_metrics["reorder_point"]

        # Determine replenishment action and recommended order
        items_df["net_need"] = np.maximum(items_df["reorder_point"] - items_df["on_hand_stock"], 0)
        items_df["recommended_order"] = np.where(
            items_df["net_need"] > 0,
            np.maximum(items_df["net_need"], items_df["moq"]),
            0
        )
        items_df["action"] = np.where(
            items_df["on_hand_stock"] < items_df["safety_stock"],
            "REORDER_NOW",
            np.where(items_df["on_hand_stock"] > items_df["p90_demand"] * 2.5, "HOLD (EXCESS)", "OPTIMAL")
        )
        items_df["order_cost"] = (items_df["recommended_order"] * items_df["unit_cost"]).round(2)

        # Build Risk Matrix Scatter Figure
        color_map = {"REORDER_NOW": "#ef4444", "HOLD (EXCESS)": "#f59e0b", "OPTIMAL": "#10b981"}
        fig_risk = go.Figure()
        for action, color in color_map.items():
            sub = items_df[items_df["action"] == action]
            if not sub.empty:
                fig_risk.add_trace(go.Scatter(
                    x=sub["p90_demand"],
                    y=sub["on_hand_stock"],
                    mode="markers",
                    name=action,
                    marker=dict(size=9, color=color, opacity=0.8),
                    text=sub["item_id"],
                    hovertemplate="<b>%{text}</b><br>P90 Demand: %{x}<br>Stock: %{y}<extra></extra>"
                ))

        max_val = max(items_df["p90_demand"].max(), items_df["on_hand_stock"].max(), 50)
        fig_risk.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Parity Line",
            line=dict(color="#cbd5e1", dash="dash")
        ))
        fig_risk.update_layout(
            margin=dict(l=40, r=20, t=20, b=30),
            plot_bgcolor="#fcfcfc",
            paper_bgcolor="#ffffff",
            xaxis=dict(title="Projected P90 Demand (Units)", showgrid=True, gridcolor="#f3f4f6"),
            yaxis=dict(title="Current On-Hand Stock (Units)", showgrid=True, gridcolor="#f3f4f6"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Format DataTable rows & columns
        display_df = items_df[[
            "item_id", "dept_id", "on_hand_stock", "safety_stock", "reorder_point",
            "recommended_order", "moq", "order_cost", "action"
        ]].copy()

        columns = [{"name": c.replace("_", " ").title(), "id": c} for c in display_df.columns]
        table_data = display_df.head(25).to_dict("records")

        # Recompute dynamic KPIs
        total_projected_demand = int(np.sum(p50))
        total_spend = float(items_df["order_cost"].sum())
        stockout_items = int((items_df["action"] == "REORDER_NOW").sum())
        stockout_pct = (stockout_items / max(len(items_df), 1)) * 100.0

        kpi_cards = [
            create_kpi_card("Projected Horizon Demand", f"{total_projected_demand:,} Units", f"Store {store_id} ({category})"),
            create_kpi_card("Stockout Risk Items", f"{stockout_pct:.1f}% ({stockout_items} SKUs)", "Immediate reorder required" if stockout_items > 0 else "All SKUs safely buffered"),
            create_kpi_card("Recommended Spend", f"${total_spend:,.2f}", f"Daily budget: ${budget:,.0f}"),
            create_kpi_card("Service Level Target", f"{int(service_level*100)}%", f"Lead buffer: +{lead_time}d")
        ]

        return fig_fan, fig_risk, table_data, columns, kpi_cards, badge_text, badge_style
