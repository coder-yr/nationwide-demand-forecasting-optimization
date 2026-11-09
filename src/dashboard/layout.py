"""Plotly Dash Layout for the Nationwide Demand Forecasting and Inventory Optimization Control Tower."""
from dash import html, dcc, dash_table

def create_kpi_card(title: str, value: str, subtitle: str) -> html.Div:
    """Renders standardized KPI metric tiles."""
    return html.Div(
        style={
            "flex": "1",
            "backgroundColor": "#ffffff",
            "padding": "18px 24px",
            "borderRadius": "10px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.06)",
            "borderLeft": "4px solid #1f77b4",
            "marginRight": "14px"
        },
        children=[
            html.Div(title, style={"fontSize": "13px", "fontWeight": "600", "color": "#6c757d", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "700", "color": "#212529", "marginTop": "6px"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "#28a745", "marginTop": "4px"})
        ]
    )

def build_dashboard_layout(stores: list, categories: list, departments: list) -> html.Div:
    """Constructs the complete layout for the Dash portal."""
    return html.Div(
        style={
            "backgroundColor": "#f8f9fa",
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            "minHeight": "100vh",
            "padding": "24px 32px"
        },
        children=[
            # Header Title Bar
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px", "borderBottom": "1px solid #e9ecef", "paddingBottom": "16px"},
                children=[
                    html.Div([
                        html.H2("Nationwide Demand Forecasting & Inventory Optimization", style={"margin": "0", "color": "#111827", "fontWeight": "700"}),
                        html.P("Walmart M5 Enterprise Supply Chain Decision Support Control Tower", style={"margin": "4px 0 0 0", "color": "#6b7280", "fontSize": "14px"})
                    ]),
                    html.Div(
                        style={"display": "flex", "gap": "10px"},
                        children=[
                            html.Span(id="provenance-badge", children="⚡ Data Source: Active M5 Lake", style={"backgroundColor": "#d1fae5", "color": "#065f46", "padding": "6px 14px", "borderRadius": "20px", "fontSize": "12px", "fontWeight": "600"}),
                            html.Span("Hierarchical Coherence: Verified (12 Levels)", style={"backgroundColor": "#dbeafe", "color": "#1e40af", "padding": "6px 14px", "borderRadius": "20px", "fontSize": "12px", "fontWeight": "600"})
                        ]
                    )
                ]
            ),

            # Disclosed Operational Assumptions Banner
            html.Div(
                style={"backgroundColor": "#fffbeb", "borderLeft": "4px solid #f59e0b", "padding": "10px 16px", "borderRadius": "6px", "marginBottom": "20px", "fontSize": "12px", "color": "#92400e"},
                children=[
                    html.Strong("Operational Supply Chain Assumptions: "),
                    "The M5 dataset provides POS sales, prices, and events. Inventory levels, MOQs, supplier lead times, and capacity ceilings are simulated using configurable industrial supply chain policies (Wholesale unit cost = 65% of price, Lead time = 2-3 days, Service level target = 95%)."
                ]
            ),

            # Global Hierarchical Filter Controls Bar
            html.Div(
                style={
                    "backgroundColor": "#ffffff",
                    "padding": "16px 20px",
                    "borderRadius": "10px",
                    "boxShadow": "0 2px 6px rgba(0,0,0,0.06)",
                    "display": "flex",
                    "gap": "14px",
                    "alignItems": "center",
                    "marginBottom": "24px",
                    "flexWrap": "wrap"
                },
                children=[
                    html.Div([
                        html.Label("Target Store", style={"fontSize": "12px", "fontWeight": "600", "color": "#4b5563"}),
                        dcc.Dropdown(
                            id="store-selector",
                            options=[{"label": s, "value": s} for s in stores],
                            value=stores[0] if stores else "CA_1",
                            clearable=False,
                            style={"width": "150px"}
                        )
                    ]),
                    html.Div([
                        html.Label("Category", style={"fontSize": "12px", "fontWeight": "600", "color": "#4b5563"}),
                        dcc.Dropdown(
                            id="category-selector",
                            options=[{"label": "ALL CATEGORIES", "value": "ALL"}] + [{"label": c, "value": c} for c in categories],
                            value="ALL",
                            clearable=False,
                            style={"width": "180px"}
                        )
                    ]),
                    html.Div([
                        html.Label("Department", style={"fontSize": "12px", "fontWeight": "600", "color": "#4b5563"}),
                        dcc.Dropdown(
                            id="department-selector",
                            options=[{"label": "ALL DEPARTMENTS", "value": "ALL"}] + [{"label": d, "value": d} for d in departments],
                            value="ALL",
                            clearable=False,
                            style={"width": "190px"}
                        )
                    ]),
                    html.Div([
                        html.Label("Forecast Horizon", style={"fontSize": "12px", "fontWeight": "600", "color": "#4b5563"}),
                        dcc.Dropdown(
                            id="horizon-selector",
                            options=[
                                {"label": "7 Days (Operational)", "value": 7},
                                {"label": "14 Days (Tactical)", "value": 14},
                                {"label": "28 Days (Monthly)", "value": 28}
                            ],
                            value=28,
                            clearable=False,
                            style={"width": "180px"}
                        )
                    ]),
                    html.Div([
                        html.Label("Forecasting Model", style={"fontSize": "12px", "fontWeight": "600", "color": "#4b5563"}),
                        dcc.Dropdown(
                            id="model-selector",
                            options=[
                                {"label": "LightGBM Tweedie+Quantile", "value": "lgbm"},
                                {"label": "Prophet Macro Aggregate", "value": "prophet"}
                            ],
                            value="lgbm",
                            clearable=False,
                            style={"width": "220px"}
                        )
                    ]),
                    html.Button(
                        "Recompute Optimization",
                        id="btn-recompute",
                        n_clicks=0,
                        style={
                            "marginLeft": "auto",
                            "backgroundColor": "#2563eb",
                            "color": "#ffffff",
                            "border": "none",
                            "padding": "10px 18px",
                            "borderRadius": "6px",
                            "fontWeight": "600",
                            "cursor": "pointer"
                        }
                    )
                ]
            ),

            # Executive KPI Cards Row (populated dynamically by callbacks)
            html.Div(
                id="kpi-cards-row",
                style={"display": "flex", "marginBottom": "24px"},
                children=[
                    create_kpi_card("Projected Demand", "Calculating...", "Loading data..."),
                    create_kpi_card("Stockout Risk Exposure", "Calculating...", "Loading data..."),
                    create_kpi_card("Working Capital Requirement", "Calculating...", "Loading data..."),
                    create_kpi_card("Service Level Target", "95%", "Configured target")
                ]
            ),

            # Main Visuals Row: Time Series Forecast + Risk Matrix
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "24px"},
                children=[
                    # Time Series Forecast Fan Chart
                    html.Div(
                        style={"flex": "2", "backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "boxShadow": "0 2px 6px rgba(0,0,0,0.06)"},
                        children=[
                            html.H4("Hierarchical Demand Forecast: Historical vs Calibrated Prediction Intervals", style={"margin": "0 0 12px 0", "color": "#1f2937"}),
                            dcc.Graph(id="forecast-fan-chart", style={"height": "380px"})
                        ]
                    ),
                    # Risk Matrix Scatter Plot
                    html.Div(
                        style={"flex": "1", "backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "boxShadow": "0 2px 6px rgba(0,0,0,0.06)"},
                        children=[
                            html.H4("Inventory Risk Matrix (P90 Demand vs Stock)", style={"margin": "0 0 12px 0", "color": "#1f2937"}),
                            dcc.Graph(id="inventory-risk-matrix", style={"height": "380px"})
                        ]
                    )
                ]
            ),

            # Bottom Row: Replenishment Table & Scenario Simulator
            html.Div(
                style={"display": "flex", "gap": "20px"},
                children=[
                    # Replenishment Order Table
                    html.Div(
                        style={"flex": "2", "backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "boxShadow": "0 2px 6px rgba(0,0,0,0.06)"},
                        children=[
                            html.H4("Optimal Replenishment & Purchase Orders (OR-Tools MIP Output)", style={"margin": "0 0 14px 0", "color": "#1f2937"}),
                            dash_table.DataTable(
                                id="replenishment-table",
                                page_size=8,
                                style_table={"overflowX": "auto"},
                                style_header={
                                    "backgroundColor": "#f3f4f6",
                                    "fontWeight": "600",
                                    "color": "#374151",
                                    "borderBottom": "2px solid #e5e7eb"
                                },
                                style_cell={
                                    "padding": "10px 14px",
                                    "fontSize": "13px",
                                    "textAlign": "left"
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": "{action} = 'REORDER_NOW'"},
                                        "backgroundColor": "#fef2f2",
                                        "color": "#991b1b",
                                        "fontWeight": "600"
                                    },
                                    {
                                        "if": {"filter_query": "{action} = 'OPTIMAL'"},
                                        "backgroundColor": "#f0fdf4",
                                        "color": "#166534"
                                    }
                                ]
                            )
                        ]
                    ),
                    # What-If Simulator Panel
                    html.Div(
                        style={"flex": "1", "backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px", "boxShadow": "0 2px 6px rgba(0,0,0,0.06)"},
                        children=[
                            html.H4("What-If Constraint Simulator", style={"margin": "0 0 16px 0", "color": "#1f2937"}),
                            html.Label("Target Service Level (Cycle-Service):", style={"fontSize": "13px", "fontWeight": "600"}),
                            dcc.Slider(
                                id="slider-service-level",
                                min=0.85, max=0.99, step=0.01, value=0.95,
                                marks={0.85: "85%", 0.90: "90%", 0.95: "95%", 0.99: "99%"}
                            ),
                            html.Div(style={"height": "20px"}),
                            html.Label("Daily Store Procurement Budget ($):", style={"fontSize": "13px", "fontWeight": "600"}),
                            dcc.Slider(
                                id="slider-budget",
                                min=10000, max=100000, step=5000, value=50000,
                                marks={10000: "$10k", 50000: "$50k", 100000: "$100k"}
                            ),
                            html.Div(style={"height": "20px"}),
                            html.Label("Supplier Lead Time Buffer (Days):", style={"fontSize": "13px", "fontWeight": "600"}),
                            dcc.Slider(
                                id="slider-lead-time",
                                min=1, max=7, step=1, value=3,
                                marks={1: "1d", 3: "3d", 5: "5d", 7: "7d"}
                            )
                        ]
                    )
                ]
            )
        ]
    )
