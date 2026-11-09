"""Plotly Dash Enterprise Portal Application Entrypoint."""
import sys
import dash
from src.dashboard.layout import build_dashboard_layout
from src.dashboard.callbacks import register_callbacks
from src.dashboard.data_loader import DashboardDataLoader

def create_app() -> dash.Dash:
    """Factory function to instantiate and configure the Dash control tower app."""
    app = dash.Dash(
        __name__,
        title="Nationwide Inventory Control Tower",
        update_title=None,
        suppress_callback_exceptions=True
    )

    data_loader = DashboardDataLoader()
    stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]

    app.layout = build_dashboard_layout(stores=stores, categories=categories)
    register_callbacks(app, data_loader)

    return app

app = create_app()
server = app.server

if __name__ == "__main__":
    port = 8050
    print(f"Starting Plotly Dash Control Tower on http://127.0.0.1:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

