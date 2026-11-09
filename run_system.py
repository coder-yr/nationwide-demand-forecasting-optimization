"""Master CLI Orchestrator for Nationwide Demand Forecasting & Inventory Optimization.

Provides a unified execution interface for:
- Data Ingestion & Downcasting (--mode ingest)
- Model Training & Evaluation (--mode train)
- Inventory Optimization Solver (--mode optimize)
- Full End-to-End Execution (--mode all)
- Automated Test Suite (--mode test)
"""
import argparse
import sys
import subprocess
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("orchestrator")

def run_ingest(store_id: str, days: int = 365):
    logger.info(f"=== Starting Ingestion Pipeline for Store {store_id} ({days} days) ===")
    from src.data.pipeline import M5DataPipeline
    pipeline = M5DataPipeline(history_days_window=days)
    cal = pipeline.load_calendar()
    output_path = pipeline.process_store(store_id, df_calendar=cal, overwrite=True)
    logger.info(f"=== Ingestion Complete: {output_path} ===")

def run_train(store_id: str):
    logger.info(f"=== Starting Model Training Pipeline for Store {store_id} ===")
    from src.models.train_pipeline import ModelTrainingPipeline
    pipeline = ModelTrainingPipeline(store_id=store_id)
    metrics = pipeline.run()
    logger.info(f"=== Training Complete. Summary: {metrics} ===")

def run_optimize(store_id: str):
    logger.info(f"=== Starting Replenishment Optimization for Store {store_id} ===")
    from src.optimization.replenishment_pipeline import ReplenishmentPipeline
    pipeline = ReplenishmentPipeline(store_id=store_id)
    df = pipeline.run()
    logger.info(f"=== Optimization Complete. {len(df)} orders generated ===")

def run_tests():
    logger.info("=== Running Pytest Suite ===")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-p", "no:dash"])
    sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Nationwide Demand Forecasting & Inventory Optimization CLI")
    parser.add_argument(
        "--mode",
        choices=["ingest", "train", "optimize", "all", "test"],
        default="all",
        help="Pipeline phase to execute"
    )
    parser.add_argument("--store", default="CA_1", help="Store code (e.g. CA_1, TX_2, WI_1)")
    parser.add_argument("--days", type=int, default=365, help="Historical window days for ingestion")

    args = parser.parse_args()

    if args.mode == "test":
        run_tests()
    elif args.mode == "ingest":
        run_ingest(store_id=args.store, days=args.days)
    elif args.mode == "train":
        run_train(store_id=args.store)
    elif args.mode == "optimize":
        run_optimize(store_id=args.store)
    elif args.mode == "all":
        # Check if parquet exists
        parquet_file = Path("data/processed/stores") / f"store_{args.store}.parquet"
        if not parquet_file.exists():
            run_ingest(store_id=args.store, days=args.days)
        run_train(store_id=args.store)
        run_optimize(store_id=args.store)
        logger.info("🎉 Full End-to-End Pipeline Executed Successfully!")

if __name__ == "__main__":
    main()

