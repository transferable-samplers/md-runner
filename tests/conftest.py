import os

# Create report directory if it doesn't exist
report_dir = os.environ.get("PYTEST_REPORT_DIR", "tests/")
os.makedirs(report_dir, exist_ok=True)
