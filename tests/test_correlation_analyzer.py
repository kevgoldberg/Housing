import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.correlation_analyzer import HousingCorrelationAnalyzer


def test_generate_report_output(tmp_path):
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [4, 3, 2, 1],
        "C": ["x", "y", "x", "y"],
    })

    analyzer = HousingCorrelationAnalyzer(df)
    output_file = tmp_path / "report.json"
    analyzer.generate_comprehensive_report(
        target_var="A",
        save_plots=False,
        output_path=output_file,
    )
    assert output_file.exists()
