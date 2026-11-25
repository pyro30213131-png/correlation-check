from portfolio_analyzer.core.data_loader import load_from_csv
from portfolio_analyzer.core.analyzer import PortfolioAnalyzer

data     = load_from_csv("prices.csv")   # your CSV here
analyzer = PortfolioAnalyzer(data.returns, data.tickers)
report   = analyzer.full_report()
print(report.summary())