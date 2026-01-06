# Project-python-git-linux

## Daily report (cron)

The project includes a separate Python script that generates a daily report from the stored price data:

- Script: `scripts/daily_report.py`
- Input data: `data/prices.csv`
- Output folder: `reports/`
- Output file: `reports/report_YYYY-MM-DD.txt` (UTC date)

### What is in the report?
The report is a simple text file containing:
- number of observations
- open price
- close price
- annualized volatility (simple)
- max drawdown (buy & hold, simple)

### Cron configuration (Linux VM)
On the Linux server, we schedule the report generation every day at 20:00 using `crontab -e`:

```cron
0 20 * * * cd /home/ubuntu/Project-python-git-linux && /usr/bin/python3 scripts/daily_report.py >> reports/cron.log 2>&1
