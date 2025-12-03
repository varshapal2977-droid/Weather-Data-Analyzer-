#!/usr/bin/env python3
"""
weather_analysis.py

Usage:
    python weather_analysis.py --input path/to/weather.csv --output outputs/

This script:
 - Loads a weather CSV (detects date column)
 - Cleans data (date parsing, fill/drop NaNs)
 - Computes daily/monthly/yearly stats
 - Produces and saves plots:
     - line chart: daily temperature
     - bar chart: monthly rainfall totals
     - scatter: humidity vs temperature
     - combined figure with two subplots
 - Saves cleaned CSV and a Markdown summary report.

Requires: pandas, numpy, matplotlib
Install: pip install pandas numpy matplotlib
"""

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True  # better layout by default

# ---------- Utilities ----------

def find_date_column(df):
    """Try to find a date-like column name in the dataframe."""
    candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if candidates:
        return candidates[0]
    # fallback: try to parse each column until one parses
    for c in df.columns:
        try:
            pd.to_datetime(df[c].dropna().iloc[:10])
            return c
        except Exception:
            continue
    raise ValueError("No date/time column found — please include a 'date' column in the CSV.")

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

# ---------- Core pipeline functions ----------

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df, date_col=None, numeric_fill_strategy='ffill'):
    # detect date column if not provided
    if date_col is None:
        date_col = find_date_column(df)
    # convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # drop rows without a valid date
    df = df.dropna(subset=[date_col]).copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # set index
    df = df.set_index(date_col).sort_index()
    # handle numeric columns: convert where possible
    for col in df.columns:
        if df[col].dtype == object:
            # try to coerce numbers (remove commas, percent signs if any)
            cleaned = df[col].astype(str).str.replace(',', '').str.replace('%', '')
            coerced = pd.to_numeric(cleaned, errors='coerce')
            if coerced.notna().sum() > 0:
                df[col] = coerced
    # fill numeric NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_fill_strategy == 'ffill':
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    elif numeric_fill_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:
        df = df.dropna(subset=numeric_cols, how='all')
    return df

def compute_statistics(df, value_cols=None):
    """Compute daily, monthly, yearly statistics for given numeric columns."""
    if value_cols is None:
        value_cols = list(df.select_dtypes(include=[np.number]).columns)
    stats = {}
    # daily: if df.index is datetime, group by date
    daily = df[value_cols].resample('D').agg(['mean', 'min', 'max', 'std', 'sum'])
    monthly = df[value_cols].resample('M').agg(['mean', 'min', 'max', 'std', 'sum'])
    yearly = df[value_cols].resample('Y').agg(['mean', 'min', 'max', 'std', 'sum'])
    # also quick numeric summaries using numpy
    quick_summary = {}
    for col in value_cols:
        arr = df[col].dropna().to_numpy()
        quick_summary[col] = {
            'mean': float(np.nanmean(arr)),
            'min': float(np.nanmin(arr)) if arr.size else None,
            'max': float(np.nanmax(arr)) if arr.size else None,
            'std': float(np.nanstd(arr, ddof=1)) if arr.size else None,
            'count': int(np.count_nonzero(~np.isnan(arr)))
        }
    stats['daily'] = daily
    stats['monthly'] = monthly
    stats['yearly'] = yearly
    stats['quick'] = quick_summary
    return stats

# ---------- Plotting functions ----------

def plot_daily_temperature(df, temp_col, outpath):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df.index, df[temp_col], label=temp_col)
    ax.set_title('Daily Temperature Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel(temp_col)
    ax.grid(True, linestyle='--', linewidth=0.4)
    ax.legend()
    fig.savefig(outpath)
    plt.close(fig)

def plot_monthly_rainfall(df, rain_col, outpath):
    monthly = df[rain_col].resample('M').sum()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(monthly.index.to_pydatetime(), monthly.values)
    ax.set_title('Monthly Rainfall Totals')
    ax.set_xlabel('Month')
    ax.set_ylabel(f'Total {rain_col}')
    ax.grid(axis='y', linestyle='--', linewidth=0.4)
    fig.savefig(outpath)
    plt.close(fig)

def plot_humidity_vs_temp(df, humidity_col, temp_col, outpath):
    x = df[temp_col].dropna()
    y = df[humidity_col].reindex(x.index)
    mask = x.notna() & y.notna()
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(x[mask], y[mask], alpha=0.6)
    ax.set_xlabel(temp_col)
    ax.set_ylabel(humidity_col)
    ax.set_title(f'{humidity_col} vs {temp_col}')
    ax.grid(True, linestyle='--', linewidth=0.4)
    fig.savefig(outpath)
    plt.close(fig)

def plot_combined(df, temp_col, rain_col, outpath):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    ax1.plot(df.index, df[temp_col])
    ax1.set_title('Daily Temperature')
    ax1.set_ylabel(temp_col)
    ax1.grid(True, linestyle='--', linewidth=0.4)

    monthly_rain = df[rain_col].resample('M').sum()
    ax2.bar(monthly_rain.index.to_pydatetime(), monthly_rain.values)
    ax2.set_title('Monthly Rainfall Totals')
    ax2.set_ylabel(rain_col)
    ax2.grid(axis='y', linestyle='--', linewidth=0.4)

    fig.suptitle('Combined: Temp (daily) & Rainfall (monthly)')
    fig.savefig(outpath)
    plt.close(fig)

# ---------- Export / report ----------

def save_cleaned_csv(df, outpath):
    df.reset_index().to_csv(outpath, index=False)

def write_summary_markdown(stats, paths, outpath):
    """
    stats: output of compute_statistics
    paths: dict with keys 'temp_plot','rain_plot','scatter','combined','cleaned_csv'
    """
    lines = []
    lines.append(f"# Weather Data Analysis Summary\n")
    lines.append(f"Generated on: {datetime.utcnow().isoformat()} UTC\n")
    lines.append("## Quick numeric summary\n")
    for col, s in stats['quick'].items():
        lines.append(f"### {col}")
        lines.append(f"- mean: {s['mean']}")
        lines.append(f"- min: {s['min']}")
        lines.append(f"- max: {s['max']}")
        lines.append(f"- std: {s['std']}")
        lines.append(f"- count: {s['count']}\n")

    lines.append("## Output files\n")
    for k,v in paths.items():
        lines.append(f"- **{k}**: `{v}`")

    lines.append("\n## Short observations (auto-generated)\n")
    # trivial observations: top/bottom for first numeric column
    try:
        first = next(iter(stats['quick']))
        s = stats['quick'][first]
        lines.append(f"- The column **{first}** has mean {s['mean']}, min {s['min']}, max {s['max']}.")
    except StopIteration:
        lines.append("- No numeric columns found for auto-summary.")
    # write to file
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ---------- Main CLI ----------

def main(args):
    safe_mkdir(args.output)
    print("Loading data...")
    df = load_data(args.input)

    print("Cleaning data...")
    df_clean = clean_data(df, date_col=args.date_column, numeric_fill_strategy=args.fill_strategy)

    # heuristics for column names
    col_candidates = {c.lower():c for c in df_clean.columns}
    # try common names
    temp_col = None
    for name in ['temp','temperature','t_avg','tmax','tmin']:
        if name in col_candidates:
            temp_col = col_candidates[name]; break
    # humidity
    humidity_col = None
    for name in ['humidity','rh','relative_humidity']:
        if name in col_candidates:
            humidity_col = col_candidates[name]; break
    # rainfall
    rain_col = None
    for name in ['rain','rainfall','precipitation','precip']:
        if name in col_candidates:
            rain_col = col_candidates[name]; break

    # pick first numeric column fallback
    numeric_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
    if temp_col is None and numeric_cols:
        temp_col = numeric_cols[0]
    if humidity_col is None and len(numeric_cols) > 1:
        humidity_col = numeric_cols[1] if numeric_cols[1] != temp_col else numeric_cols[0]
    if rain_col is None and len(numeric_cols) > 2:
        rain_col = numeric_cols[2]

    print("Detected columns: temp:", temp_col, "humidity:", humidity_col, "rain:", rain_col)
    # compute stats
    stats = compute_statistics(df_clean, value_cols=numeric_cols)

    # produce plots (only if the required columns exist)
    outputs = {}
    if temp_col:
        temp_plot = os.path.join(args.output, 'daily_temperature.png')
        plot_daily_temperature(df_clean, temp_col, temp_plot)
        outputs['temp_plot'] = temp_plot
    if rain_col:
        rain_plot = os.path.join(args.output, 'monthly_rainfall.png')
        plot_monthly_rainfall(df_clean, rain_col, rain_plot)
        outputs['rain_plot'] = rain_plot
    if humidity_col and temp_col:
        scatter_plot = os.path.join(args.output, 'humidity_vs_temp.png')
        plot_humidity_vs_temp(df_clean, humidity_col, temp_col, scatter_plot)
        outputs['scatter'] = scatter_plot
    if temp_col and rain_col:
        combined = os.path.join(args.output, 'combined_temp_rain.png')
        plot_combined(df_clean, temp_col, rain_col, combined)
        outputs['combined'] = combined

    # save cleaned CSV
    cleaned_csv = os.path.join(args.output, 'cleaned_weather.csv')
    save_cleaned_csv(df_clean, cleaned_csv)
    outputs['cleaned_csv'] = cleaned_csv

    # save summary
    summary_md = os.path.join(args.output, 'summary_report.md')
    write_summary_markdown(stats, outputs, summary_md)

    print("Done. Outputs written to:", args.output)
    for k,v in outputs.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather data analysis pipeline")
    parser.add_argument('--input', required=True, help='Path to weather CSV file')
    parser.add_argument('--output', required=True, help='Output directory for cleaned CSV, plots, and summary')
    parser.add_argument('--date-column', default=None, help='(optional) name of the date column in CSV')
    parser.add_argument('--fill-strategy', choices=['ffill','mean','drop'], default='ffill',
                        help='How to fill numeric missing values')
    args = parser.parse_args()
    main(args)
