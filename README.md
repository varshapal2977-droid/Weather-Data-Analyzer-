# Weather-Data-Analyzer-
<img width="491" height="289" alt="2025-12-03" src="https://github.com/user-attachments/assets/220a8f15-feee-444b-8c78-cc6a6cb066c1" />

🌦️ Weather Analyzer — Python Weather Data Analysis Tool

# A complete weather-data analysis pipeline that:

 Loads and cleans a weather CSV
 Detects the date column automatically
 Handles missing values
 Produces daily, monthly, yearly statistics
 Generates multiple graphs
 Exports a cleaned dataset
 Creates an auto-generated summary report

# Perfect for data analysis, projects, or school assignments.

 👉👉👉Features
 1. Intelligent CSV Loader

Automatically detects date/time column

Converts messy columns to numeric values

Handles commas, percent signs, and mixed data

 2. Cleaning & Preprocessing

Converts date column to proper datetime

Removes invalid rows

Fills missing numeric values using:

forward fill (ffill)

mean fill (mean)

or drops missing rows (drop)

 3. Statistics Computation

Creates daily, monthly, yearly stats using:

mean

min

max

standard deviation

sum

Also builds a quick numerical summary using NumPy.

 4. Visualization (Matplotlib)

Generates and saves:

📈 Daily temperature line chart

🌧️ Monthly rainfall bar chart

💧 Humidity vs Temperature scatter plot

📊 Combined chart (temperature + rainfall)

All images saved as .png files.

 5. Export Outputs

The script saves:

File	Description
cleaned_weather.csv	Cleaned dataset
daily_temperature.png	Line graph
monthly_rainfall.png	Bar graph
humidity_vs_temp.png	Scatter plot
combined_temp_rain.png	Multi-subplot figure
summary_report.md	Auto summary in Markdown
📁 Output Folder Structure
output/
│
├── cleaned_weather.csv
├── daily_temperature.png
├── monthly_rainfall.png
├── humidity_vs_temp.png
├── combined_temp_rain.png
└── summary_report.md

 Requirements

Install dependencies using:

pip install pandas numpy matplotlib

 # Usage

Run the script from the terminal:

python weather_analysis.py --input path/to/weather.csv --output results/

Optional arguments:
Flag	Description
--date-column COLUMN	Specify the date column if not auto-detected
--fill-strategy STRATEGY	ffill, mean, or drop
Example:
python weather_analysis.py --input data/weather.csv --output output --fill-strategy mean

# How It Works (Flow)

Load the CSV

Detect / parse the date column

Clean numeric fields

Fill or drop missing values

Compute statistics

Generate all graphs

Save cleaned CSV

Write Markdown summary

# Summary Report

The tool generates a Markdown file containing:

Mean, min, max, std for each numeric column

Paths of all generated files

Auto-generated observations

Example:

### temperature
- mean: 22.4
- min: 16.1
- max: 35.2
- std: 5.3
- count: 360

### Technologies Used

Python 3

NumPy

Pandas

Matplotlib

Argparse

OS / File Handling

 👩‍🎓Author

 ### Weather Data Analyzer created by [Varsha Pal]
Feel free to modify, improve, or extend!
