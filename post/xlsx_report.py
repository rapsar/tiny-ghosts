#!/usr/bin/env python3
"""
xlsx_report.py

Generates an Excel report from a folder of images by extracting EXIF date/time metadata.
For each image, it compiles filename, date, and time into a “metadata” sheet (sorted chronologically),
counts daily photo totals, computes earliest and latest photo times per day,
and embeds line charts of these metrics into a “counts” sheet.

Usage:
    Configure `input_folder`, `output_folder`, and `report_name` in the __main__ block.
    Run the script to produce an .xlsx file readable by Excel, LibreOffice Calc, Apple Numbers, etc.

Dependencies:
    - Python 3
    - pandas
    - Pillow (PIL)
    - XlsxWriter
"""


import os
import sys
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
import re





def extract_datetime(path):
    """
    Try to pull DateTimeOriginal (or fallback DateTime) EXIF tag
    Returns a datetime.datetime or None
    """
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        # map tag names to keys
        tag_map = {ExifTags.TAGS[k]: k for k in exif.keys() if k in ExifTags.TAGS}
        for tag_name in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
            if tag_name in tag_map:
                dt_str = exif[tag_map[tag_name]]
                # EXIF format is "YYYY:MM:DD HH:MM:SS"
                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


# --- New helper function: extract_temperature
def extract_temperature(path):
    """
    Extracts temperature from the MakerNote EXIF tag by finding the typoed 'tempture' pattern
    and reading the two characters following it. Returns an int or None.
    """
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        # Find MakerNote tag key
        maker_tag = next((k for k, v in ExifTags.TAGS.items() if v == "MakerNote"), None)
        if maker_tag is None or maker_tag not in exif:
            return None
        maker = exif[maker_tag]
        if isinstance(maker, bytes):
            maker = maker.decode(errors="ignore")
        # Look for the typoed pattern
        pattern = "tempture"
        idx = maker.find(pattern)
        if idx != -1 and len(maker) >= idx + 11:
            # MATLAB uses k+9:k+10 in 1-based indexing; Python slice idx+9:idx+11 captures those two chars
            temp_str = maker[idx + 9 : idx + 11]
            try:
                return int(temp_str)
            except ValueError:
                return None
    except Exception:
        pass
    return None

def main(input_folder, output_xlsx, start_date=None, end_date=None):
    records = []
    for fname in os.listdir(input_folder):
        full = os.path.join(input_folder, fname)
        if not os.path.isfile(full):
            continue
        dt = extract_datetime(full)
        if dt:
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
        else:
            date_str = None
            time_str = None
        temp = extract_temperature(full)
        records.append({
            "filename": fname,
            "datetime": dt,
            "date": date_str,
            "time": time_str,
            "temperature": temp
        })

    # Create DataFrame
    df = pd.DataFrame(records)
    # Only keep filename, date, time, and temperature, then sort by date & time
    df = df[["filename", "date", "time", "temperature"]]
    df["temperature_f"] = df["temperature"].apply(
        lambda c: c * 9/5 + 32 if pd.notna(c) else None
    )
    df = df[["filename", "date", "time", "temperature", "temperature_f"]]
    df.sort_values(by=["date", "time"], inplace=True)

    # Count photos per day (skip None)
    counts = (
        df[df["date"].notna()]
        .groupby("date")
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )

    # Compute earliest and latest photo times per day
    # Convert 'time' strings to timestamps
    df_times = df[df["time"].notna()].copy()
    df_times["time_dt"] = pd.to_datetime(df_times["time"], format="%H:%M:%S")
    extremes = (
        df_times
        .groupby("date")["time_dt"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "earliest", "max": "latest"})
    )
    # Convert earliest and latest to Excel time fractions for plotting
    extremes["earliest"] = (
        extremes["earliest"].dt.hour / 24
        + extremes["earliest"].dt.minute / (24 * 60)
        + extremes["earliest"].dt.second / (24 * 3600)
    )
    extremes["latest"] = (
        extremes["latest"].dt.hour / 24
        + extremes["latest"].dt.minute / (24 * 60)
        + extremes["latest"].dt.second / (24 * 3600)
    )
    # Merge extremes into counts DataFrame
    counts = counts.merge(extremes, on="date", how="left")

    # If a start_date or end_date is provided, pad missing days with zero counts and NaN times
    try:
        if start_date or end_date:
            counts["date"] = pd.to_datetime(counts["date"])
            start = start_date or counts["date"].min()
            end = end_date or counts["date"].max()
            full_idx = pd.date_range(start=start, end=end, freq="D")
            full_df = pd.DataFrame({"date": full_idx})
            counts = full_df.merge(counts, on="date", how="left")
            counts["count"] = counts["count"].fillna(0).astype(int)
            # earliest/latest remain NaN for days without photos
    except NameError:
        pass

    # Convert date column back to "YYYY-MM-DD" strings for counts sheet
    counts["date"] = counts["date"].dt.strftime("%Y-%m-%d")
    # Compute 7-day moving average of daily photo counts
    counts["ma7"] = counts["count"].rolling(window=7, min_periods=1).mean()

    # Compute average temperature per day and merge into counts
    temp_avg = (
        df[df["temperature"].notna()]
        .groupby("date")["temperature"]
        .mean()
        .reset_index(name="avg_temp")
    )
    counts = counts.merge(temp_avg, on="date", how="left")

    # Write to Excel with XlsxWriter engine and embed a chart
    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="metadata")
        # Make filenames in metadata sheet clickable via HYPERLINK formula
        meta_ws = writer.sheets["metadata"]
        for row_idx, fname in enumerate(df["filename"], start=1):
            file_path = os.path.abspath(os.path.join(input_folder, fname))
            # Build a hyperlink formula that falls back to the filename if the link is invalid
            link = file_path
            formula = f'=HYPERLINK("{link}", "{fname}")'
            # formula = f'=IFERROR(HYPERLINK("{link}", "{fname}"), "{fname}")'
            meta_ws.write_formula(row_idx, 0, formula)
        counts.to_excel(writer, index=False, sheet_name="counts")

        workbook  = writer.book
        worksheet = writer.sheets["counts"]

        # Format earliest/latest columns C and D as times
        time_fmt = workbook.add_format({"num_format": "hh:mm:ss"})
        worksheet.set_column("C:D", 12, time_fmt)

        # Build a line chart
        chart = workbook.add_chart({"type": "line"})
        max_row = len(counts) + 1
        chart.add_series({
            "name":       "Photos per day",
            "categories": ["counts", 1, 0, max_row, 0],
            "values":     ["counts", 1, 1, max_row, 1],
        })
        chart.add_series({
            "name":       "7-Day Moving Avg",
            "categories": ["counts", 1, 0, max_row, 0],
            "values":     ["counts", 1, 4, max_row, 4],
        })
        chart.set_title ({"name": "Pictures per Day"})
        chart.set_x_axis({"name": "Date"})
        chart.set_y_axis({"name": "Count"})

        # Insert the chart next to the data
        worksheet.insert_chart("G2", chart)

        # Build a line chart for earliest and latest photo times
        time_chart = workbook.add_chart({"type": "line"})
        # max_row is already set above (len(counts) + 1)
        # Earliest time series (column C)
        time_chart.add_series({
            "name":       "Earliest Photo Time",
            "categories": ["counts", 1, 0, max_row, 0],
            "values":     ["counts", 1, 2, max_row, 2],
        })
        # Latest time series (column D)
        time_chart.add_series({
            "name":       "Latest Photo Time",
            "categories": ["counts", 1, 0, max_row, 0],
            "values":     ["counts", 1, 3, max_row, 3],
        })
        time_chart.set_title({"name": "Earliest and Latest Photo Times"})
        time_chart.set_x_axis({"name": "Date"})
        time_chart.set_y_axis({"name": "Time of Day", "num_format": "hh:mm:ss"})
        # Insert the time chart below the first chart
        worksheet.insert_chart("G20", time_chart)

        # Compute temperature range for secondary axis scaling
        temp_min = counts["avg_temp"].min()
        temp_max = counts["avg_temp"].max()
        # Build a line chart for Photos per Day and Avg Temperature
        temp_chart = workbook.add_chart({"type": "line"})
        temp_chart.add_series({
            "name":       "Photos per Day",
            "categories": ["counts", 1, 0, max_row, 0],
            "values":     ["counts", 1, 1, max_row, 1],
        })
        temp_chart.add_series({
            "name":       "Average Temperature",
            "categories": ["counts", 1, 0, max_row, 0],
            "values":     ["counts", 1, 5, max_row, 5],
            "y2_axis":    True,
            "line":       {"color": "orange"},
        })
        temp_chart.set_title({"name": "Photos per Day and Avg Temperature"})
        temp_chart.set_x_axis({"name": "Date"})
        temp_chart.set_y_axis({"name": "Photos per Day"})
        temp_chart.set_y2_axis({"name": "Avg Temperature", "min": temp_min * 0.9, "max": temp_max * 1.1})
        # Insert the combined line chart
        worksheet.insert_chart("G38", temp_chart)

        # # Compute axis ranges for scatter plot
        # scatter_x_min = temp_min
        # scatter_x_max = temp_max
        # count_min = counts["count"].min()
        # count_max = counts["count"].max()
        # # Build a scatter plot for Photos per Day vs Avg Temperature
        # scatter_chart = workbook.add_chart({"type": "scatter", "subtype": "marker"})
        # scatter_chart.add_series({
        #     "name":       "Photos vs Temperature",
        #     "categories": ["counts", 1, 5, max_row, 5],
        #     "values":     ["counts", 1, 1, max_row, 1],
        # })
        # scatter_chart.set_title({"name": "Photos per Day vs Avg Temperature"})
        # scatter_chart.set_x_axis({"name": "Avg Temperature", "min": scatter_x_min * 0.9, "max": scatter_x_max * 1.1})
        # scatter_chart.set_y_axis({"name": "Photos per Day", "min": count_min * 0.9, "max": count_max * 1.1})
        # # Insert the scatter plot below the line chart
        # worksheet.insert_chart("F56", scatter_chart)

    print(f"✔️  Written report to {output_xlsx}")



if __name__ == "__main__":
    # === Configuration: set your paths here ===
    input_folder  = "/Users/rss367/Desktop/2024bww/Muleshoe/results/UpperWildcat/_flashes"
    output_folder = "/Users/rss367/Desktop/2024bww/Muleshoe/results/UpperWildcat/"
    report_name   = "flashes_UpperWildcat_2024_0621_0909.xlsx"
    start_date    = "2024-06-21"  # e.g. "2024-06-01" to include days before first photo; set None otherwise
    end_date      = "2024-09-09"  # Optionally set an end date, e.g. "2024-08-14"
    # =========================================
    # add temperature data

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Build the full output path and run
    output_path = os.path.join(output_folder, report_name)
    # Confirm before overwriting existing report
    if os.path.exists(output_path):
        resp = input(f"File '{output_path}' already exists. Overwrite? [y/N]: ")
        if resp.strip().lower() not in ('y', 'yes'):
            print("Aborting without overwrite.")
            sys.exit(0)
    main(input_folder, output_path, start_date, end_date)