import re, time
from io import StringIO
import pandas as pd
import requests

BASE_URL = "https://carsheet.io/all-cars/{year}/"
HEADERS = {"User-Agent": "AutoValueResearch/2.0"}

EXPECTED = [
    "Make","Model","Year","Trim","MSRP","Invoice Price","Used/New Price",
    "Body Size","Body Style","Cylinders","Engine Aspiration","Drivetrain",
    "Transmission","Horsepower","Torque","Highway Fuel Economy"
]

def number(v):
    if pd.isna(v): return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None

def normalize(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(x) for x in c if str(x) != "nan").strip() for c in df.columns]
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        for e in EXPECTED:
            if c.lower() == e.lower():
                rename[c] = e
    df = df.rename(columns=rename)
    if "Make" not in df.columns or "Model" not in df.columns:
        return pd.DataFrame()
    for c in EXPECTED:
        if c not in df.columns: df[c] = None
    df = df[EXPECTED]
    df = df[df["Make"].notna() & df["Model"].notna()]
    df = df[~df["Make"].astype(str).str.lower().isin(["make","nan",""])]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    for c in ["MSRP","Invoice Price","Used/New Price","Horsepower","Torque","Highway Fuel Economy"]:
        df[c] = df[c].apply(number)
    return df

def scrape_year(year, delay=1.0):
    r = requests.get(BASE_URL.format(year=year), headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    frames = [normalize(t) for t in tables]
    frames = [x for x in frames if not x.empty]
    time.sleep(max(delay, 0))
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()

def scrape_range(start_year, end_year, delay=1.0, progress=None):
    frames, errors = [], []
    years = list(range(start_year, end_year + 1))
    for i, year in enumerate(years, 1):
        try:
            df = scrape_year(year, delay)
            if not df.empty: frames.append(df)
        except Exception as e:
            errors.append((year, str(e)))
        if progress: progress(i / len(years), year)
    if not frames: raise RuntimeError("No Carsheet data collected.")
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates().reset_index(drop=True), errors
