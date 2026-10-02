"""
Optional used-market data ingestion.

The app accepts a CSV of real used-car listings/transactions and normalizes
common column names. This lets Carsheet provide vehicle specifications while
real-market data supplies mileage, condition, location, accident history and
listing date.

Expected concepts:
make, model, year, trim, mileage, condition, location, accident_history,
listing_date, price
"""

import re
import pandas as pd

ALIASES = {
    "Make": ["make","brand","manufacturer"],
    "Model": ["model","vehicle_model"],
    "Year": ["year","model_year","vehicle_year"],
    "Trim": ["trim","variant","grade"],
    "Mileage": ["mileage","miles","odometer","odometer_reading","km","kilometers","kilometres"],
    "Condition": ["condition","vehicle_condition"],
    "Location": ["location","city","state","region","market"],
    "Accident History": ["accident_history","accident","accidents","damage_history"],
    "Listing Date": ["listing_date","date_listed","listed_date","date","created_at"],
    "Price": ["price","selling_price","sale_price","asking_price","used_price","market_price"],
}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def normalize_market_csv(df):
    lookup = {norm(c): c for c in df.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        for a in aliases:
            if norm(a) in lookup:
                rename[lookup[norm(a)]] = canonical
                break
    out = df.rename(columns=rename).copy()

    for c in ALIASES:
        if c not in out.columns:
            out[c] = None

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["Mileage"] = pd.to_numeric(
        out["Mileage"].astype(str).str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce"
    )
    out["Price"] = pd.to_numeric(
        out["Price"].astype(str).str.replace(r"[$,]", "", regex=True)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce"
    )
    out["Listing Date"] = pd.to_datetime(out["Listing Date"], errors="coerce")
    return out

def build_training_table(carsheet, market):
    """
    Merge real market observations onto Carsheet specs by Make/Model/Year/Trim.
    Exact trim matches are preferred; missing trim is allowed as a fallback.
    """
    cs = carsheet.copy()
    mk = normalize_market_csv(market)

    keys = ["Make","Model","Year"]
    cs["__key"] = cs[keys].astype(str).apply(lambda r: "|".join(r.str.lower().str.strip()), axis=1)
    mk["__key"] = mk[keys].astype(str).apply(lambda r: "|".join(r.str.lower().str.strip()), axis=1)

    # Prefer market rows with actual price.
    mk = mk[mk["Price"].notna() & (mk["Price"] > 0)].copy()
    merged = mk.merge(
        cs.drop_duplicates("__key"),
        on="__key",
        how="left",
        suffixes=("_market","")
    )

    # Market values are the authoritative used-price fields.
    merged["Make"] = merged["Make_market"].fillna(merged["Make"])
    merged["Model"] = merged["Model_market"].fillna(merged["Model"])
    merged["Year"] = merged["Year_market"].fillna(merged["Year"])
    merged["Trim"] = merged["Trim_market"].fillna(merged["Trim"])

    # Preserve market-specific features.
    for c in ["Mileage","Condition","Location","Accident History","Listing Date"]:
        if c not in merged.columns: merged[c] = None

    keep = [
        "Make","Model","Year","Trim","Mileage","Condition","Location",
        "Accident History","Listing Date","Price",
        "MSRP","Invoice Price","Body Size","Body Style","Cylinders",
        "Engine Aspiration","Drivetrain","Transmission","Horsepower",
        "Torque","Highway Fuel Economy"
    ]
    return merged[[c for c in keep if c in merged.columns]].drop_duplicates()

def synthetic_market_template():
    return pd.DataFrame(columns=[
        "Make","Model","Year","Trim","Mileage","Condition","Location",
        "Accident History","Listing Date","Price"
    ])
