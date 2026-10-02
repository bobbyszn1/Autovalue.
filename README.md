# AutoValue v2 — Used-Car Valuation Web App

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data workflow

### 1. Carsheet
Use the sidebar to scrape Carsheet by year. This provides structured vehicle
specifications such as make, model, year, trim, drivetrain, engine and MSRP.

### 2. Used-market CSV
Upload a real listing/transaction dataset. The normalizer recognizes common
names for:

- make/model/year/trim
- mileage
- condition
- location
- accident history
- listing date
- price

### 3. Train
The app matches used-market vehicles to Carsheet specifications and trains
a valuation model using the actual market `Price` as the target.

The model can use:
- mileage
- condition
- location
- accident history
- listing date
- vehicle specifications

It evaluates Ridge, Random Forest and Extra Trees and stores the best model.

## Used-market CSV example

```csv
make,model,year,trim,mileage,condition,location,accident_history,listing_date,price
Lexus,RX 350,2017,F Sport,72000,Good,Lagos,No,2026-09-20,32000000
Toyota,Camry,2019,XSE,58000,Excellent,Abuja,No,2026-09-21,28000000
```

Prices can be in the currency used by your market dataset. The app does not
automatically convert currencies.

## Important

Carsheet alone should not be treated as a current used-car transaction
dataset. The production valuation model should be trained on real market
observations. More market observations and better coverage generally improve
the usefulness of the model, but no ML model guarantees a transaction price.
