
import pandas as pd

def run_integrity_checks():
    # 1. Load only the training data
    print("--- STARTING INTEGRITY CHECKS ---")
    try:
        df = pd.read_csv('data/processed/train.csv')
    except FileNotFoundError:
        print("Error: data/interim/train.csv not found")
        return

    # 2. Check for empty fields (Missing Values)
    null_count = df.isnull().sum().sum()
    print(f"[Null Check]: Total number of empty fields in the dataset: {null_count}")

    # 3. Check for valid ranges (Months and Weather)
    valid_months = df['mnth'].between(1, 12).all()
    valid_weather = df['weathersit'].between(1, 4).all()
    valid_season = df['season'].between(1, 4).all()
    
    print(f"[Range Check]: Are all months between 1 and 12? {'Yes' if valid_months else 'NO'}")
    print(f"[Range Check]: Are all seasons between 1 and 4? {'Yes' if valid_season else 'NO'}")
    print(f"[Range Check]: Is all weather between 1 and 4? {'Yes' if valid_weather else 'NO'}")

    # 4. Verify which classes actually exist in weathersit
    weather_classes = df['weathersit'].unique()
    weather_classes.sort()
    print(f"[Class Check]: Weather types present in the data: {weather_classes}")

    # 5. Cross-integrity check: workingday vs holiday/weekend
    # weekend is usually weekday 0 (Sunday) and 6 (Saturday) according to standards, 
    # we will check if there is any workingday=1 that is also holiday=1
    logical_errors = df[(df['workingday'] == 1) & (df['holiday'] == 1)]
    print(f"[Logic Check]: Days marked as 'Workingday' but are also 'Holiday': {len(logical_errors)}")

    # 6. Strict category checks
    valid_yr = df['yr'].between(0, 1).all()
    valid_weekday = df['weekday'].between(0, 6).all()
    
    print(f"\n[Category Check]: Is all 'yr' 0 or 1? {'Yes' if valid_yr else 'NO'}")
    print(f"[Category Check]: Is all 'weekday' between 0 and 6? {'Yes' if valid_weekday else 'NO'}")

    # 7. Normalized variables check (0 to 1)
    print("\n[Normalization Check]: Verifying continuous variables...")
    normalized_vars = ['temp', 'atemp', 'hum', 'windspeed']
    for var in normalized_vars:
        valid_range = df[var].between(0, 1).all()
        print(f" - Is '{var}' strictly between 0 and 1? {'Yes' if valid_range else 'NO'}")
        
        # If it is false, we print the extremes to see the severity of the error
        if not valid_range:
            print(f"   -> ALERT in '{var}': The minimum is {df[var].min()} and the maximum is {df[var].max()}")

    print("--- CHECKS FINISHED ---")

if __name__ == "__main__":
    run_integrity_checks()
