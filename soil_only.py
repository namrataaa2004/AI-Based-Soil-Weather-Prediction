import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "nitrogen_ppm",
    "phosphorus_ppm",
    "potassium_ppm",
    "ph",
    "ec_ds_m",
    "organic_carbon_pct",
    "moisture_pct",
]
CAT_FEATURES = ["soil_type"]
TARGET_COL = "label"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/test a soil label classifier.")
    parser.add_argument(
        "--csv",
        default=r"d:\OneDrive\Desktop\cursor\soil_data.csv",
        help="Path to soil_data.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    needed = set(NUMERIC_FEATURES + CAT_FEATURES + [TARGET_COL])
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[NUMERIC_FEATURES + CAT_FEATURES]
    y = df[TARGET_COL]

    # Stratify when it's valid (each class needs >=2 samples).
    stratify = None
    if y.nunique() > 1:
        class_counts = y.value_counts(dropna=False)
        if int(class_counts.min()) >= 2:
            stratify = y
        else:
            rare = class_counts[class_counts < 2].index.tolist()
            print(
                "Note: not using stratify because some classes have <2 samples: "
                f"{rare}"
            )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
        )
    except ValueError as e:
        # Some edge cases can still fail (e.g., too-small test size for stratification).
        print(f"Note: train/test split with stratify failed ({e}); retrying without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=None
        )

    pre = ColumnTransformer(
        transformers=[
            
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    model = RandomForestClassifier(random_state=args.seed)
    clf = Pipeline(steps=[("pre", pre), ("model", model)])
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    # Predict the first row as a quick sanity check.
    one = X.iloc[[0]]
    print("First row prediction:", clf.predict(one)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    