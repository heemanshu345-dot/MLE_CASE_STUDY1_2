import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    path = os.path.join(args.data_dir, "train_transaction.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, usecols=["TransactionDT", "TransactionAmt", "ProductCD", "isFraud"])

    print(df.info())
    print("\nClass distribution:")
    print(df["isFraud"].value_counts())
    print("\nFraud rate:", df["isFraud"].mean())

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="isFraud")
    plt.title("Fraud vs Legitimate Transactions")
    plt.xlabel("isFraud")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "class_distribution.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df, x="TransactionAmt", hue="isFraud",
        bins=100, element="step", stat="density", common_norm=False
    )
    plt.xlim(0, df["TransactionAmt"].quantile(0.99))
    plt.title("Transaction Amount Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "transaction_amount_distribution.png"), dpi=180)
    plt.close()

if __name__ == "__main__":
    main()
