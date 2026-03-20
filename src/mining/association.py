import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def create_transaction_matrix(df, text_col):
    keywords = [
        "staff", "location", "room", "clean", "service",
        "breakfast", "wifi", "price", "noise", "bed"
    ]

    data = []
    for review in df[text_col]:
        row = {k: (k in review) for k in keywords}
        data.append(row)

    return pd.DataFrame(data)

def mine_rules(df, text_col):
    df_bin = create_transaction_matrix(df, text_col)

    freq = apriori(df_bin, min_support=0.02, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)

    return rules.sort_values("lift", ascending=False)