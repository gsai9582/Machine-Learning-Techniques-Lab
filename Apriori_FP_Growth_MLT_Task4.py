# ======================================================
# MLT LAB – TASK 4
# APRIORI & FP-GROWTH (FINAL WORKING CODE)
# ======================================================

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pyfpgrowth

# ---------- DATASET ----------
buying_books_data = [
    ['Book1', 'Book2', 'Book3'],
    ['Book2', 'Book3', 'Book4'],
    ['Book1', 'Book3', 'Book5'],
    ['Book2', 'Book4', 'Book5']
]

# ======================================================
# APRIORI ALGORITHM
# ======================================================
print("\n========== APRIORI ALGORITHM ==========\n")

te = TransactionEncoder()
te_array = te.fit(buying_books_data).transform(buying_books_data)
df = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

rules_apriori = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("Frequent Itemsets (Apriori):")
print(frequent_itemsets)

print("\nAssociation Rules (Apriori):")
print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ======================================================
# FP-GROWTH ALGORITHM
# ======================================================
print("\n========== FP-GROWTH ALGORITHM ==========\n")

transactions = [tuple(t) for t in buying_books_data]

# ✅ FIXED LINE HERE
patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)

rules_fp = pyfpgrowth.generate_association_rules(patterns, 0.5)

print("Frequent Itemsets (FP-Growth):")
print(patterns)

print("\nAssociation Rules (FP-Growth):")
print(rules_fp)
