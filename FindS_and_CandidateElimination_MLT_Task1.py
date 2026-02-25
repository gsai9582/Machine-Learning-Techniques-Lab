# =========================================
# MLT LAB – FIND-S & CANDIDATE ELIMINATION
# =========================================

# ---------- FIND-S ALGORITHM ----------
def find_s_algorithm(positive_examples):
    hypothesis = ['0'] * len(positive_examples[0])
    print("\n--- FIND-S ALGORITHM ---\n")

    for idx, example in enumerate(positive_examples, start=1):
        for i in range(len(example)):
            if hypothesis[i] == '0':
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i]:
                hypothesis[i] = '?'
        print(f"After Example {idx} → Hypothesis: {hypothesis}")

    return hypothesis


# ---------- CANDIDATE ELIMINATION ----------
def candidate_elimination(positive_examples, negative_examples):
    print("\n--- CANDIDATE ELIMINATION ALGORITHM ---\n")

    specific_h = positive_examples[0].copy()
    general_h = [['?' for _ in range(len(specific_h))]]

    # Process positive examples
    for example in positive_examples:
        for i in range(len(specific_h)):
            if example[i] != specific_h[i]:
                specific_h[i] = '?'
                for g in general_h:
                    g[i] = '?'

    # Process negative examples
    for example in negative_examples:
        for i in range(len(specific_h)):
            if example[i] != specific_h[i] and specific_h[i] != '?':
                new_h = ['?' for _ in range(len(specific_h))]
                new_h[i] = specific_h[i]
                general_h.append(new_h)

    # Remove fully generic hypothesis
    general_h = [g for g in general_h if g != ['?' for _ in range(len(specific_h))]]

    print("Specific Hypothesis:", specific_h)
    print("General Hypotheses:")
    for g in general_h:
        print(g)

    return specific_h, general_h


# ---------- TRAINING DATA ----------
positive_examples = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
]

negative_examples = [
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
    ['Sunny', 'Warm', 'Normal', 'Weak', 'Cool', 'Same']
]

# ---------- EXECUTION ----------
final_find_s = find_s_algorithm(positive_examples)
specific, general = candidate_elimination(positive_examples, negative_examples)

print("\n--- FINAL OUTPUT ---")
print("Final FIND-S Hypothesis:", final_find_s)
print("Final Specific Hypothesis:", specific)
print("Final General Hypotheses:", general)
