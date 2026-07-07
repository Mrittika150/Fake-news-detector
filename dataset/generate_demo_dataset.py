"""
generate_demo_dataset.py
-------------------------
Creates a synthetic demo dataset (news.csv) so the project runs end-to-end
out of the box. This is ONLY for demoing the pipeline.

For a portfolio-quality model, replace dataset/news.csv with a real dataset such as:
  Kaggle "Fake and Real News Dataset" by Clement Bisaillon
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

That dataset has two files (Fake.csv, True.csv). To convert them into the
format this project expects, run:

    import pandas as pd
    fake = pd.read_csv("Fake.csv"); fake["label"] = "FAKE"
    real = pd.read_csv("True.csv"); real["label"] = "REAL"
    df = pd.concat([fake, real])[["title", "text", "label"]]
    df["text"] = df["title"] + " " + df["text"]
    df[["text", "label"]].sample(frac=1).to_csv("dataset/news.csv", index=False)
"""

import random
import pandas as pd

random.seed(42)

# ---- Building blocks for synthetic REAL (measured, factual) news ----
real_subjects = [
    "The Ministry of Finance", "Local health officials", "The city council",
    "Researchers at the university", "The central bank", "State education authorities",
    "The transportation department", "Company executives", "Federal regulators",
    "The World Health Organization", "The national weather service", "Election officials",
]
real_actions = [
    "announced a new policy regarding", "released a report on", "confirmed updated guidelines for",
    "held a press conference about", "published quarterly data on", "approved funding for",
    "began an investigation into", "issued a statement clarifying", "scheduled a public hearing on",
    "provided an update on",
]
real_topics = [
    "infrastructure spending", "public health measures", "school curriculum changes",
    "interest rate adjustments", "renewable energy projects", "road safety regulations",
    "tax policy reforms", "vaccine distribution", "employment statistics", "housing affordability",
    "climate adaptation plans", "flood recovery efforts",
]
real_tail = [
    "Officials said further details would be shared in the coming weeks.",
    "The report is based on data collected over the past fiscal year.",
    "A follow-up briefing is expected next month.",
    "Analysts noted the change aligns with previous projections.",
    "The department said it welcomes public feedback on the matter.",
    "Independent experts were consulted before the decision was finalized.",
]

# ---- Building blocks for synthetic FAKE (sensational, unverified) news ----
fake_openers = [
    "SHOCKING:", "You won't believe what", "BREAKING (exclusive):", "Doctors HATE this:",
    "Leaked documents reveal", "Secret report exposes", "They don't want you to know:",
    "Anonymous insider claims", "Viral post reveals", "EXPOSED:",
]
fake_subjects = [
    "the government", "Big Pharma", "a secret society", "mainstream media",
    "a rogue scientist", "aliens", "a billionaire cabal", "the deep state",
    "a mysterious whistleblower", "a hidden agency",
]
fake_claims = [
    "is hiding a miracle cure that costs nothing", "has been secretly controlling the weather",
    "is planning to microchip every citizen next year", "faked the entire election overnight",
    "discovered a way to reverse aging using tap water", "is covering up contact with extraterrestrials",
    "is bribing scientists to hide the truth about vaccines", "plans to shut down the internet forever",
    "has proof the moon landing was staged in a basement", "is using 5G towers to control your thoughts",
]
fake_tail = [
    "Share this before it gets DELETED!!!",
    "Mainstream media refuses to report this - wake up!",
    "This changes EVERYTHING and nobody is talking about it.",
    "Sources say this will shock the entire world.",
    "Forward to everyone you know before it's too late!",
    "The truth is finally coming out and it's terrifying.",
]

def make_real(n):
    rows = []
    for _ in range(n):
        text = (f"{random.choice(real_subjects)} {random.choice(real_actions)} "
                f"{random.choice(real_topics)}. {random.choice(real_tail)}")
        rows.append({"text": text, "label": "REAL"})
    return rows

def make_fake(n):
    rows = []
    for _ in range(n):
        text = (f"{random.choice(fake_openers)} {random.choice(fake_subjects)} "
                f"{random.choice(fake_claims)}. {random.choice(fake_tail)}")
        rows.append({"text": text, "label": "FAKE"})
    return rows

N_PER_CLASS = 400
data = make_real(N_PER_CLASS) + make_fake(N_PER_CLASS)
df = pd.DataFrame(data).drop_duplicates(subset="text").sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("dataset/news.csv", index=False)
print(f"Wrote {len(df)} rows to dataset/news.csv")
print(df["label"].value_counts())
