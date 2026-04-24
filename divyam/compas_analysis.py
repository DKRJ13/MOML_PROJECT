# %%
!pip install lifelines

# %% [markdown]
# # Loading the Data
# We select fields for severity of charge, number of priors, demographics, age, sex, compas scores, and whether each person was accused of a crime within two years.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Load and clean data
raw = pd.read_csv("./compas-scores-two-years.csv")

# %% [markdown]
# However not all of the rows are useable for the first round of analysis.
# 
# There are a number of reasons remove rows because of missing data:
# 
# 
# 
# *   If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
# *   Coded the recidivist flag -- is_recid -- to be -1 if could not find a compas case at all.
# 
# 
# *   In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
# *   Filtering the underlying data to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
# 
# 
# 
# 

# %%
# Select relevant columns and apply filters
df = raw.loc[
    (raw.days_b_screening_arrest <= 30) &
    (raw.days_b_screening_arrest >= -30) &
    (raw.is_recid != -1) &
    (raw.c_charge_degree != 'O') &
    (raw.score_text != 'N/A'),
    ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
     'priors_count', 'days_b_screening_arrest', 'decile_score',
     'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
].copy()

# 2. Basic counts and percentages
total = len(df)
recid_count = df.two_year_recid.sum()
print(f"Total records: {total}")
print(f"Recidivism count: {recid_count}")

# %% [markdown]
# Higher COMPAS scores are slightly correlated with a longer length of stay.

# %%
df['length_of_stay'] = pd.to_numeric(pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in']))
# Convert timedelta to days
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).dt.days
# The .dt.days accessor is used directly on the timedelta result
correlation = df['length_of_stay'].corr(df['decile_score'])
print(f"Correlation between length of stay and decile score: {correlation}")

df["age_cat"].value_counts()

# %%
df["race"].value_counts()

# %%
print(f"Recidivism rate: {recid_count/total*100:.2f}%")

print("Black defendants: %.2f%%" %            (3175 / 6172 * 100))
print("White defendants: %.2f%%" %            (2103 / 6172 * 100))
print("Hispanic defendants: %.2f%%" %         (509  / 6172 * 100))
print("Asian defendants: %.2f%%" %            (31   / 6172 * 100))
print("Native American defendants: %.2f%%" %  (11   / 6172 * 100))

# 3. Distribution by decile_score for African-American vs Caucasian
for race_label in ['African-American', 'Caucasian']:
    plt.figure()
    subset = df[df.race == race_label]
    sns.countplot(x='decile_score', data=subset, order=sorted(df.decile_score.unique()))
    plt.title(f"Decile Score Distribution: {race_label}")
    plt.xlabel('Decile Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# %%
cross_tab = pd.crosstab(df['decile_score'], df['race'])
cross_tab

# %%
!pip install --upgrade statsmodels

# %% [markdown]
# # Racial Bias in Compas
# After filtering out bad rows, our first question is whether there is a significant difference in Compas scores between races. To do so we need to change some variables into factors, and run a logistic regression, comparing low scores to high scores.

# %%
import statsmodels.formula.api as smf
import statsmodels.api as sm
# Assuming 'df' is your pandas DataFrame as defined in the previous code

# Convert relevant columns to categorical factors
df['crime_factor'] = pd.Categorical(df['c_charge_degree'])
df['age_factor'] = pd.Categorical(df['age_cat'])
df['age_factor'] = df['age_factor'].cat.reorder_categories(df['age_factor'].cat.categories, ordered=True)  # Ensure correct order
df['age_factor'] = df['age_factor'].cat.set_categories(df['age_factor'].cat.categories, ordered=True)
df['race_factor'] = pd.Categorical(df['race'])
df['race_factor'] = df['race_factor'].cat.reorder_categories(df['race_factor'].cat.categories, ordered=True)
df['gender_factor'] = pd.Categorical(df['sex']).rename_categories({'Male': "Male", 'Female': 'Female'})
df['gender_factor'] = df['gender_factor'].cat.reorder_categories(df['gender_factor'].cat.categories, ordered=True)
df['score_factor'] = pd.Categorical(df['score_text'] != "Low").rename_categories({True: "HighScore", False: "LowScore"})

# Fit the logistic regression model using statsmodels
model = smf.glm(formula='score_factor ~ gender_factor + age_factor + race_factor + priors_count + crime_factor + two_year_recid',
                data=df, family=sm.families.Binomial()).fit() # Change smf.families to sm.families

# Print the model summary
print(model.summary())

# %% [markdown]
# Black defendants are 45% more likely than white defendants to receive a higher score correcting for the seriousness of their crime, previous arrests, and future criminal behavior.

# %%
import math

control = math.exp(-1.52554) / (1 + math.exp(-1.52554))
result = math.exp(0.47721) / (1 - control + (control * math.exp(0.47721)))
result


# %% [markdown]
# Women are 19.4% more likely than men to get a higher score.

# %%
control = math.exp(-1.52554) / (1 + math.exp(-1.52554))
result = math.exp(0.22127) / (1 - control + (control * math.exp(0.22127)))
result

# %% [markdown]
# People under 25 are 2.5 times as likely to get a higher score as middle aged defendants.

# %%
control = math.exp(-1.52554) / (1 + math.exp(-1.52554))
result = math.exp(1.30839) / (1 - control + (control * math.exp(1.30839)))
result


# %% [markdown]
# # Risk of Violent Recidivism
# Compas also offers a score that aims to measure a persons risk of violent recidivism, which has a similar overall accuracy to the Recidivism score. As before, we can use a logistic regression to test for racial bias.

# %%
violent = pd.read_csv("./compas-scores-two-years-violent.csv")
violent = violent.loc[
    (violent.days_b_screening_arrest <= 30) &
    (violent.days_b_screening_arrest >= -30) &
    (violent.is_recid != -1) &
    (violent.c_charge_degree != 'O') &
    (violent.score_text != 'N/A')
].copy()

print(len(violent))
print(violent.two_year_recid.sum())

# %%
violent["age_cat"].value_counts()

# %%
violent["race"].value_counts()

# %%
violent["v_score_text"].value_counts()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, race_label in enumerate(['African-American', 'Caucasian']):
    subset = violent[violent.race == race_label]
    sns.countplot(x='v_decile_score', data=subset, order=sorted(violent.v_decile_score.unique()), ax=axes[i])
    axes[i].set_title(f"Violent Decile Score Distribution: {race_label}")
    axes[i].set_xlabel("Violent Decile Score")
    axes[i].set_ylabel("Count")
    axes[i].set_ylim(0, 700)

plt.tight_layout()
plt.show()


# %%
df['crime_factor'] = pd.Categorical(df['c_charge_degree'])
df['age_factor'] = pd.Categorical(df['age_cat'])
# Reorder categories to ensure correct order and set them as ordered
df['age_factor'] = df['age_factor'].cat.reorder_categories(df['age_factor'].cat.categories, ordered=True)
df['age_factor'] = df['age_factor'].cat.set_categories(df['age_factor'].cat.categories, ordered=True)

df['race_factor'] = pd.Categorical(df['race'])
race_labels = ["African-American", "Asian", "Caucasian", "Hispanic", "Native American", "Other"]
df['race_factor'] = df['race_factor'].cat.rename_categories(race_labels)
# Set reference level
df['race_factor'] = df['race_factor'].cat.reorder_categories(['Caucasian'] + [r for r in race_labels if r != 'Caucasian'], ordered=True)


df['gender_factor'] = pd.Categorical(df['sex']).rename_categories({'Male': "Male", 'Female': 'Female'})
df['gender_factor'] = df['gender_factor'].cat.reorder_categories(['Male', 'Female'], ordered=True)

df['score_factor'] = pd.Categorical(df['score_text'] != "Low").rename_categories({True: "HighScore", False: "LowScore"})
# Fit the logistic regression model using statsmodels
model = smf.glm(formula='score_factor ~ gender_factor + age_factor + race_factor + priors_count + crime_factor + two_year_recid',
                data=df, family=sm.families.Binomial()).fit()

# Print the model summary
print(model.summary())


# %% [markdown]
# The violent score overpredicts recidivism for black defendants by 77.3% compared to white defendants.

# %%
control = math.exp(-2.24274) / (1 + math.exp(-2.24274))
result = math.exp(0.65893) / (1 - control + (control * math.exp(0.65893)))
result

# %% [markdown]
# Defendands under 25 are 7.4 times as likely to get a higher score as middle aged defendants.

# %%
control = math.exp(-2.24274) / (1 + math.exp(-2.24274))
result = math.exp(3.14591) / (1 - control + (control * math.exp(3.14591)))
result

# %% [markdown]
# # Directions of the Racial Bias

# %%
import csv
from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable

# 1. Read and validate all Person records
people = []
with open("./cox-parsed.csv", newline='') as f:
    reader = PeekyReader(csv.DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                people.append(p)
    except StopIteration:
        pass

# 2. Build the “population”:
#    include everyone with a valid score
#    AND (either a recidivist discharged ≤ 730 days OR lifetime > 730 days)
population = [
    person for person in people
    if person.score_valid
    and (
        (person.recidivist and person.lifetime <= 730)
        or person.lifetime > 730
    )
]

# 3. Extract the recidivists subset:
#    recidivists whose discharge was within 730 days
recidivists = [
    person for person in population
    if person.recidivist and person.lifetime <= 730
]

# 4. Everyone else in “population” (i.e., survivors)
recid_set = set(recidivists)
survivors = [
    person for person in population
    if person not in recid_set
]

# 5. (Optional) Inspect your counts
print(f"Total valid people:   {len(people)}")
print(f"Filtered population:  {len(population)}")
print(f"Recidivists (≤730d):  {len(recidivists)}")
print(f"Survivors (>730d):     {len(survivors)}")

# %%
print("All defendants")
table(list(recidivists), list(survivors))

# %%
import statistics
print("Average followup time %.2f (sd %.2f)" % (statistics.mean(map(lambda i: i.lifetime, population)),
                                                statistics.stdev(map(lambda i: i.lifetime, population))))
print("Median followup time %i" % (statistics.median(map(lambda i: i.lifetime, population))))

# %% [markdown]
# Overall, the false positive rate is 32.35%.

# %%
print("Black defendants")
is_afam = is_race("African-American")
table(list(filter(is_afam, recidivists)), list(filter(is_afam, survivors)))

# %% [markdown]
# That number is higher for African Americans at 44.85%.

# %%
print("White defendants")
is_white = is_race("Caucasian")
table(list(filter(is_white, recidivists)), list(filter(is_white, survivors)))

# %% [markdown]
# And lower for whites at 23.45%.

# %% [markdown]
# Which means under COMPAS black defendants are 91% more likely to get a higher score and not go on to commit more crimes than white defendants after two year.

# %%
44.85 / 23.45

# %% [markdown]
# COMPAS scores misclassify white reoffenders as low risk at 70.4% more often than black reoffenders.

# %%
47.72 / 27.99

# %%
hightable(list(filter(is_white, recidivists)), list(filter(is_white, survivors)))

# %%
hightable(list(filter(is_afam, recidivists)), list(filter(is_afam, survivors)))

# %% [markdown]
# # Risk of Violent Recidivism

# %%
import csv
from truth_tables import PeekyReader, Person

# 1. Read and validate all Person records from the violent CSV
vpeople = []
with open("./cox-violent-parsed.csv", newline="") as f:
    reader = PeekyReader(csv.DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                vpeople.append(p)
    except StopIteration:
        pass

# 2. Build the “violent population”:
#    include everyone with a valid violent score
#    AND (either a violent recidivist discharged ≤ 730 days OR lifetime > 730 days)
vpop = [
    person for person in vpeople
    if person.vscore_valid
    and (
        (person.violent_recidivist and person.lifetime <= 730)
        or person.lifetime > 730
    )
]

# 3. Extract the violent recidivists subset:
#    those violent recidivists whose discharge was within 730 days
vrecid = [
    person for person in vpop
    if person.violent_recidivist and person.lifetime <= 730
]

# 4. Everyone else in “violent population” (i.e., violent survivors)
vrecid_set = set(vrecid)
vsurv = [
    person for person in vpop
    if person not in vrecid_set
]

# 5. (Optional) Inspect your counts
print(f"Total valid violent people:      {len(vpeople)}")
print(f"Filtered violent population:     {len(vpop)}")
print(f"Violent recidivists (≤730d):      {len(vrecid)}")
print(f"Violent survivors (>730d):        {len(vsurv)}")


# %%
print("All defendants")
vtable(list(vrecid), list(vsurv))

# %%
print("Black defendants")
is_afam = is_race("African-American")
vtable(list(filter(is_afam, vrecid)), list(filter(is_afam, vsurv)))

# %%
print("White defendants")
is_white = is_race("Caucasian")
vtable(list(filter(is_white, vrecid)), list(filter(is_white, vsurv)))

# %% [markdown]
# Black defendants are twice as likely to be false positives for a Higher violent score than white defendants.

# %%
31.45 / 13.89

# %% [markdown]
# White defendants are 95% more likely to get a lower score and commit another crime than Black defendants.

# %%
57.55 / 29.37



