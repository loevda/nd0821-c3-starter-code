# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a scikit-learn `RandomForestClassifier` with 100 estimators
(`n_estimators=100`, `random_state=42`) that predicts whether an individual's
annual income exceeds $50,000. It was trained on the cleaned Census Income
dataset (`data/census_clean.csv`) using an 80/20 train/test split
(`random_state=42`). The eight categorical features (`workclass`, `education`,
`marital-status`, `occupation`, `relationship`, `race`, `sex`,
`native-country`) are one-hot encoded with a `OneHotEncoder`
(`handle_unknown="ignore"`), and the binary label is encoded with a
`LabelBinarizer`. The trained model and encoder are persisted as
`model/trained_model.pkl` and `model/trained_encoder.pkl` and are loaded by the
FastAPI service for inference.

## Intended Use

The intended use of this model is to estimate the probability category of an
individual's income level (above or below $50,000) from census-style
attributes, for example as a demonstration of a tabular classification
pipeline or as a baseline for income-related analytics. It is intended for
research, educational, and prototyping contexts. It is not intended to make or
directly inform high-stakes individual decisions such as credit approval,
hiring, or benefits eligibility without additional fairness review and
validation.

## Training Data

The model was trained on `data/census_clean.csv`, a cleaned version of the
public UCI Adult / Census Income dataset, which contains 32,561 records and 15
columns. Each record describes one individual with demographic and
employment attributes, including age, workclass, education, marital status,
occupation, relationship, race, sex, capital gain and loss, hours per week,
and native country. The target column is `salary`, a binary label with the
values `<=50K` and `>50K`. The data was split into a training set of 26,049
records (80%) and a held-out test set of 6,512 records (20%) using
`train_test_split` with `test_size=0.20` and `random_state=42`. No resampling,
imputation, or feature selection was applied beyond the one-hot encoding
described above.

## Evaluation Data

The model was evaluated on the held-out test set of 6,512 records described
above, which was not used during training. In addition, the test set was
sliced by the `education` feature so that performance could be measured
separately for each of the 16 unique education levels; the per-slice results
are reported in the Metrics section and written to `slice_output.txt` by the
training script.

## Metrics

The model's performance is measured with precision, recall, and the F1 score
(the F-beta score with beta=1), computed on the binarized labels. On the full
held-out test set, the model achieved:

- **Precision: 0.742** — of the individuals predicted to earn more than
  $50,000, about 74.2% actually did.
- **Recall: 0.638** — of the individuals who actually earn more than
  $50,000, the model correctly identified about 63.8%.
- **F1: 0.686** — the harmonic mean of precision and recall.

Performance on slices of the test set, with the `education` feature held
fixed, is as follows (one row per unique education value):

| Education    | Precision | Recall | F1    |
|--------------|-----------|--------|-------|
| 1st-4th      | 1.000     | 1.000  | 1.000 |
| 5th-6th      | 1.000     | 0.500  | 0.667 |
| 7th-8th      | 0.000     | 0.000  | 0.000 |
| 9th          | 1.000     | 0.333  | 0.500 |
| 10th         | 0.400     | 0.167  | 0.235 |
| 11th         | 1.000     | 0.273  | 0.429 |
| 12th         | 1.000     | 0.400  | 0.571 |
| Preschool    | 1.000     | 1.000  | 1.000 |
| HS-grad      | 0.659     | 0.438  | 0.526 |
| Some-college | 0.686     | 0.520  | 0.591 |
| Assoc-voc    | 0.647     | 0.524  | 0.579 |
| Assoc-acdm   | 0.700     | 0.596  | 0.644 |
| Bachelors    | 0.752     | 0.729  | 0.740 |
| Masters      | 0.827     | 0.855  | 0.841 |
| Doctorate    | 0.864     | 0.895  | 0.879 |
| Prof-school  | 0.818     | 0.964  | 0.885 |

## Ethical Considerations

The training data contains sensitive attributes, most notably `race` and
`sex`, and the target variable (income) is correlated with socioeconomic
factors that are themselves historically shaped by discrimination. A model
trained on this data can therefore learn and amplify existing biases: for
example, it may systematically under-predict high income for certain groups
even when their qualifications are comparable. If this model were used in
decisions that affect people (credit, employment, insurance), disparate
impact on protected groups would need to be measured and mitigated before
deployment. The dataset also reflects a specific time period and population
(the United States), so its assumptions may not transfer to other contexts.
Users should treat its outputs as statistical estimates, not as
determinations of an individual's worth or eligibility.

## Caveats and Recommendations

The overall F1 of 0.686 masks substantial variation across education slices:
the model performs well for individuals with a Bachelor's degree or higher
(F1 between 0.740 and 0.885) but poorly for several lower-education groups,
with F1 of 0.235 for `10th` and 0.000 for `7th-8th`. Some of these slices
contain very few test records, so their per-slice metrics are statistically
noisy and should not be over-interpreted. We recommend the following before
any production use: (1) evaluate fairness metrics (e.g., equalized odds or
demographic parity) across `race` and `sex` slices, not only `education`;
(2) consider the precision/recall trade-off explicitly, since the current
model favors precision over recall; (3) validate on a more recent and
representative dataset, since income distributions shift over time; and
(4) keep human review in the loop for any decision that materially affects
individuals.
