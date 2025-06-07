# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a binary classification model built using a `RandomForestClassifier` from scikit-learn. It predicts whether an individual's income exceeds $50K per year using demographic and employment features. The model is part of a scalable ML pipeline served via a FastAPI RESTful interface.

## Intended Use
This model is intended for educational purposes, specifically to demonstrate:
- ML pipeline design
- Data preprocessing and model training
- API deployment and CI/CD integration

It is **not intended** for production use or real-world decision making.

## Training Data
- Source: UCI Census Income dataset
- Size: Approximately 32,500 rows
- Features: age, workclass, education, marital-status, occupation, relationship, race, sex, native-country, capital-gain, capital-loss, hours-per-week
- Target: salary (`<=50K` or `>50K`)

## Evaluation Data
The dataset was split into training and testing sets using an 80/20 ratio. The same test set was also used for slice-based evaluation across categorical features.

## Metrics
**Overall Performance:**
- Precision: 0.7419 
- Recall: 0.6384 
- F1 Score: 0.6863 

**Slice Evaluation:**
- Performance was measured across categorical slices such as `education`, `workclass`, and `race`.
- Results are saved in `slice_output.txt`.

## Ethical Considerations
- The training data may contain historical or societal biases.
- Certain demographic groups may be underrepresented, affecting fairness.
- This model should not be used in real applications involving employment, housing, or credit decisions without bias mitigation strategies.

## Caveats and Recommendations
- Predictions may be inaccurate for underrepresented groups or outlier profiles.
- Model assumes data is preprocessed identically to training.
- Fairness evaluations and frequent retraining with fresh data are recommended if deployed beyond the classroom.
- Use only in a controlled, educational setting.
