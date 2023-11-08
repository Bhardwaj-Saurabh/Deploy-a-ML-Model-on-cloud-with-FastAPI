# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Selected model: Random Forest Classifier
Hyper-parameters: Default

## Intended Use
Model predicts if the income is more than $50K per year or not.

## Training Data
The dataset used is [Census data]('https://archive.ics.uci.edu/ml/datasets/census+income').

## Evaluation Data
Steps taken: Data Preprocessing and 80/20 train and test split.

## Metrics
The model has achieved following mertrics on test dataset.

Precision:  0.74
Recall:  0.63
Fbeta:  0.68

## Ethical Considerations
As the data contains personal information such as sex, race, age, it is important to check for model bias and model evaluation should be done across different features.

## Caveats and Recommendations
Certainly, model has chances to improve it's overall performance. Following steps can be taken to improve it's performance.

- Hyper-parameter optimisation
- Feature Engineering
- Outlier Detection
- Multi-model Techniques for different groups
