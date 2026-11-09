# Spec Delta

## Purpose

Lets a user train classifiers on simulated or sampled runs to name a coming bifurcation, predict tipping and its
probability, and locate phase transitions along a sweep with a lower bound of the Fisher information.

## ADDED Requirements

### Requirement: Transitions are classified from labelled runs
`transition_classifier(source, training=..., labels=...)` SHALL train a classifier on labelled training series and
SHALL return a frozen result with the classes, the probability of every class for every series of `source`, the most
probable class, the training and validation accuracy, and the trained classifier. The labels SHALL be one per training
series, or the name of a column of an experiment result.

#### Scenario: Bifurcation type of an untrained system
- **WHEN** the classifier is trained on forced and null runs of the discrete normal forms of the five bifurcations and
  is applied to forced runs of the Fox map and of the Westerhoff model at 80% of the way to the bifurcation
- **THEN** the most probable class is period-doubling for most Fox runs and Neimark-Sacker for most Westerhoff runs

#### Scenario: Null runs
- **WHEN** the same classifier is applied to null runs of the Fox map with a fixed stimulation period
- **THEN** the most probable class is the null class for most runs

#### Scenario: Labels from an experiment
- **WHEN** the training runs are a `sweep` result and `labels` names its design column `kind`
- **THEN** the classes are the values of that column in sorted order

### Requirement: Classifiers are neural or from scikit-learn
The model SHALL be `'cnn_lstm'`, a convolutional layer with pooling followed by recurrent layers and a dense output,
trained by Adam on the cross entropy, or any scikit-learn classifier with `predict_proba`. Training series SHALL be
censored to random windows, normalised by their mean absolute value, and padded to a fixed length, as in 2303.09669.

#### Scenario: A scikit-learn model
- **WHEN** `model` is a `HistGradientBoostingClassifier` and the training runs are normal-form runs
- **THEN** the result holds that fitted classifier, and its probabilities sum to 1 for every series

#### Scenario: Reproducible training
- **WHEN** the CNN-LSTM is trained twice with the same `random_state`
- **THEN** the two results hold equal probabilities

### Requirement: Trained classifiers are reused
`transition_classifier` SHALL take `classifier`, the trained classifier of an earlier result, in place of training data,
and SHALL then classify `source` without training. Giving both or neither SHALL raise `AnalysisError`.

#### Scenario: Reuse
- **WHEN** the classifier of a first result classifies new runs
- **THEN** the probabilities equal those of a result that classifies the same runs while training with the same seed

#### Scenario: Missing training data
- **WHEN** neither training runs nor a classifier is given
- **THEN** `AnalysisError` is raised and names both settings

### Requirement: Irregular samples are classified
A series SHALL be given with its sample times, and the classifier SHALL then read the time gaps as a second input
channel, so series with irregular sampling are trained and classified, as in 2407.18693.

#### Scenario: Irregular tipping
- **WHEN** the classifier is trained on runs of continuous normal forms sampled at random times, labelled by whether
  they tip, and is applied to irregularly sampled runs of `RampedStochasticFoldSystem`
- **THEN** its accuracy on these runs is above the share of the most common class

### Requirement: Tipping probabilities are calibrated
For binary labels the probability of the positive class SHALL be the predicted probability of tipping, and the result
SHALL report its Brier score on the validation series, as in 2409.07590 for rate-induced tipping.

#### Scenario: Rate-induced tipping of the overturning circulation
- **WHEN** the classifier is trained on early windows of runs of `StommelSystem` under freshwater ramps of random rate
  and noise, labelled by whether the run crosses to the reversed circulation, and applied to new runs
- **THEN** the Brier score of its probabilities is lower than that of the constant base rate of tipping

### Requirement: Phase transitions are located by learned indicators
`phase_indicator(source, parameter=..., output=..., kind=...)` SHALL read samples at every value of one swept parameter,
from an experiment result or from arrays, and SHALL return a frozen result with the parameter values, the indicator, the
critical value at its maximum, and the model. The kinds SHALL be `'supervised'` with labels from the two ends,
`'confusion'` with a bipartition at every value, and `'prediction'` that regresses the parameter, as in 2311.10710.

#### Scenario: Ising transition
- **WHEN** the confusion indicator runs on configurations of a 16 × 16 `IsingSystem` sampled at 21 temperatures from 1.5
  to 3.5
- **THEN** the critical value lies between 2.0 and 2.6

#### Scenario: Unknown kind
- **WHEN** `kind` is not one of the three kinds
- **THEN** `AnalysisError` is raised and names the kinds

### Requirement: Indicators bound the Fisher information
The result of `phase_indicator` SHALL hold `fisher_bound`, the square of the lower bound of the square root of the
Fisher information that the indicator gives at every parameter value: the supervised indicator over the standard
deviation of the prediction, twice the confusion indicator over the parameter step, or the prediction-based indicator.

#### Scenario: Bound below the exact value
- **WHEN** the prediction-based indicator runs on Ising configurations along temperature, and the Fisher information
  `Var(E) / T^4` is computed from the energies of the same configurations
- **THEN** `fisher_bound` stays below that Fisher information within 10% at every temperature, and peaks within 0.3 of
  the temperature where it peaks
