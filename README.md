# LicketySPLIT

LicketySPLIT is a Python package with a C++ backend for learning sparse decision trees for classification. It builds on the LicketySPLIT algorithm from [Near Optimal Decision Trees in a SPLIT Second](https://arxiv.org/abs/2502.15988) and generalizes it with recursive pilot ideas as in [From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](https://arxiv.org/abs/2606.00202). For continuous features, we recommend the included binarization techniques from [*Fast Sparse Decision Tree Optimization via Reference Ensembles*](https://arxiv.org/abs/2112.00798), which preserve optimality with respect to a gradient-boosted tree ensemble.


The package supports binary and multi-class classification, optional sample weights, efficient subproblem caching, and threshold-based binarization for continuous features.

See the example notebook [here.](https://github.com/zakk-h/LicketySPLIT/blob/main/example.ipynb)

## Objective

LicketySPLIT learns a sparse decision tree by attempting to minimize a regularized empirical training objective.

For an unweighted dataset with `n` training samples, the objective is

```text
objective(tree) = training_mistakes(tree) + lambda_leaf * n * number_of_leaves(tree)
```

where:

- `training_mistakes(tree)` is the number of training samples misclassified by the tree
- `number_of_leaves(tree)` is the number of leaves in the tree
- `lambda_leaf` is the leaf regularization parameter
- `n` is the number of training samples

For a weighted dataset, `n` is replaced with the sum of weights and the `training_mistakes(tree)` penalizes incorrect points by their weight instead of by 1.

## Classification setting

Labels should be encoded as contiguous nonnegative integers:

```python
0, 1, 2, ..., num_classes - 1
```

## Features

LicketySPLIT expects binary input features. For continuous data, use `ThresholdGuessBinarizer` (included from https://arxiv.org/abs/2112.00798) to generate binary threshold features.

```python
from licketysplit import ThresholdGuessBinarizer

binarizer = ThresholdGuessBinarizer(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    random_state=0,
    column_elimination=False,
)

X_train_bin = binarizer.fit_transform(X_train_raw, y_train)
X_test_bin = binarizer.transform(X_test_raw)

feature_names = binarizer.get_feature_names_out()
```

Each generated feature has the form

```text
original_feature <= threshold
```

## Basic usage

```python
from licketysplit import LicketySPLIT

model = LicketySPLIT(
    cache_mode="fingerprint",
    cost_caching_enabled=True,
)

model.fit(
    X_train_bin,
    y_train,
    lambda_leaf=0.001,
    depth_budget=6,
    lookahead_k=1,
)

y_pred = model.predict(X_test_bin)
```

## Full example

A complete notebook showing loading data, binarization, fitting, prediction, tree inspection, plotting, caching, and sample weights is available [here](https://github.com/zakk-h/LicketySPLIT/blob/main/example.ipynb).

## References

This package builds on the LicketySPLIT algorithm from:

[Near Optimal Decision Trees in a SPLIT Second](https://arxiv.org/abs/2502.15988)

and generalizes it with a pilot algorithm approach as in:

[From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](https://arxiv.org/abs/2606.00202)

For continuous-feature binarization, we recommend:

[*Fast Sparse Decision Tree Optimization via Reference Ensembles*](https://arxiv.org/abs/2112.00798)

