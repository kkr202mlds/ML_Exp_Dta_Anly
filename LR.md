### Logistic Regression is a supervised machine learning algorithm used for binary classification problems in a discrete (non-continuous) outcome. Logistic regression predicts the discrete (non-continuous) output of a categorical dependent variable. i.e., Sigmoid(0, 1), Hypothesis(likelihood) and Log Loss Function

```
# LogisticRegression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# o/p -> LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
y_pred = classifier.predict(X_test)
```

Linear model types:-
Regression
Generalized linear models
Linear regression
Logistic regression
Poisson distribution
ElasticNet regression
Linear predictor function
Mixed models
Multilevel structural equation modeling
Ordinal regression
