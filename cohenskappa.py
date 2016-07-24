from sklearn import metrics

def cohens_scorer(estimator, X, y):
	y_pred = estimator.predict(X)
	return metrics.cohen_kappa_score(y_pred, y)