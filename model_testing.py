from sklearn import ensemble, linear_model, tree, neighbors, naive_bayes, \
					lda, svm, cross_validation
import numpy as np
from cohenskappa import cohens_scorer

merged_dir = "data/assistments/merged/"
labels = np.load(merged_dir + "labels.npy")
features = np.load(merged_dir + "features.npy")


cv = cross_validation.StratifiedShuffleSplit(labels, random_state = 0)

print "Cohen's kappa for the different basic classifiers - "
clf = ensemble.RandomForestClassifier(random_state=0)
scores = cross_validation.cross_val_score(clf, features, labels, cv=cv, scoring=cohens_scorer).mean()
print "Random Forest %f" % scores
clf = linear_model.LogisticRegression(random_state=0)
scores = cross_validation.cross_val_score(clf, features, labels, cv=cv, scoring=cohens_scorer).mean()
print "Logistic Regression %f" % scores
clf = tree.DecisionTreeClassifier(random_state=0)
scores = cross_validation.cross_val_score(clf, features, labels, cv=cv, scoring=cohens_scorer).mean()
print "Decision Tree %f" % scores
clf = neighbors.KNeighborsClassifier()
scores = cross_validation.cross_val_score(clf, features, labels, cv=cv, scoring=cohens_scorer).mean()
print "KNN %f" % scores
clf = naive_bayes.GaussianNB()
scores = cross_validation.cross_val_score(clf, features, labels, cv=cv, scoring=cohens_scorer).mean()
print "Naive Bayes %f" % scores
clf = lda.LDA()
scores = cross_validation.cross_val_score(clf, features, labels, cv=cv, scoring=cohens_scorer).mean()
print "LDA %f" % scores
clf = svm.SVC(kernel='rbf',random_state = 0)
scores = cross_validation.cross_val_score(clf, features, labels, cv=cv, scoring=cohens_scorer).mean()
print "RBF SVM %f" % scores