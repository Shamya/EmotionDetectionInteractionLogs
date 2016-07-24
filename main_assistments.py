import numpy as np
from sklearn import cross_validation, ensemble, decomposition
import matplotlib.pyplot as plt
from cohenskappa import cohens_scorer

merged_dir = "data/assistments/merged/"
history_dir = "data/assistments/history/"

pipeline_dir = "data/pipeline/"

Y_merged = np.load(merged_dir + "labels.npy")
X_merged = np.load(merged_dir + "features.npy")

Y = np.load(history_dir + "labels.npy")
X = np.load(history_dir + "features.npy")
column_labels = np.load(history_dir + "column_labels.npy")
splitter = cross_validation.StratifiedShuffleSplit(Y, n_iter=1)
for train_index, test_index in splitter:
	X_test = X[test_index]
	Y_test = Y[test_index]
	X = X[train_index]
	Y = Y[train_index]

label_identifiers = np.load(history_dir + "label_identifiers.npy")

clf = ensemble.RandomForestClassifier(n_estimators=10)
cv = cross_validation.StratifiedShuffleSplit(Y, n_iter=3)

np.random.seed(273)
score = cross_validation.cross_val_score(clf, X_merged, Y_merged,
					scoring=cohens_scorer, cv=cv).mean()
print "Default model with merged data scores Cohen's Kappa of %.3f" % score

np.random.seed(273)
D = X.shape[1]
score = cross_validation.cross_val_score(clf, X, Y,
					scoring=cohens_scorer, cv=cv).mean()

print "Default model with history-enriched data scores Cohen's Kappa of %.3f" % score

### Dimensionality Reduction
#Does not perform well
np.random.seed(273)
pca_data = [] #stores (n_components, score) 
for D_t in xrange(1,D+1):
	np.random.seed(273)
	transformer = decomposition.PCA(n_components=D_t)
	X_t = transformer.fit_transform(X)
	score = cross_validation.cross_val_score(clf, X_t, Y,
						scoring=cohens_scorer, cv=cv).mean()
	pca_data.append((D_t, score))
	print "Cohen's Kappa of %.3f with %03d principal components." % \
			(score, D_t)

pca_data = np.array(pca_data)
np.save(pipeline_dir + "pca_data.npy", pca_data)

plt.figure()
plt.title("Principal Component Analysis")
plt.xlabel("Number of Components")
plt.ylabel("Cohen's Kappa Score")
plt.plot(pca_data[:,0], pca_data[:,1])
plt.savefig("figures/PCA.png")
plt.clf()

### Feature Selection

np.random.seed(273)

features = np.zeros(D, dtype=np.bool_)
feature_data = [] #stores tuples of (num_features, score, feature index)
best_feature_set = (float("-inf"), -1) #(score, num_features)
for num_features in xrange(1,D+1):
	best_feature = (float("-inf"), -1)
	for i in xrange(D):
		if not features[i]:
			np.random.seed(273)
			features[i] = True
			score = cross_validation.cross_val_score(clf, X[:,features], Y,
						scoring=cohens_scorer, cv=cv).mean()
			if score > best_feature[0]:
				best_feature = (score, i)
			features[i] = False
	features[best_feature[1]] = True
	feature_data.append((num_features, best_feature[0], best_feature[1]))
	print "Cohen's Kappa of %.3f with %03d features (selected feature %d)." % \
		(best_feature[0], num_features, best_feature[1])
	if best_feature[0] > best_feature_set[0]:
		best_feature_set = (best_feature[0], num_features)

feature_data = np.array(feature_data)
np.save(pipeline_dir + "feature_selection_data.npy", feature_data)

#Reconstruct the greatest of the feature sets
D_t = best_feature_set[1]
features = np.zeros(D_t, dtype=np.int_)
for i in range(D_t-1):
	features[i] = feature_data[i, 2]

#convert to index format instead of boolean mask
# features = np.arange(D)[features]
np.save(pipeline_dir + "selected_features.npy", features)

plt.figure()
plt.title("Forward Stepwise Feature Selection")
plt.xlabel("Number of Features")
plt.ylabel("Cohen's Kappa Score")
plt.plot(feature_data[:,0], feature_data[:,1])
plt.savefig("figures/FeatureSelection.png")
plt.clf()

np.random.seed(273)
# 
features = np.load(pipeline_dir + "selected_features.npy")

trees_data = [] #stores (num_trees, score)
for num_trees in xrange(10, 301, 10):
	np.random.seed(273)
	clf.set_params(n_estimators=num_trees)
	score = cross_validation.cross_val_score(clf, X[:,features], Y,
						scoring=cohens_scorer, cv=cv).mean()
	trees_data.append((num_trees, score))
	print "Cohen's Kappa of %.3f with %03d trees." % (score, num_trees)

trees_data = np.array(trees_data)

plt.figure()
plt.title("Hyperparamter Selection: Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Cohen's Kappa Score")
plt.plot(trees_data[:,0], trees_data[:,1])
plt.savefig("figures/TreesSelection.png")
plt.clf()

np.save(pipeline_dir + "tree_selection_data.npy", trees_data)

trees_data = np.load(pipeline_dir + "tree_selection_data.npy")

best_score = float("-inf")
n_trees = 10
for trees,score in trees_data:
	if score > best_score:
		best_score = score
		n_trees = trees
n_trees = int(n_trees)

np.random.seed(273)

print "Final result of the pipeline, using %d features and %d trees:" % \
	(len(features), n_trees)

print "Cohen's Kappa of %.3f." % (best_score,)

np.random.seed(273)
print "Running this on the full training set and testing on the held-out test:"
clf.fit(X, Y)
print "Cohen's Kappa of %.3f." % cohens_scorer(clf, X_test, Y_test) 

print

print "Features selected, in order of importance:"
for feature_index in features:
	print column_labels[feature_index]