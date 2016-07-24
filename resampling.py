from sklearn.cross_validation import StratifiedKFold,bincount,check_random_state
import numpy as np

class ResampledKFold(StratifiedKFold):
	"""Resampled Stratified K-Folds cross validation iterator

	Provides train/test indices to split data in train test sets.

	This cross-validation object is a variation of KFold that
	returns resampled stratified folds. The folds are made by preserving
	the percentage of samples for each class, then resampling from each
	class to result in equal shares.

	Parameters
	----------
	y : array-like, [n_samples]
		Samples to split in K folds.

	n_folds : int, default=3
		Number of folds. Must be at least 2.

	shuffle : boolean, optional
		Whether to shuffle each stratification of the data before splitting
		into batches.

	random_state : None, int or RandomState
		When shuffle=True, pseudo-random number generator state used for
		shuffling. If None, use default numpy RNG for shuffling.
	"""
	
	def __init__(self, y, n_folds=3, shuffle=False, random_state=None):
		super(ResampledKFold, self).__init__(y, n_folds, shuffle, random_state)
		self.y = y
	
	#We need to override the _PartitionIterator version of this, which
	# only lets each index appear once
	def __iter__(self):
		ind = np.arange(self.n)
		for test_index in self._iter_test_masks():
			train_index = np.logical_not(test_index)
			train_index = self._resample_partition(ind[train_index])
			test_index = self._resample_partition(ind[test_index])
			yield train_index, test_index

	def _resample_partition(self, partition):
		rng = check_random_state(self.random_state)
		y = self.y[partition]
		unique_labels, y_inversed = np.unique(y, return_inverse=True)
		label_counts = bincount(y_inversed)
		class_share = max(label_counts)
		resampled_partition = np.empty(class_share*len(unique_labels),
										dtype=np.int_)
		for i,label in enumerate(unique_labels):
			indices = partition[y == label]
			class_size = len(indices)
			offset = class_share*i
			added = 0
			while added < class_share:
				rng.shuffle(indices)
				to_add = min(class_share - added, class_size)
				resampled_partition[offset+added:offset+added+to_add] = \
					indices[:to_add]
				added += to_add
		return resampled_partition
