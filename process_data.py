import numpy as np
import re

filenames = ["bored", "concentrating", "confused", "frustrated"]
history_dir = "data/assistments/rearranged/"
merged_dir = "data/assistments/merged/"
history_dir = "data/assistments/history/"
grouped_dir = "data/assistments/grouped/"

UNIQUE_ID_PATTERN = "^(.*-(\d{2}.\d{2}.\d{4}_at_\d{2}.\d{2}.\d{2})(-\d+)?)-(\d+)$"
SESSION_TIME_PATTERN = "^(\d{2}.\d{2}).(\d{4})_at_(.*)$"

data = {}
for affect in filenames:
	data[affect] = np.genfromtxt(history_dir + affect + ".csv",
								dtype=str, delimiter=",")
# 	print affect, data[affect].shape

student_ids = data['confused'][1:,0]
data['confused'] = data['confused'][:, 1:]
# student_names = data['frustrated'][1:,0] #turns out these are student ids too
data['frustrated'] = data['frustrated'][:, 1:]
column_labels = data['frustrated'][0][1:-1] #remove the UNIQUE_ID label

### Index the original resampled data by the unique row identifiers

np.random.seed(273)

masks = {}
i = 1
row_lookup = {}
student_id_lookup = {}
for affect,array in data.items():
	array = array[1:] #slice off column headers
	masks[affect] = i #build a mask for telling if a row exists in a data set
	i *= 2
	for r,row in enumerate(array):
		#get the unique row identifier
		unique_id = row[-1]
		if unique_id not in row_lookup:
			row_lookup[unique_id] = [0, []]
		
		#add this data set to the mask for this row
		row_lookup[unique_id][0] = row_lookup[unique_id][0] | masks[affect]
		
		#store a reference to this row
		row_lookup[unique_id][1].append(row)
		
		if affect == 'confused':
			student_id_lookup[unique_id] = student_ids[r]
full_mask = i-1

### Generate a merged data set to allow for multiclass classification

merged_data = []
unique_ids = []
affects = np.array(["UNKNOWN",
				"BORED", "CONCENTRATING", "CONFUSED", "FRUSTRATED"])
affect_lookup = {affect:identifier for identifier,affect in enumerate(affects)}
counts = dict(BORED=0, FRUSTRATED=0, CONCENTRATING=0, CONFUSED=0, UNKNOWN=0)
for unique_id,(mask, rows) in row_lookup.items():
	if mask == full_mask:
		merged_data.append(rows[0][:-1].copy())
		unique_ids.append(unique_id)
		merged_data[-1][0] = affect_lookup["UNKNOWN"]
		for row in rows:
			if row[0] != "NOT":
				merged_data[-1][0] = affect_lookup[row[0]]
		counts[affects[int(merged_data[-1][0])]] += 1
	else:
		#This can be kept, it indicates something's wrong
		print mask, unique_id

print "Counts of data cases representing each class in the data:"
for affect,count in counts.items():
	print str(count).rjust(4), affect
print

merged_data = np.array(merged_data)
unique_ids = np.array(unique_ids)
Y_merged = merged_data[:, 0].astype(np.int_)
X_merged = merged_data[:, 1:].astype(np.float_)

# Shuffle the merged data before saving
indices = np.arange(len(Y_merged))
np.random.shuffle(indices)

np.save(merged_dir + "labels.npy", Y_merged[indices])
np.save(merged_dir + "features.npy", X_merged[indices])
np.save(merged_dir + "column_labels.npy", column_labels)
np.save(merged_dir + "label_identifiers.npy", affects)

### Generate a data set with features keeping track of past affects

#add the slots for the new features
affect_history = np.zeros((X_merged.shape[0], len(affects)))
affect_column_labels = np.array(["PRIOR " + affect for affect in affects])
column_labels = np.concatenate((column_labels, affect_column_labels))
X_historical = np.concatenate((X_merged, affect_history), axis=1)
affect_offset = X_merged.shape[1]

base = 0.5
complement = 1.0-base

ordered_indices = []

#Group rows by sessions and students within sessions
sessions = {}
session_list = []
for rownum,unique_id in enumerate(unique_ids):
	session_id, session_time, obs_num = \
		re.search(UNIQUE_ID_PATTERN, unique_id).group(1,2,4)
	student_id = student_id_lookup[unique_id]
	if session_id not in sessions:
		sessions[session_id] = {}
		monthday, year, time = \
			re.search(SESSION_TIME_PATTERN, session_time).group(1,2,3)
		session_time = year + "." + monthday + "_" + time
		session_list.append((session_time, session_id))
	if student_id not in sessions[session_id]:
		sessions[session_id][student_id] = []
	sessions[session_id][student_id].append((int(obs_num), int(rownum)))

session_list = np.array(session_list)
session_list = session_list[session_list[:,0].argsort()] #sort by time

for session_id in session_list[:,1]:
	for student_id,rows in sessions[session_id].items():
		rows = np.array(rows)
		rows = rows[rows[:,0].argsort()] #sort by observation number; time-order
		last_affect = affect_lookup["UNKNOWN"]
		last_row = None
# 		print
		for row in rows:
			rownum = row[1]
			ordered_indices.append(rownum)
			if last_row is not None:
				#copy the previous values, weighting them down a bit
				X_historical[rownum, affect_offset:] = \
						X_historical[last_row, affect_offset:] * complement
			last_row = rownum
			#increase the value for the most recently observed affect
			X_historical[rownum, affect_offset + last_affect] += base
# 			print last_affect, X_historical[rownum, affect_offset:]
			last_affect = Y_merged[rownum]

np.save(history_dir + "labels.npy", Y_merged[indices])
np.save(history_dir + "features.npy", X_historical[indices])
np.save(history_dir + "column_labels.npy", column_labels)
np.save(history_dir + "label_identifiers.npy", affects)

### Also save a version that's not scrambled, but instead sorted by:
###  session -> student -> observations in temporal order

ordered_indices = np.array(ordered_indices, dtype=np.int_)
np.save(grouped_dir + "labels.npy", Y_merged[ordered_indices])
np.save(grouped_dir + "features.npy", X_historical[ordered_indices])
np.save(grouped_dir + "column_labels.npy", column_labels)
np.save(grouped_dir + "label_identifiers.npy", affects)
