import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']

# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full = np.column_stack((f1,f2))
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

ph_1 = phoneme_id[phoneme_id == 1]
ph_2 = phoneme_id[phoneme_id == 2]

X_phoneme_1 = np.zeros((np.sum(phoneme_id==1), 2))
X_phoneme_1 = np.column_stack((X_full[phoneme_id==1,0], X_full[phoneme_id==1,1]))
X_phoneme_2 = np.zeros((np.sum(phoneme_id==2), 2))
X_phoneme_2 = np.column_stack((X_full[phoneme_id==2,0], X_full[phoneme_id==2,1]))

ph_1_2 = np.concatenate([ph_1, ph_2])
X_phonemes_1_2 = np.concatenate((X_phoneme_1, X_phoneme_2))

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"


X=X_phonemes_1_2.copy()

# file path for phoneme 1 and phoneme 2 params learned in task 2
phoneme1_file = 'data/GMM_params_phoneme_01_k_0'+str(k)+'.npy'
phoneme2_file = 'data/GMM_params_phoneme_02_k_0'+str(k)+'.npy'

# prediction for phoneme 1
phoneme_1_data = np.ndarray.tolist(np.load(phoneme1_file, allow_pickle=True))
predictions1 = get_predictions(phoneme_1_data['mu'], phoneme_1_data['s'], phoneme_1_data['p'], X)
predictions1 = np.sum(predictions1, axis=1)

# predictions for phoneme 2
phoneme_2_data = np.ndarray.tolist(np.load(phoneme2_file, allow_pickle=True))
predictions2 = get_predictions(phoneme_2_data['mu'], phoneme_2_data['s'], phoneme_2_data['p'], X)
predictions2 = np.sum(predictions2, axis=1)

predictions = np.zeros((np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2), 1))

for i in range(len(predictions)):
    if predictions1[i] > predictions2[i]:
        predictions[i] = 1
    else:
        predictions[i] = 2

# calculating the count of correct data
total_correct_data = 0
for i in range(len(ph_1_2)):
    if predictions[i] == ph_1_2[i]:
        total_correct_data += 1

# calculating the accuracy
total_data = len(ph_1_2)
accuracy = (total_correct_data/total_data)*100
miss_classification = (1-(total_correct_data/total_data))*100
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))
print('Misclassification error using GMMs with {} components: {:.2f}%'.format(k, miss_classification))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()