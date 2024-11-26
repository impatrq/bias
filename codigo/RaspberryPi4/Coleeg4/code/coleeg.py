__version__ = 4

# importing modules
import mne
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import scipy.io
from scipy.fft import dct
import sys
import time
import os
from IPython import get_ipython
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda, Permute
from keras import activations
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, InputLayer, Conv1D
from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, LSTM
from tensorflow.keras.layers import Reshape, GlobalAveragePooling1D, TimeDistributed, Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import csv
import re




# Module variables
# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
this.DATA_FOLDER = None
this.RESULT_FOLDER = None
this.CONTENT_FOLDER = None
this.TIME_ZONE = None
this.METRICS = None
this.METRICS_TO_SAVE = None





####################### defining CohenKappa Class ##########################
class CohenKappa(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='cohen_kappa', **kwargs):
        super(CohenKappa, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(name='conf_mtx', shape=(num_classes, num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        confusion = tf.cast(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes), dtype=tf.float32)
        self.conf_mtx.assign_add(confusion)

    def result(self):
        sum_diag = tf.reduce_sum(tf.linalg.diag_part(self.conf_mtx))
        sum_rows = tf.reduce_sum(self.conf_mtx, axis=0)
        sum_cols = tf.reduce_sum(self.conf_mtx, axis=1)
        total_samples = tf.reduce_sum(sum_rows)

        po = sum_diag / total_samples
        pe = tf.reduce_sum(sum_rows * sum_cols) / (total_samples ** 2)
        kappa = (po - pe) / (1 - pe)

        return kappa

    def reset_state(self):
        # Reset the confusion matrix
        self.conf_mtx.assign(tf.zeros_like(self.conf_mtx))
############################################################################





############################ Function Start ################################
def coleeg_version():
  print(this.__version__)
############################ Function End #################################

############################ Function Start ################################
def set_metrics(metrics,n_outputs=4):
  this.METRICS = []
  this.METRICS_TO_SAVE = metrics
  if any(item in ['accuracy','val_accuracy'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append('accuracy')
  if any(item in ['cohen_kappa','val_cohen_kappa'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append(CohenKappa(num_classes=n_outputs))
  if any(item in ['specificity','val_specificity'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append(specificity)
  if any(item in ['sensitivity','val_sensitivity'] for item in this.METRICS_TO_SAVE):
    this.METRICS.append(sensitivity)
############################ Function End #################################

############################ Function Start ################################
def get_metrics():
  return this.METRICS_TO_SAVE
############################ Function End #################################

############################ Function Start ################################
def set_time_zone(zone):
  this.TIME_ZONE = zone
############################ Function End #################################

############################ Function Start ################################
def set_data_folder(folder):
  from os import path

  this.DATA_FOLDER = folder
  if path.exists(this.DATA_FOLDER) == False: os.mkdir(this.DATA_FOLDER)
  if 'google.colab' in sys.modules:
    this.CONTENT_FOLDER = '/content'
  else:
    this.CONTENT_FOLDER = f'{this.DATA_FOLDER}/content'
    if path.exists(this.CONTENT_FOLDER) == False: os.mkdir(this.CONTENT_FOLDER)

############################ Function End #################################

############################ Function Start ################################
def set_result_folder(folder):
  from os import path
  this.RESULT_FOLDER = f'{this.DATA_FOLDER}/{folder}'
  if path.exists(this.RESULT_FOLDER) == False: os.mkdir(this.RESULT_FOLDER)

############################ Function End #################################

############################ Function Start #################################
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
############################ Function End #################################

############################ Function Start #################################
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
############################ Function End #################################


############################ Function Start #################################
def coleeg_info():
  info = """
* Coleeg is an open source initiative for collaborating on making a piece of software for investigating the use of Neural Networks in EEG signal classification on different datasets.

* License GPL V2.0

## Team:
Mahmoud Alnaanah (malnaanah@gmail.com)
Moutz Wahdow (m.wahdow@gmail.com)


## How to install Coleeg to your google drive

  1- Download Coleeg4.zip and extract it.
  2- Copy Coleeg4 folder to the root of your google drive.
  3- Open RunColeeg.ipynb which is inside Coleeg4 folder.
  4- Grant permissions for the notebook to enable its access your google drive.
  5- The data needed for Coleeg will be located in the directory ColeegData in the root of your google drive.
  6- To use Colab online, choose Connect>Connect to a hosted runtime.
  7- To use your local machine, copy Coleeg4 folder to your home (i.e. personal) folder.
  Local runtime was tested in Linux (Kubuntu 22.04)
  8- Run Colab_local script using the command (bash Colab_Local) and copy the generated url.
  9- Run RunColeeg.ipynb from google drive (not the the one on your home folder).
  10- Choose Connect>Connect to a local runtime, then paste the url and select connect.

## The required open dataset files are downloaded automatically.

  * Physionet link:
    https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip
  * BCI competetion IV-2a  links:
    http://www.bbci.de/competition/iv/#download
    http://www.bbci.de/competition/iv/results/#labels
  * TTK dataset is a private dataset and (up to now) not available publicly.
    

## What's new in Version 4.0
  * Adding CHB-MIT Scalp EEG Database.
  * Link: https://physionet.org/content/chbmit/1.0.0/
  * Some enhancements and bug fixes.

## What's new in Version 3.2
  * Some bugs are fixed
  * Dataset files are downloaded automatically.
  * Coleeg folder structure is changed. Coleeg code and data are in separate folders.
  * Specific python packages are installed in local runtime to guarantee compatibility.

## What's new in Version 3.0
  * The code is cleaner and more intuitive
  * Support of local runtime is added.
  * More metrics are supported and they can be selected during initialization, these metrics are:
    loss, accuracy, cohen_kappa, specificity, sensitivity
  * local timezone can be set.
  * A subset of classes and subjects can be selected for evaluation.
  * Plots of the resutls can be displayed and saved as pdf files in resutls/plots folder.
  * The result average value for last epochs can can be displayed and saved in the results folder.
  * The time for evaluation can be displayed and saved in the results folder
  * Validation can be done for dct and fft transforms of the origianl time signals.



## Links:
Human activity recognition (similar problem to EEG classification)
* https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/

Example on using CNN1D for reading thoughts 
* https://medium.com/@justlv/using-ai-to-read-your-thoughts-with-keras-and-an-eeg-sensor-167ace32e84a

Video classification approaches
* http://francescopochetti.com/video-classification-in-keras-a-couple-of-approaches/

MNE related links
* https://mne.tools/stable/auto_examples/time_frequency/plot_time_frequency_global_field_power.html?highlight=alpha%20beta%20gamma

* https://mne.tools/stable/auto_tutorials/preprocessing/plot_30_filtering_resampling.html#sphx-glr-auto-tutorials-preprocessing-plot-30-filtering-resampling-py

K-Fold cross-validation
* https://androidkt.com/k-fold-cross-validation-with-tensorflow-keras/

Fix memory leak in Colab
* https://github.com/tensorflow/tensorflow/issues/31312
* https://stackoverflow.com/questions/58137677/keras-model-training-memory-leak

Difference between Conv2D DepthWiseConv2D and SeparableConv2D
* https://amp.reddit.com/r/MLQuestions/comments/gp2pj9/what_is_depthwiseconv2d_and_separableconv2d_in/

Model difinition for EEGNet, ShallowConvNet, DeepConvNet was taken from the following link:
* https://github.com/rootskar/EEGMotorImagery


Customize Your Keras Metrics (how sensitivity and specificity are defined)
https://medium.com/@mostafa.m.ayoub/customize-your-keras-metrics-44ac2e2980bd

CHB-MIT Scalp EEG Database
https://physionet.org/content/chbmit/1.0.0/
  """
  print(info)
  return
############################ Function End #################################


############################ Function Start #################################
def get_data_ttk(Subjects = np.arange(1,26), Exclude=None, data_path=None,
                        Bands=None, resample_freq=None, Data_type=np.float32, tmin=0, tmax=2,Baseline=None, notch_freqs=None):

  if data_path == None:
    data_path=f'{this.DATA_FOLDER}/datasets/TTK/'

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_subject_index = np.zeros((len(Subjects),2)).astype(int) # starting data index for each subject and its number
  data_subject_index[:,1] = Subjects

  data_count = 0
  for sub_index, subject in enumerate(Subjects):
    print(f'\rLoading subject {sub_index+1}/{len(Subjects)}', end = '')
    data_subject_index[subject_count,0] = data_count
    subject_count+=1   
    if subject == 1:
      raw_fnames = [f'{data_path}subject{subject:02d}/rec01.vhdr', f'{data_path}subject{subject:02d}/rec02.vhdr', f'{data_path}subject{subject:02d}/rec03.vhdr']
      raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True, verbose=False) for f in raw_fnames])
    elif subject == 9:
      # rec02.vhdr is skipped because it is has a problem
      raw_fnames = [f'{data_path}subject{subject:02d}/rec01.vhdr']
      raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True, verbose=False) for f in raw_fnames])
    elif subject == 10:
      file_name = f'{data_path}subject{subject:02d}/rec01-uj.vhdr'
      raw = mne.io.read_raw_brainvision(file_name, preload=True, verbose=False)
    elif subject == 17:
      raw_fnames = [f'{data_path}subject{subject:02d}/rec01.vhdr', f'{data_path}subject{subject:02d}/rec02.vhdr']
      raw = mne.io.concatenate_raws([mne.io.read_raw_brainvision(f, preload=True, verbose=False) for f in raw_fnames])
    else:
      file_name = f'{data_path}subject{subject:02d}/rec01.vhdr'
      raw = mne.io.read_raw_brainvision(file_name, preload=True, verbose=False)

    raw.load_data(verbose=False) # needed for filteration
    if resample_freq is not None:
      raw.resample(resample_freq, npad="auto")
    if notch_freqs is not None:
        raw.notch_filter(notch_freqs)
    event_dict = {'Stimulus/S  1':1, 'Stimulus/S  5':2, 'Stimulus/S  7':3, 'Stimulus/S  9':4, 'Stimulus/S 11':5} 
    events, _ = mne.events_from_annotations(raw,event_id = event_dict, verbose=False)
    if Bands is not None:
      # Getting epochs for different frequncy bands   
      for band, (fmin, fmax) in enumerate(Bands):
        raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin',verbose=False)
        epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
        if not 'bands_data' in locals():
          D_SZ = epochs.get_data(copy=False).shape
          bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data[:,:,:,band] = epochs.get_data(copy=True).transpose(0,2,1) 
        del raw_bandpass
    else:
      epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
      bands_data = epochs.get_data(copy=True)
      SZ = bands_data.shape
      # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
      bands_data = bands_data.transpose(0,2,1).reshape(SZ[0],SZ[2],SZ[1],1)
    

    tmp_y = epochs.events[:,2]

    # Adjusting events numbers to be compatible with output classes numbers
    tmp_y = tmp_y - 1

    max_epochs = (284+65*4)
    SZ = bands_data.shape      
    # Creating output x and y matrices
    if not 'data_x' in locals():
      data_x = np.empty((max_epochs*len(Subjects),SZ[1],SZ[2],SZ[3]),dtype=Data_type)
      data_y = np.empty(max_epochs*len(Subjects),dtype=np.uint16)

    ## adjusting data type
    bands_data = bands_data.astype(Data_type)
    tmp_y = tmp_y.astype(np.uint16)

  
    data_x[data_count:data_count + SZ[0],:,:,:] = bands_data
    data_y[data_count:data_count + SZ[0]] = tmp_y

    data_count+=SZ[0]

    del raw, epochs, tmp_y, bands_data # saving memory 


  data_x = data_x[0:data_count,:,:,:]
  data_y = data_y[0:data_count]

  return data_x, data_y, data_subject_index

########################### Function End ####################################


############################ Function Start #################################
def get_data_chbmit(Subjects = np.arange(1,25), Exclude=None, data_path=None,Bands=None, tmin = 0, tmax =2,gap=None, max_epoc=None,  resample_freq=None, Data_type=np.float32,Baseline=None, notch_freqs=None):
  assert gap != None
  assert max_epoc != None

  period = int(tmax - tmin) # period in seconds

  included_data = get_chbmit_array(gap)

  selected_ch_names= ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']

  if data_path == None: data_path=f'{this.CONTENT_FOLDER}/physionet.org/files/chbmit/1.0.0/'

  # First subject in Subjects and Exclude has number = 1



  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
    #for i in sorted(Exclude, reverse=True):
      #del included_data[i-1]
  else:
    print('No subject excluded')

  #included_data = [included_data[i - 1] for i in Subjects]

  sampling_freq = resample_freq if resample_freq else 256

  # caculating number of epochs for the included data
  epoch_count = 0
  for subject in Subjects:
    for cls in range(len(included_data[subject - 1])):
      for ictal in range(len(included_data[subject - 1][cls])):
        ictal_length = included_data[subject-1][cls][ictal][2] - included_data[subject-1][cls][ictal][1]
        assert (ictal_length) >= period
        epoch_count += min(max_epoc,ictal_length//period)
        #epoch_count += max_epoc


  #print(f'Number of epochs = {epoch_count}')
  # epochs, samples, channels,bands
  data_x = np.empty((epoch_count,period*sampling_freq,len(selected_ch_names), len(Bands) if Bands else 1),dtype=Data_type)
  data_y = -1*np.ones(epoch_count,dtype=np.uint16)

  #print_prefix = '\n'

  data_count = 0
  subject_count = 0
  data_subject_index = np.zeros((len(Subjects),2)).astype(np.int32) # starting data index for each subject and its number
  data_subject_index[:,1] = Subjects



  #bands_data = None
  
  for subject in Subjects:

    print(f'Loading subject {subject}/{len(Subjects)}')
    # print(f'\rLoading subject [{subject}], Total subjects [{len(Subjects)}]', end = '')
    #print_prefix = '\n'

    data_subject_index[subject_count,0] = data_count
    subject_count+=1

    for cls in range(len(included_data[subject - 1])): # assuming ictal, preictal, and interictal lists have the same sizes
      skipped_file = ""
      last_file = ""
      for ictal in range(len(included_data[subject - 1][cls])):  # clases: 0 -> ictal, 1 -> preictal
        file_name = f'{data_path}chb{subject:02d}/{included_data[subject-1][cls][ictal][0]}'
        if file_name == skipped_file:
          continue
        if file_name != last_file:
          try:
            raw = read_raw(file_name, selected_ch_names)
            raw.load_data(verbose=False) # needed for filteration
            if resample_freq is not None:
              raw.resample(resample_freq, npad="auto")
            if notch_freqs is not None:
              raw.notch_filter(notch_freqs)

            if Bands is not None:
              # Getting epochs for frequncy bands
              for band, (fmin, fmax) in enumerate(Bands):
                raw_data = raw.copy().filter(fmin, fmax, fir_design='firwin',verbose=False).get_data()
                if band == 0:
                  SZ = raw_data.shape
                  # Swapping dimensions from (channel, sample) to (sample, channel)
                  bands_data = raw_data.transpose(1,0).reshape(SZ[1],SZ[0],1)
                else:
                  band_data = np.stack([bands_data, raw_data.transpose(1,0).reshape(SZ[1],SZ[0],1)], axis=-1)
            else:
              raw_data = raw.get_data()
              SZ = raw_data.shape
              # Swapping dimensions from (channel, sample) to (sample, channel)
              bands_data = raw_data.transpose(1,0).reshape(SZ[1],SZ[0],1)
          except Exception as e:
            print ("\n"+ f'Skipping [{file_name.split("/")[-1]}]  {e}')
            skipped_file = file_name
            continue

        #getting data
        ictal_length = included_data[subject-1][cls][ictal][2] - included_data[subject-1][cls][ictal][1]
        for i in range(0,min((ictal_length//period)*period,max_epoc*period), period):
          start = int((included_data[subject - 1][cls][ictal][1] + i)*sampling_freq)
          end = start + int(period*sampling_freq)
          data_x[data_count,:,:,:] = bands_data[start:end]
          data_y[data_count] = cls
          data_count+=1
        last_file = file_name

  data_x = data_x[0:data_count]
  data_y = data_y[0:data_count]

  return data_x, data_y, data_subject_index

########################### Function End ####################################

############################ Function Start #################################
def get_data_bcicomptIV2a(Subjects = np.arange(1,19), Exclude=None, data_path=None,Bands=None, resample_freq=None, Data_type=np.float32, tmin=0, tmax=4,Baseline=None, notch_freqs=None):

  if data_path == None:
    data_path=f'{this.CONTENT_FOLDER}/bcicomptIV2a/'

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_subject_index = np.zeros((len(Subjects),2)).astype(int) # starting data index for each subject and its number
  data_subject_index[:,1] = Subjects
  data_count = 0
  for sub_index,subject in enumerate(Subjects):
    print(f'\rLoading subject {sub_index+1}/{len(Subjects)}', end = '')
    data_subject_index[subject_count,0] = data_count
    subject_count+=1

    if subject in range(1,10):
      dataset_type = 'T'
      sub =subject
      event_dict = {'769':1, '770':2, '771':3, '772':4}
    else:
      dataset_type = 'E'
      sub =subject - 9
      event_dict= {'783':1}

    file_name = f'{data_path}A{sub:02d}{dataset_type}.gdf'
    raw = mne.io.read_raw_gdf(file_name,verbose='ERROR')
    raw.load_data(verbose=False) # needed for filteration
    if resample_freq is not None:
      raw.resample(resample_freq, npad="auto")

    if notch_freqs is not None:
      raw.notch_filter(notch_freqs)

    events, _ = mne.events_from_annotations(raw,event_id = event_dict, verbose=False)
    picks = mne.pick_channels_regexp(raw.ch_names, regexp=r'EEG*')
    if Bands is not None:
      # Getting epochs for different frequncy bands   
      for band, (fmin, fmax) in enumerate(Bands):
        raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin',verbose=False)
        epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, picks=picks, event_repeated = 'drop',verbose=False)
        if not 'bands_data' in locals():
          D_SZ = epochs.get_data(copy=False).shape
          bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data[:,:,:,band] = epochs.get_data(copy=True).transpose(0,2,1) 
        del raw_bandpass
    else:
      epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, picks=picks, event_repeated = 'drop',verbose=False)
      bands_data = epochs.get_data(copy=True)
      SZ = bands_data.shape
      # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
      bands_data = bands_data.transpose(0,2,1).reshape(SZ[0],SZ[2],SZ[1],1)
    
    # reading event type from .mat files 
    mat_vals = scipy.io.loadmat(f'{data_path}A{sub:02d}{dataset_type}.mat')
    tmp_y = mat_vals['classlabel'].flatten()


    # Adjusting events numbers to be compatible with output classes numbers
    tmp_y = tmp_y - 1

    max_epochs = 288
    SZ = bands_data.shape      
    # Creating output x and y matrices
    if not 'data_x' in locals():
      data_x = np.empty((max_epochs*len(Subjects),SZ[1],SZ[2],SZ[3]),dtype=Data_type)
      data_y = np.empty(max_epochs*len(Subjects),dtype=np.uint16)

    ## adjusting data type
    bands_data = bands_data.astype(Data_type)
    tmp_y = tmp_y.astype(np.uint16)

  
    data_x[data_count:data_count + SZ[0],:,:,:] = bands_data
    data_y[data_count:data_count + SZ[0]] = tmp_y
      
    data_count+=SZ[0]

    del raw, epochs, tmp_y, bands_data # saving memory 

  # data_x = data_x[0:data_count,:,:,:]
  # data_y = data_y[0:data_count]


  return data_x, data_y, data_subject_index

########################### Function End ####################################


############################ Function Start #################################
def get_data_physionet(Subjects=np.arange(1,110), Exclude=np.array([88,89,92,100,104]), data_path=None,Bands=None, resample_freq=None, Total=100, Data_type=np.float32, tmin=0, tmax=2,Tasks=np.array([[3,7,11],[5,9,13]]),  Baseline=None, notch_freqs=None):


  if data_path == None:
    data_path=f'{this.CONTENT_FOLDER}/physionet/'

  # First subject in Subjects and Exclude has number = 1

  if Exclude is not None:
    print('Excluded Subjects are '+ str(Exclude))
    Subjects = np.delete(Subjects,np.isin(Subjects,Exclude))
  else:
    print('No subject excluded')

  subject_count = 0
  data_subject_index = np.zeros((len(Subjects),2)).astype(int) # starting data index for each subject and its number
  data_subject_index[:,1] = Subjects
  data_count = 0
  for sub_index, subject in enumerate(Subjects):
    print(f'\rLoading subject {sub_index+1}/{len(Subjects)}', end = '')
    data_subject_index[subject_count,0] = data_count
    subject_count+=1
    for run in Tasks.flatten():
      file_name = f'{data_path}/files/S{subject:03d}/S{subject:03d}R{run:02d}.edf'
      raw = mne.io.read_raw_edf(file_name,verbose=False)
      raw.load_data(verbose=False) # needed for filteration
      if resample_freq is not None:
        raw.resample(resample_freq, npad="auto")
      if notch_freqs is not None:
        raw.notch_filter(notch_freqs)
      events, event_dict = mne.events_from_annotations(raw,verbose=False)
      if Bands is not None:
        # Getting epochs for different frequncy bands   
        for band, (fmin, fmax) in enumerate(Bands):
          raw_bandpass = raw.copy().filter(fmin, fmax, fir_design='firwin',verbose=False)
          epochs = mne.Epochs(raw_bandpass, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
          if not 'bands_data' in locals():
            D_SZ = epochs.get_data(copy=False).shape
            bands_data = np.empty((D_SZ[0],D_SZ[2],D_SZ[1],len(Bands)))
          # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
          bands_data[:,:,:,band] = epochs.get_data(copy=True).transpose(0,2,1) 
          del raw_bandpass
      else:
        epochs = mne.Epochs(raw, events, event_id=event_dict,baseline=Baseline, tmin=tmin, tmax=tmax, preload=True, event_repeated = 'drop',verbose=False)
        bands_data = epochs.get_data(copy=True)
        SZ = bands_data.shape
        # Swapping dimensions from (epoch, channel, sample) to (epoch, sample, channel)
        bands_data = bands_data.transpose(0,2,1).reshape(SZ[0],SZ[2],SZ[1],1)
      tmp_y = epochs.events[:,2]

      # Adjusting events numbers to be compatible with output classes numbers
      if run in Tasks[0]:
        tmp_y = tmp_y - 1
      elif run in Tasks[1]:
        tmp_y = tmp_y - 1
        tmp_y[tmp_y==1]=3
        tmp_y[tmp_y==2]=4

      SZ = bands_data.shape

      # Creating output x matrix
      if not 'data_x' in locals():
        file_count = 6
        max_epochs = 30
        data_x = np.empty((max_epochs*len(Subjects)*file_count,SZ[1],SZ[2],SZ[3]),dtype=Data_type)
        data_y = np.empty((max_epochs*len(Subjects)*file_count),dtype=np.uint16)

      ## adjusting data type
      bands_data = bands_data.astype(Data_type)
      tmp_y = tmp_y.astype(np.uint16)


      data_x[data_count:data_count + SZ[0],:,:,:] = bands_data
      data_y[data_count:data_count + SZ[0]] = tmp_y
      data_count+=SZ[0]
      del raw, epochs, tmp_y, bands_data # saving memory 
      
      

  data_x = data_x[0:data_count,:,:,:]
  data_y = data_y[0:data_count]

  # removing samples with odd output
  min_cat = 0
  max_cat = 4

  idx = np.flatnonzero(np.logical_or(data_y > max_cat,data_y < min_cat))
  data_x = np.delete(data_x,idx, axis=0)
  data_y = np.delete(data_y,idx)
  del idx

  return data_x, data_y, data_subject_index

########################### Function End ####################################

########################### Function Start ##################################
def get_pos_map(dataset):
  if dataset=='bcicomptIV2a':
    pos_map = np.array([
      [-1,-1,-1, 1,-1,-1,-1], 
      [-1, 2, 3, 4, 5, 6,-1], 
      [ 7, 8, 9,10,11,12,13], 
      [-1,14,15,16,17,18,-1], 
      [-1,-1,19,20,21,-1,-1], 
      [-1,-1,-1,22,-1,-1,-1]])
  elif dataset=='physionet':
    pos_map = np.array([
      [-1,-1,-1,-1,22,23,24,-1,-1,-1,-1],
      [-1,-1,-1,25,26,27,28,29,-1,-1,-1], 
      [-1,30,31,32,33,34,35,36,37,38,-1], 
      [-1,39, 1, 2, 3, 4, 5, 6, 7,40,-1], 
      [43,41, 8, 9,10,11,12,13,14,42,44], 
      [-1,45,15,16,17,18,19,20,21,46,-1], 
      [-1,47,48,49,50,51,52,53,54,55,-1], 
      [-1,-1,-1,56,57,58,59,60,-1,-1,-1], 
      [-1,-1,-1,-1,61,62,63,-1,-1,-1,-1],
      [-1,-1,-1,-1,-1,64,-1,-1,-1,-1,-1]])
  elif dataset=='ttk':
    pos_map = np.array([
      [-1,-1,-1,-1, 1,-1,31,-1,-1,-1,-1],
      [-1,-1,-1,32,33,34,61,60,-1,-1,-1],
      [-1, 4,36, 3,35, 2,62,29,59,30,-1],
      [ 5,37, 6,38, 7,63,28,57,27,58,26],
      [-1, 9,40, 8,39,-1,56,24,55,25,-1],
      [10,41,11,42,12,52,23,53,22,54,21],
      [-1,15,44,14,43,13,51,19,50,20,-1],
      [-1,-1,-1,45,46,47,48,49,-1,-1,-1],
      [-1,-1,-1,-1,16,17,18,-1,-1,-1,-1]])
  elif dataset=='chbmit':
    raise Exception("Position matrix not yet implemented for Chbmit dataset")
  else:
    sys.exit("Position map not defined for this dataset.")
  return pos_map
########################### Function End  ###################################

########################### Function Start ##################################

def make_into_2d(data_1d,pos_map):
  Map = pos_map
  map_sz = Map.shape
  SZ = data_1d.shape # (epochs, time samples, eeg channels, bands)
  Map = Map.flatten()
  idx = np.arange(Map.shape[0])
  idx = idx[Map > 0]
  Map = Map[Map > 0]
  Map = Map -1 # adjusting index to start from 0
  data_2d = np.zeros((SZ[0],SZ[1], map_sz[0]*map_sz[1], SZ[3]),dtype=data_1d.dtype)
  
  data_2d[:,:,idx,:] = data_1d[:,:,Map,:]
  data_2d = data_2d.reshape(SZ[0],SZ[1], map_sz[0],map_sz[1], SZ[3])
  return data_2d
########################### Function End ####################################


############################ Function Start #################################
def unzip_dataset(dataset):
  dataset_folder=f'{this.DATA_FOLDER}/datasets'
  if os.path.exists(dataset_folder) == False: os.mkdir(dataset_folder)

  if dataset=='physionet':
    zipfile_list = [f'{this.DATA_FOLDER}/datasets/eeg-motor-movementimagery-dataset-1.0.0.zip']
    zipfile_url_list = ['https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip']

    output_path = f'{this.CONTENT_FOLDER}/physionet/'
    
  elif dataset=='bcicomptIV2a':
    zipfile_list = [f'{this.DATA_FOLDER}/datasets/BCICIV_2a_gdf.zip', f'{this.DATA_FOLDER}/datasets/true_labels.zip']
    zipfile_url_list = ['https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip', 'https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip']
    output_path = f'{this.CONTENT_FOLDER}/bcicomptIV2a/'
  elif dataset=='ttk':
    print(f'Dataset ttk has no zip files')
    return
  elif dataset=='chbmit':
    print(f'Dataset chbmit has no zip files.  Data files are downloaded automatically as needed.')
    return
  else:
    raise ValueError('Unknown dataset')

  if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
    print('Data already exists.')
    return

  for zipfile,url in zip(zipfile_list,zipfile_url_list):
    get_ipython().system(f'wget -c -q --show-progress {url} -P {dataset_folder}')


  
  print ('Unzipping data.')
  for zipfile in zipfile_list:
    get_ipython().system('unzip -qq ' + zipfile +' -d ' + output_path)
  print ('Unzipping done.')  
  return
########################### Function End ####################################


########################### Function Start ##################################
def balance(data_x, data_y, data_subject_index):
  # make number of trial for each subject equal to the minimum per class
  data_subject_index = data_subject_index.copy() # to avoid changing the original array
  Classes, counts = (),()
  end=len(data_y)
  for sub_index in reversed(range(len(data_subject_index))):
    start=data_subject_index[sub_index,0]
    val, cnt = np.unique(data_y[start:end], return_counts=True)
    Classes=(val,*Classes)
    counts = (cnt,*counts)
    end=start
  MIN = np.array(counts).min(axis=1)

  for Class in Classes[0]:
    index=np.flatnonzero(data_y==Class)
    end=len(data_y)
    index1=np.array([]).astype(int)
    for sub_index in reversed(range(len(data_subject_index))):
      start=int(data_subject_index[sub_index,0])
      diff = int(counts[sub_index][Classes[sub_index]==Class]-MIN[sub_index])
      index1 = np.append(index[index>=start][0:diff], index1)
      data_subject_index[sub_index+1:,0] -= diff
      end=start
    
    data_x = np.delete(data_x,index1,axis=0)
    data_y = np.delete(data_y,index1,axis=0)

    print ('Balancing data done.')

    return data_x, data_y, data_subject_index
############################ Function End ###################################


############################ Function Start #################################
def normalize(data_x):
  """
  Normalize the data (for each band) to have zero mean and unity standard deviation
  works in place
  """
  SZ = data_x.shape
  for i in range(SZ[-1]):
    if len(SZ) == 4:
      mean = np.mean(data_x[:,:,:,i])
      std = np.std(data_x[:,:,:,i])
      data_x[:,:,:,i] -= mean
      data_x[:,:,:,i] /= std
    elif len(SZ) == 5:
      mean = np.mean(data_x[:,:,:,:,i])
      std = np.std(data_x[:,:,:,:,i])
      data_x[:,:,:,:,i] -= mean
      data_x[:,:,:,:,i] /= std
    else:
      raise ValueError('data_x has unexpected size')
      
  print ('Normalizing data done.')
  return

############################ Function End ###################################


############################ Function Start #################################
def video_array(data_x, data_y, Class=0, Band=0,Rows=4, Cols=5, Seed=100):

  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
  from IPython.display import HTML

  fig = plt.figure()
  ims = []
  index = np.flatnonzero(data_y == Class)
  np.random.seed(Seed)
  np.random.shuffle(index)
  index = index[:Rows*Cols]
  print('samples = ' +str(index))
  total_frames = data_x.shape[1]
  for frame in range(total_frames):
    img_count = 0
    img = data_x[index[img_count],frame,:,:,Band]
    img_count+=1
    for col in range(1,Cols):
      img = np.append(img,data_x[index[img_count],frame,:,:,Band],axis=1)
      img_count+=1
    for row in range(1,Rows):
      img_col = data_x[index[img_count],frame,:,:,Band]
      img_count+=1
      for col in range(1,Cols):
        img_col = np.append(img_col,data_x[index[img_count],frame,:,:,Band],axis=1)
        img_count+=1
      img = np.append(img,img_col,axis=0)

    im = plt.imshow(img, animated=True)
    plt.clim(-2, 2)  
    ims.append([im])

  plt.colorbar()
  plt.xticks([])
  plt.yticks([])
  plt.close()
  ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
  display(HTML(ani.to_html5_video()))
  return
############################# Function End ##################################

############################ Function Start #################################
def keep_classes(data_x, data_y, data_subject_index, selected_classes):
  selected_index = np.isin(data_y,selected_classes)
  data_y_out = data_y[selected_index]
  data_x_out = data_x[selected_index]
  data_subject_index_out = data_subject_index.copy()
  for subject in range(1,len(data_subject_index)):
    start=data_subject_index[subject-1,0]
    end=data_subject_index[subject,0]
    data_subject_index_out[subject:,0] = data_subject_index_out[subject:,0] - np.count_nonzero(~selected_index[start:end])

  return data_x_out, data_y_out, data_subject_index_out
############################ Function End ###################################

############################ Function Start #################################
def keep_subjects(data_x, data_y, data_subject_index, selected_subjects):

  Subjects = list(data_subject_index[:,1])
  data_subject_index_out = data_subject_index.copy()
  selected_sub_indices = np.isin(Subjects,selected_subjects)
  selected_data_indices = np.ones(len(data_x), dtype=bool)

  for subject in range(len(Subjects)):
    if not selected_sub_indices[subject]:
      start = data_subject_index_out[subject,0]
      if subject == len(Subjects) - 1:
        end = len(data_x)
      else:
        end = data_subject_index_out[subject+1,0]
      data_subject_index_out[subject:,0] = data_subject_index_out[subject:,0] - (end-start)
      selected_data_indices[start:end] = False

  data_y_out = data_y[selected_data_indices]
  data_x_out = data_x[selected_data_indices]
  data_subject_index_out = data_subject_index_out[selected_sub_indices]
  return data_x_out, data_y_out, data_subject_index_out
############################ Function End ###################################

############################ Function Start #################################
def shuffle_subjects(data_x,data_y,data_subject_index,seed=100):

  data_subject_index = data_subject_index.copy()

  data_length = np.append(data_subject_index[1:,0],data_x.shape[0])  - data_subject_index[:,0]


  np.random.seed(seed)
  np.random.shuffle(data_length)

  np.random.seed(seed)
  np.random.shuffle(data_subject_index)

  data_x_out = np.zeros(data_x.shape)
  data_y_out = np.zeros(data_y.shape)

  start_out = 0
  for c  in range(len(data_subject_index)):
    end_out = start_out + data_length[c]
    start_in = data_subject_index[c,0]
    end_in = start_in + data_length[c]
    data_x_out[start_out:end_out] = data_x[start_in:end_in]
    data_y_out[start_out:end_out] = data_y[start_in:end_in]

    data_subject_index[c,0] = start_out
    start_out = end_out
  
  return data_x, data_y, data_subject_index
############################ Function End ###################################

############################ Function Start #################################
def build_model(data_x, data_yc, model_type='CNN1D', show_summary=True, batch_norm=False,apply_spectral=False,dropout_rate=0.5,Max_norm=None):

  if isinstance(data_x,list):
    input_list_SZ = len(data_x)
    data_x = data_x[0]
  else:
    input_list_SZ = 1

    
  SZ = data_x.shape

  if len(SZ)== 4:
    x_dim = SZ[2]
    y_dim = 1
    n_bands = SZ[3]
  elif len(SZ)==5:
    x_dim = SZ[2]
    y_dim = SZ[3]
    n_bands = SZ[4]

  n_timesteps, n_outputs = SZ[1], data_yc.shape[1]

  # update metrics
  set_metrics(this.METRICS_TO_SAVE,n_outputs=n_outputs)




  if model_type=='EEGNet':
    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    kernLength=64
    F1=8
    D=2
    F2=16
    dropoutType=Dropout
    #dropoutType=SpatialDropout2D # another option for dropout
    norm_rate=.25
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (kernLength, 1)
    depth_filters = (1, Chans)
    pool_size = (6, 1)
    pool_size2 = (12, 1)
    separable_filters = (20, 1)
    axis = -1

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same', input_shape=input_shape,use_bias=False)(input1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters, use_bias=False, padding='same')(block1)
    if batch_norm: block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    model = Model(inputs=input1, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
    if show_summary:
      model.summary()
    return model


  if model_type=='ShallowConvNet':
    ########### need these for ShallowConvNet
    def square(x):
        return K.square(x)
    def log(x):
        return K.log(K.clip(x, min_value=1e-7, max_value=10000))
    ######################################

    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    norm_rate=0.5

    
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (25, 1)
    conv_filters2 = (1, Chans)
    pool_size = (45, 1)
    strides = (15, 1)
    axis = -1


    input_main = Input(input_shape)
    block1 = Conv2D(20, conv_filters, input_shape=input_shape, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(20, conv_filters2, use_bias=False, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=pool_size, strides=strides)(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
    if show_summary:
      model.summary()
    return model

  if model_type=='DeepConvNet':
    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    norm_rate=0.5
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (2, 1)
    conv_filters2 = (1, Chans)
    pool = (2, 1)
    strides = (2, 1)
    axis = -1


    # start the model
    input_main = Input(input_shape)
    block1 = Conv2D(25, conv_filters, input_shape=input_shape, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, conv_filters2, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=pool, strides=strides)(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, conv_filters, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    if batch_norm: block2 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=pool, strides=strides)(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, conv_filters, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    if batch_norm: block3 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=pool, strides=strides)(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, conv_filters, kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    if batch_norm: block4 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=pool, strides=strides)(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    model = Model(inputs=input_main, outputs=softmax)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
    if show_summary:
      model.summary()
    return model


  if model_type=='EEGNet_fusion':
    """
    The author of this model is Karel Roots and was published along with the paper titled 
    "Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification"
    """

    nb_classes = n_outputs
    Chans=x_dim*y_dim
    Samples=n_timesteps
    dropoutRate=0.5
    dropoutType=Dropout
    #dropoutType=SpatialDropout2D # another option for dropout
    norm_rate=0.25
    input_shape = (Samples, Chans, n_bands)
    conv_filters = (64, 1)
    conv_filters2 = (96, 1)
    conv_filters3 = (128, 1)    
    #depth_filters = (1, Chans)
    depth_filters = (n_bands, Chans) # made improvement
    pool_size = (4, 1)
    pool_size2 = (8, 1)
    separable_filters = (8, 1)
    separable_filters2 = (16, 1)
    separable_filters3 = (32, 1)
    axis = -1

    F1 = 8
    F1_2 = 16
    F1_3 = 32
    F2 = 16
    F2_2 = 32
    F2_3 = 64
    D = 2
    D2 = 2
    D3 = 2

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same', input_shape=input_shape, use_bias=False)(input1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    if batch_norm: block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters, use_bias=False, padding='same')(block1)  # 8
    if batch_norm: block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)
    block2 = Flatten()(block2)  # 13

    # 8 - 13

    input2 = Input(shape=input_shape)
    block3 = Conv2D(F1_2, conv_filters2, padding='same', input_shape=input_shape, use_bias=False)(input2)
    if batch_norm: block3 = BatchNormalization(axis=axis)(block3)
    block3 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D2, depthwise_constraint=max_norm(1.))(block3)
    if batch_norm: block3 = BatchNormalization(axis=axis)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D(pool_size)(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2_2, separable_filters2, use_bias=False, padding='same')(block3)  # 22
    if batch_norm: block4 = BatchNormalization(axis=axis)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D(pool_size2)(block4)
    block4 = dropoutType(dropoutRate)(block4)
    block4 = Flatten()(block4)  # 27
    # 22 - 27

    input3 = Input(shape=input_shape)
    block5 = Conv2D(F1_3, conv_filters3, padding='same', input_shape=input_shape, use_bias=False)(input3)
    if batch_norm: block5 = BatchNormalization(axis=axis)(block5)
    block5 = DepthwiseConv2D(depth_filters, use_bias=False, depth_multiplier=D3, depthwise_constraint=max_norm(1.))(block5)
    if batch_norm: block5 = BatchNormalization(axis=axis)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D(pool_size)(block5)
    block5 = dropoutType(dropoutRate)(block5)

    block6 = SeparableConv2D(F2_3, separable_filters3, use_bias=False, padding='same')(block5)  # 36
    if batch_norm: block6 = BatchNormalization(axis=axis)(block6)
    block6 = Activation('elu')(block6)
    block6 = AveragePooling2D(pool_size2)(block6)
    block6 = dropoutType(dropoutRate)(block6)
    block6 = Flatten()(block6)  # 41

    # 36 - 41

    merge_one = concatenate([block2, block4])
    merge_two = concatenate([merge_one, block6])

    flatten = Flatten()(merge_two)

    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.25))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    model= Model(inputs=[input1, input2, input3], outputs=softmax)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
    if show_summary:
      model.summary()
    return model

  if model_type=='CNN1D_MFBF':

    input_set=[]
    block_set=[]
    dropoutRate = dropout_rate
    
    for band in range(input_list_SZ):
      input_set.append(Input(shape=(n_timesteps, x_dim*y_dim)))
      #block1 = Lambda(lambda x: x[:,:,:,band])(input_set[band])
      
      block1 = AveragePooling1D(pool_size=(2))(input_set[band])
      block1 = Conv1D(50,5,padding='same')(block1)
      block1 = Activation('elu')(block1) # try elu instead of relu
      block1 = Dropout(dropoutRate)(block1)
      
      block1 = AveragePooling1D(pool_size=(2))(block1)
      block1 = Conv1D(50,5,padding='same')(block1)
      block1 = Activation('elu')(block1) # try elu instead of relu
      block1 = Dropout(dropoutRate)(block1)
      
      block1 = AveragePooling1D(pool_size=(2))(block1)
      block1 = Conv1D(50,5,padding='same')(block1)
      block1 = Activation('elu')(block1) # try elu instead of relu
      block1 = Dropout(dropoutRate)(block1)

      block1 = AveragePooling1D(pool_size=(2))(block1)
      block1 = Dropout(dropoutRate)(block1)
      block_set.append(Flatten()(block1))

    # merging models  
    merge_block  = concatenate(block_set)
    flatten = Flatten()(merge_block)
    dense = Dense(n_outputs, name='dense', kernel_constraint=max_norm(Max_norm))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    model= Model(inputs=input_set, outputs=softmax)

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
    if show_summary:
      model.summary()
    return model

  # Adding separate model.add(Activation(activations.relu)) line
  # helps with memeory leak
  # see link: https://github.com/tensorflow/tensorflow/issues/31312

  model = Sequential()
  if model_type=='Basic':
    model.add(InputLayer(input_shape=(n_timesteps, x_dim * y_dim, n_bands)))
    model.add(Flatten())
  elif model_type=='CNN1D':
    time_axis = -1
    model.add(InputLayer(input_shape=(n_timesteps, x_dim * y_dim, n_bands)))
    model.add(Reshape((n_timesteps, x_dim*y_dim*n_bands)) )
    # model.add(Permute((2,1), input_shape=(n_timesteps, x_dim * y_dim * n_bands)))
    if batch_norm: model.add(BatchNormalization(axis=time_axis, epsilon=1e-05, momentum=0.1))
    model.add(AveragePooling1D(pool_size=(5)))
    
    model.add(Conv1D(50, 5,  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling1D(pool_size=(2)))
    model.add(Conv1D(50, 5, padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling1D(pool_size=(2)))
    model.add(Conv1D(50, 5, padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(100, activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))

  elif model_type=='CNN2D':
    model.add(InputLayer(input_shape=(n_timesteps, x_dim , y_dim, n_bands)))
    model.add(Reshape((n_timesteps, x_dim*y_dim,n_bands)) )
    model.add(AveragePooling2D(pool_size=(5,1)))
    model.add(Conv2D(50, (5,5),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling2D(pool_size=(2,1)))
    model.add(Conv2D(50, (5,5),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling2D(pool_size=(2,1)))
    model.add(Conv2D(50, (5,2),  padding='same', activation=None))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100, activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
  elif model_type=='CNN3D':
    model.add(InputLayer(input_shape=(n_timesteps, x_dim , y_dim, n_bands)))
    model.add(AveragePooling3D(pool_size=(2,1,1)))
    model.add(Conv3D(50, (5,2,2),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling3D(pool_size=(2,1,1)))
    model.add(Conv3D(50, (5,2,2),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(AveragePooling3D(pool_size=(2,1,1)))
    model.add(Conv3D(50, (5,2,2),  padding='same', activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100, activation=None))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
  elif model_type=='TimeDist': # Time distributed
    model.add(InputLayer(input_shape=(n_timesteps, x_dim , y_dim, n_bands)))
    model.add(AveragePooling3D(pool_size=(5,1,1))) 
    model.add(TimeDistributed(Conv2D(50, (3, 3), strides=(1, 1), activation=None, padding='same')))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Conv2D(50, (2, 2), strides=(1, 1), activation=None, padding='same')))
    model.add(Activation(activations.relu))
    model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(dropout_rate))
  
  model.add(Flatten()) # added in Version 3.2 to enhance model validation accuracy
  model.add(Dense(n_outputs, activation='softmax',kernel_constraint=max_norm(Max_norm)))
  model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=this.METRICS)
  if show_summary:
    model.summary()

  return model
######################### Function End ######################################


######################### Function Start ####################################
def split_data(data, Parts=5, part_index=0, model_type=None, shuffle=False, shuffle_seed=100):
  # specify model_type when spliting x data to adjust the output according the the model

  data=data.copy()

  if shuffle:
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)
    
  SZ = data.shape
  part_length = np.floor(SZ[0]/Parts).astype(int)
  start = part_length*part_index
  end = start + part_length
  if part_index == Parts-1:
    end = SZ[0]



  index = np.ones(data.shape[0],dtype=bool)
  index[start:end] = False

  train = data[index]
  index = np.invert(index)
  test = data[index]

  # adjusting output depending on the model

  if model_type=='EEGNet_fusion':
    train=[train, train, train]
    test=[test, test, test]

  if model_type=='CNN1D_MFBF':
    Bands=data[0].shape[-1]
    train=[np.expand_dims(train[:,:,:,i],axis=3) for i in range(Bands)]
    test=[np.expand_dims(test[:,:,:,i],axis=3) for i in range(Bands)]


  return train, test

########################## Function End #####################################

######################### Function Start ####################################
def split_subjects(data,data_subject_index,test_subject_index, model_type=None):

  test_subject_index = np.array([test_subject_index]).flatten()
  test_index = np.zeros(len(data), dtype=bool)
  
  for sub_index in test_subject_index:
    start = data_subject_index[sub_index,0]
    if sub_index == len(data_subject_index)-1:
      end=None
    else:
      end=data_subject_index[sub_index+1,0]
    test_index[start:end] = True 

  test = data[test_index]
  train = data[np.invert(test_index)]

  # adjusting output depending on the model

  if model_type=='EEGNet_fusion':
    train=[train, train, train]
    test=[test, test, test]

  if model_type=='CNN1D_MFBF':
    Bands=data[0].shape[-1]
    train=[np.expand_dims(train[:,:,:,i],axis=3) for i in range(Bands)]
    test=[np.expand_dims(test[:,:,:,i],axis=3) for i in range(Bands)]
  return train, test
########################## Function End #####################################


######################### Function Start ####################################
def evaluate_model(model_list,dataset,Bands,data_x, data_y, data_subject_index,fold_num,Folds, epochs, batch_size=64, verbose=0, show_summary = False,batch_norm=True, apply_spectral=False,dropout_rate=0.5, align_to_subject=True, selected_subjects=False,selected_classes=False, shuffle=True, play_audio=False, Max_norm=None):
  tz = pytz.timezone(this.TIME_ZONE)
  metric_results = {}
  time_results = {}
  kappa_results={}


  if selected_classes:
    data_x, data_y, data_subject_index = keep_classes(data_x, data_y, data_subject_index, selected_classes)

  if selected_subjects:
    data_x, data_y, data_subject_index = keep_subjects(data_x, data_y, data_subject_index, selected_subjects)

  if apply_spectral == 'dct':
    data_x = dct_1d(data_x)
    print('dct applied')
    normalize(data_x)

  if apply_spectral == 'fft':
    data_x = fft_1d(data_x)
    print('fft applied')
    normalize(data_x)



  if align_to_subject:
    subject_results = {}
  else:
    subject_results = None
  for model_type in model_list:
    print(f'Starting evaluation for {model_type} model at {datetime.now(tz).strftime("%H:%M:%S")}. Dataset: {dataset}, Bands: {"off" if not Bands else "on"}, Folds: {list(Folds)}')
    if fold_num > data_subject_index.shape[0] and align_to_subject:
      raise ValueError('Number of folds should be less than number of subjects.')
    elif fold_num>data_x.shape[0]:
      raise ValueError('Number of folds should be less than number of trials.')

    if model_type in ['CNN3D', 'TimeDist']: # these model requires 2D mapped data
      # Generating 2D mapped data
      pos_map = get_pos_map(dataset) # positions for 2D map conversion
      data_input = make_into_2d(data_x,pos_map)
    else:
      data_input = data_x

    if align_to_subject:# distributing subjects on folds in a fair way
      N = data_subject_index.shape[0]
      sub_part = np.ones(fold_num)*np.floor(N/fold_num)
      sub_part[0:(N % fold_num)] +=1
      sub_part_index = np.append(0,np.cumsum(sub_part)).astype(int)

    metric_data=[]
    time_data=[]
    kappa_data=[]

    if align_to_subject:
      subject_data=[]

    for fold in Folds:
      start_time = time.time()
      # Splitting data 
      if align_to_subject:
        sub_test_index = np.arange(sub_part_index[fold],sub_part_index[fold+1])
        train_x, test_x = split_subjects(data_x,data_subject_index,sub_test_index,model_type=model_type)
        train_y, test_y = split_subjects(data_y,data_subject_index,sub_test_index,model_type=None)
        print(f'Training: fold [{fold}], validation subjects {data_subject_index[sub_test_index,1]}')
      else:
        train_x, test_x = split_data(data_input, Parts=fold_num, part_index=fold, shuffle=shuffle,model_type=model_type)
        train_y, test_y = split_data(data_y, Parts=fold_num, part_index=fold, shuffle=True,model_type=None)
      # Generating categorical data
      train_yc = to_categorical(train_y)
      test_yc = to_categorical(test_y)
      num_classes = train_yc.shape[1]


      # Building and validating  model
      model = build_model(train_x, train_yc, model_type=model_type,show_summary=show_summary, batch_norm=batch_norm ,apply_spectral=apply_spectral,dropout_rate=dropout_rate)    
      model.fit(train_x, train_yc, epochs=epochs, batch_size=batch_size, verbose=verbose,  validation_data=(test_x, test_yc))

      # fixing memory leak
      K.clear_session()
      tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph
      gc.collect()


      metric_data.append(model.history.history)
      fold_time = time.time() - start_time
      time_data.append(fold_time)
      if align_to_subject:
        subject_data.append(data_subject_index[sub_test_index,1])

      # saving partial results
      metric_results[model_type]=metric_data
      time_results[model_type]=time_data
      if align_to_subject:
        subject_results[model_type] = subject_data
        save_results(metric_results,time_results,dataset, Bands, subject_results=subject_results, model_type=model_type,partial=True)
      else:
        save_results(metric_results,time_results,dataset, Bands,subject_results=None, model_type=model_type, partial=True)




      print(f'Fold {fold} done in {timedelta(seconds=round(fold_time))} ')
    
    # saving final resutls for model
    metric_results[model_type]=metric_data
    time_results[model_type]=time_data
    if align_to_subject:
      subject_results[model_type] = subject_data
      save_results(metric_results,time_results, dataset, Bands,subject_results=subject_results,model_type=model_type)
    else:
      save_results(metric_results,time_results, dataset, Bands,subject_results=None,model_type=model_type)






  

  # Play an audio to alert for finishing
  if play_audio:
    from google.colab import output
    output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg").play()')

  return model, metric_results, time_results,  (subject_results if align_to_subject else None)

########################## Function End #####################################

######################### Function Start ###################################
def predict_model(model,model_type, dataset, data_x, data_y,data_subject_index,fold_num,Folds, align_to_subject=True, selected_subjects=False, selected_classes=False, shuffle=True):

  if fold_num > data_subject_index.shape[0] and align_to_subject:
    raise ValueError('Number of folds should be less than number of subjects.')
  elif fold_num>data_x.shape[0]:
    raise ValueError('Number of folds should be less than number of trials.')

  if selected_classes:
    data_x, data_y, data_subject_index = keep_classes(data_x, data_y, data_subject_index, selected_classes)
  if selected_subjects:
    data_x, data_y, data_subject_index = keep_subjects(data_x, data_y, data_subject_index, selected_subjects)




  if model_type in ['CNN3D', 'TimeDist']: # these model requires 2D mapped data
    # Generating 2D mapped data
    pos_map = get_pos_map(dataset) # positions for 2D map conversion
    data_input = make_into_2d(data_x,pos_map)
  else:
    data_input = data_x

  if align_to_subject:# distributing subjects on folds in a fair way
    N = data_subject_index.shape[0]
    sub_part = np.ones(fold_num)*np.floor(N/fold_num)
    sub_part[0:(N % fold_num)] +=1
    sub_part_index = np.append(0,np.cumsum(sub_part)).astype(int)

  for fold in Folds:
    print(f'Prediction accuracy for fold {fold}')
    # Splitting data 
    if align_to_subject:
      sub_test_index = np.arange(sub_part_index[fold],sub_part_index[fold+1])
      _, test_x = split_subjects(data_x,data_subject_index,sub_test_index,model_type=model_type)
      _, test_y = split_subjects(data_y,data_subject_index,sub_test_index,model_type=None)
    else:
      _, test_x = split_data(data_input, Parts=fold_num, part_index=fold, shuffle=shuffle,model_type=model_type)
      _, test_y = split_data(data_y, Parts=fold_num, part_index=fold, shuffle=True,model_type=None)


    y_predict = np.argmax(model.predict(test_x), axis = 1)
    accuracy = np.sum(y_predict == test_y)/len(test_y)
    print (f'   All classes: {accuracy*100:0.1f}%')

    for CLS in np.unique(test_y):
      index = np.flatnonzero(test_y==CLS)
      acc = np.sum(y_predict[index] == test_y[index])/len(index)
      print (f'   Class {CLS}: {acc*100:0.1f}%')
  return
########################## Function End ######################################


######################### Function Start #####################################
def divide_time(data_x,data_y,N, data_subject_index=None):
  ax = 1 # this axis is the time samples
  if N not in range(1,21): raise ValueError("N must be between 1 and 20")
  if data_x.shape[ax] % N != 0: raise ValueError("Time samples in input data array must be multiple of N")
  if N==1: return data_x, data_y
  
  new_shape = list(data_x.shape)
  new_shape[ax-1] = data_x.shape[ax-1]*N
  new_shape[ax] = int(data_x.shape[ax]/N)
  data_x = np.reshape(data_x, new_shape)

  data_y = np.repeat(data_y,N)

  if data_subject_index is not None:
    data_subject_index_new=data_subject_index.copy()
    data_subject_index_new *= N
    return data_x, data_y, data_subject_index
  else:
    return data_x, data_y
########################## Function End ######################################


######################### Function Start #####################################
def plot_results(datasets=['physionet', 'ttk', 'bcicomptIV2a', 'chbmit'], metrics=None,ylim=[0,100],show=True,save=True):
  import glob
  from os import path
  from io import StringIO
  if save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.exists(f'{this.RESULT_FOLDER}/plots/') == False: os.mkdir(f'{this.RESULT_FOLDER}/plots/')
  if metrics is None:
    metrics=this.METRICS_TO_SAVE

  y_min = ylim[0]
  y_max = ylim[1]
  colors = ['red', 'blue','green','black','cyan',  'magenta' ,'brown','gray','teal']
  markers = ['s', '^', 'o', 'v','d','>','<','*','.']
  linestyles = ['solid','dotted']

  for dataset in datasets:
    for metric in metrics:
      files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*{metric}*.txt'))
      if len(files) == 0: continue
      if "val_" not in metric:
        files = [item for item in files if "_val_" not in item]
      sub_files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*subjects*.txt'))
 

      DataSZ = 0
      for index, file in enumerate(files):
        # finding subject count for each fold
        try:
          with open(sub_files[index],'r') as f: txt = f.read()
          txt=txt.replace('[', '').replace(']', '').split(',')
          sub_count = []
          for split in txt: sub_count.append(np.loadtxt(StringIO(split),dtype=int).size)
        except:
          sub_count = None
          
        data = 100 * np.genfromtxt(file, delimiter=',')
        if DataSZ == 0: DataSZ = data.shape[-1]

        if len(data.shape)==1:
          mean = data
        else:
          if sub_count == None:
            mean = np.mean(data,axis=0)
          else:
            mean = np.sum(np.transpose(sub_count * np.transpose(data)),axis=0)/np.sum(sub_count)
        
        Name = os.path.basename(file).split('.')[0]
        Name = Name.split('=')[1]
        #Name = Name.split('=')[1] + ('_MF' if Name.split('=')[2] == 'bandson' else '')
        plt.plot(mean,color=colors[index],linestyle = linestyles[0 if index <len(colors) else 1],linewidth=.7,marker=markers[index],markersize=5, markevery=list(np.arange(int(DataSZ/5),DataSZ,int(DataSZ/5))), label=Name)

      metric_name = metric.replace("cohen_", "")
      metric_name = metric_name if "val_" not in metric_name else f'validation {metric_name.split("_")[1]}'

      plt.xlabel('Epochs')
      plt.ylabel(f'Average {metric_name} %')
      plt.grid()
      plt.ylim(y_min,y_max)
      plt.yticks(np.arange(y_min,y_max+1,5))

      from matplotlib.ticker import MultipleLocator
      plt.gca().yaxis.set_minor_locator(MultipleLocator(4))
      plt.gca().yaxis.grid(True, which='minor',linestyle='dotted')
      plt.minorticks_on()
      
      dataset_name = {'bcicomptIV2a':'BCI Competition IV-2a', 'physionet':'Physionet','ttk':'MTA-TTK', 'chbmit':'CHB-MIT'}
      plt.title(f'Average {metric_name} for {dataset_name[dataset]} dataset.')
      plt.legend(fontsize=8)
      if save: plt.savefig(f'{this.RESULT_FOLDER}/plots/{dataset}={Name}={metric}={now}.pdf')
      if show: plt.show()
      plt.close()
  return
########################## Function End ######################################

######################### Function Start #####################################
def average_results(datasets=['physionet', 'ttk', 'bcicomptIV2a', 'chbmit'], metrics=None,epochs=50, show=True,save=True):
  import glob
  from os import path
  from io import StringIO
  if metrics is None:
    metrics=this.METRICS_TO_SAVE

  if save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_h = open(f'{this.RESULT_FOLDER}/average={now}.txt', "w")

  for dataset in datasets:
    files = glob.glob(f'{this.RESULT_FOLDER}/{dataset}*.txt')

    if len(files)==0:
      continue

    dataset_line = f'===========================\n           {dataset}          \n****************************'
    if show:
      print(dataset_line)
    if save:
      file_h.write(dataset_line+'\n')


    for metric in metrics:
      metric_name = metric.replace("cohen_", "")
      metric_name = metric_name.capitalize() if "val_" not in metric_name else f'Validation {metric_name.split("_")[1].capitalize()}'

      heading = f'-------- {metric_name}-----------'
      if save:
        file_h.write(heading+'\n')
      if show:
        print(heading)
      files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*{metric}*.txt'))
      if "val_" not in metric:
        files = [item for item in files if "_val_" not in item]
      if len(files) == 0: continue
      sub_files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*subjects*.txt'))


      DataSZ = 0
      for index, file in enumerate(files):
        # finding subject count for each fold
        try:
          with open(sub_files[index],'r') as f: txt = f.read()
          txt=txt.replace('[', '').replace(']', '').split(',')
          sub_count = []
          for split in txt: sub_count.append(np.loadtxt(StringIO(split),dtype=int).size)
        except:
          sub_count = None
        
        data = 100 * np.genfromtxt(file, delimiter=',')
        if DataSZ == 0: DataSZ = data.shape[-1]

        if len(data.shape)==1:
          mean =   data
        else:
          if sub_count == None:
            mean = np.mean(data, axis=0)
          else:
            mean = np.sum(np.transpose(sub_count * np.transpose(data)),axis=0)/np.sum(sub_count)

        Name = os.path.basename(file).split('.')[0]
        Name = Name.split('=')[1]
        #Name = Name.split('_')[1] + ('_MF' if Name.split('_')[2] == 'bandson' else '')

        line = f'{Name:15}Mean({DataSZ-epochs}-{DataSZ}) = {np.mean(mean[DataSZ-epochs:DataSZ]):0.1f}'
        if save:
          file_h.write(line+'\n')
        if show:
          print(line)
  if save:
    file_h.close()

  return
########################## Function End ######################################



######################### Function Start #####################################
def validation_time(datasets=['physionet', 'ttk', 'bcicomptIV2a', 'chbmit'], show=True, save=True):
  import glob
  from os import path
  from io import StringIO

  def time(s):
    s = int(s)
    h = s//3600
    s = s % 3600
    m = s // 60
    s = s % 60
    return f'{h:02n}:{m:02n}:{s:02n}'
  if save:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_h = open(f'{this.RESULT_FOLDER}/duration={now}.txt', "w")

  for dataset in datasets:
    files = glob.glob(f'{this.RESULT_FOLDER}/{dataset}*.txt')

    if len(files)==0:
      continue

    dataset_line = f'===========================\n           {dataset}          \n****************************'
    if show:
      print(dataset_line)
    if save:
      file_h.write(dataset_line+'\n')

    files = sorted(glob.glob(f'{this.RESULT_FOLDER}/{dataset}*time*.txt'))
    files = [item for item in files if "_validation_" not in item]
    if len(files) == 0: continue
    for file in files:
      Name = os.path.basename(file).split('.')[0]
      Name = Name.split('=')[1]
      #Name = Name.split('_')[1] + ('_MF' if Name.split('_')[2] == 'bandson' else '')

      data = np.genfromtxt(file, delimiter=',')
      line = f'{Name:15}   {time(np.round(np.sum(data)))}'
      if save:
        file_h.write(line+'\n')
      if show:
        print(line)
  if save:
    file_h.close()

  return
########################## Function End ######################################





######################### Function Start #####################################
def save_results(metric_results,time_results,dataset,Bands,subject_results=None,model_type=None, partial=False):
  Band_status= 'bandson' if Bands else 'bandsoff' 
  now = datetime.now().strftime("%Y%m%d_%H%M%S") if not partial else "partial"
  models=[model_type] if model_type else metric_results.keys()


  # remove partial result files 
  if not partial:
    command = f'rm {this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}=*=partial.txt'
    os.system(command)


  # saving metrics
  for metric in this.METRICS_TO_SAVE:
    for model_type in models:
      metric_data=metric_results[model_type]
      file_h = open(f'{this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}={metric}={now}.txt', "w")
      writer = csv.writer(file_h)
      for split in metric_data:
        d = split[metric]
        writer.writerow([f'{x:0.4f}' for x in d])
      file_h.close()

  for model_type in models:
    time_data=time_results[model_type]
    file_h = open(f'{this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}=time={now}.txt', "w")
    writer = csv.writer(file_h)
    writer.writerow([f'{x:0.2f}' for x in time_data])
    file_h.close()

  if subject_results==None:
    print('Data saved')
    return


  for model_type in models:
    subject_data=subject_results[model_type]
    file_h = open(f'{this.RESULT_FOLDER}/{dataset}={model_type}={Band_status}=subjects={now}.txt', "w")
    writer = csv.writer(file_h)
    writer.writerow(subject_data)
    file_h.close()


  if not partial: print('Data saved')
  return
########################## Function End ######################################


######################### Function Start #####################################
def dct_1d(data_x):
  from scipy import fft

  # find 1 dimentional dct for x_data
  data_x_out = fft.dct(data_x, axis=1)

  return data_x_out
########################## Function End ######################################

######################### Function Start #####################################
def fft_1d(data_x):
    # find 1 dimentional fft for x_data
  data_x_out = np.abs(np.fft.fft(data_x, axis=1))

  return data_x_out
########################## Function End ######################################



######################### Function Start #####################################
def get_chbmit_array(gap):

  # first dimension: subjects
  # second dimension: [ file number (or appended string- as in '16+'), start time (seconds), end time (seconds)]
  # The file chb02_16+.edf is taken instead of chb02_16.edf.
  # for subject No. 4 files > chb06, remove channel 24 (ECG)
  # for subject No. 9 files > chb01, remove channel 24 (VNS)
  # for subject No. 11 files > chb01, remove channels:  5,10, 13, 18, 23,




  ictal_old=[
    # Subject 1
    [[3, 2996, 3036], [4, 1467, 1494], [15, 1732, 1772], [16, 1015, 1066], [18, 1720, 1810], [21, 327, 420], [26, 1862, 1963]],
    # Subject 2
    [[16, 130,212], ['_16+', 2972, 3053], [19, 3369, 3378]],
    # Subject 3
    [[1, 362, 414], [2, 731, 796], [3, 432, 501], [4, 2162, 2214], [34, 1982, 2029], [35, 2592, 2656], [36, 1725, 1778]],
    # Subject 4
    [[5, 7804, 7853], [8, 6446, 6557], [28, 1679, 1781], [28, 3782, 3898]],
    # Subject 5
    [[6, 417, 532], [13, 1086, 1196], [16, 2317, 2413], [17, 2451, 2571], [22, 2348, 2465]],
    # Subject 6
    [[1, 1724, 1738], [1, 7461, 7476], [1, 13525, 13540], [4, 327, 347], [4, 6211, 6231], [9, 12500, 12516], [10, 10833, 10845], [13, 506, 519], [18, 7799, 7811], [24, 9387, 9403]],
    # Subject 7
    [[12, 4920, 5006], [13, 3285, 3381], [19, 13688, 13831]],
    # Subject 8
    [[2, 2670, 2841], [5, 2856, 3046], [11, 2988, 3122], [13, 2417, 2577], [21, 2083, 2347]],
    # Subject 9
    [[6, 12231, 12295], [8, 2951, 3030], [8, 9196, 9267], [19, 5299, 5361]],
    # Subject 10
    [[12, 6313, 6348], [20, 6888, 6958], [27, 2382, 2447], [30, 3021, 3079], [31, 3801, 3877], [38, 4618, 4707], [89, 1383, 1437]],
    # Subject 11
    [[82, 298, 320], [92, 2695, 2727], [99, 1454, 2206]],
    # Subject 12
    [[6, 1665, 1726], [6, 3415, 3447], [8, 1426, 1439], [8, 1591, 1614], [8, 1957, 1977], [8, 2798, 2824], [9, 3082, 3114], [9, 3503, 3535], [10, 593, 625], [10, 811, 856], [11, 1085, 1122], [23, 253, 333], [23, 425, 522], [23, 630, 670], [27, 916, 951], [27, 1097, 1124], [27, 1728, 1753], [27, 1921, 1963], [27, 2388, 2440], [27, 2621, 2669], [28, 181, 215], [29, 107, 146], [29, 554, 592], [29, 1163, 1199], [29, 1401, 1447], [29, 1884, 1921], [29, 3557, 3584], [33, 2185, 2206], [33, 2427, 2450], [36, 653, 680], [38, 1548, 1573], [38, 2798, 2821], [38, 2966, 3009], [38, 3146, 3201], [38, 3364, 3410], [42, 699, 750], [42, 945, 973], [42, 1170, 1199], [42, 1676, 1701], [42, 2213, 2236]],
    # Subject 13
    [[19, 2077, 2121], [21, 934, 1004], [40, 142, 173], [40, 530, 594], [55, 458, 478], [55, 2436, 2454], [58, 2474, 2491], [59, 3339, 3401], [60, 638, 660], [62, 851, 916], [62, 1626, 1691], [62, 2664, 2721]],
    # Subject 14
    [[3, 1986, 2000], [4, 1372, 1392], [4, 2817, 2839], [6, 1911, 1925], [11, 1838, 1879], [17, 3239, 3259], [18, 1039, 1061], [27, 2833, 2849]],
    # Subject 15
    [[6, 272, 397], [10, 1082, 1113], [15, 1591, 1748], [17, 1925, 1960], [20, 607, 662], [22, 760, 965], [28, 876, 1066], [31, 1751, 1871], [40, 834, 894], [40, 2378, 2497], [40, 3362, 3425], [46, 3322, 3429], [49, 1108, 1248], [52, 778, 849], [54, 263, 318], [54, 843, 1020], [54, 1524, 1595], [54, 2179, 2250], [54, 3428, 3460], [62, 751, 859]],
    # Subject 16
    [[10, 2290, 2299], [11, 1120, 1129], [14, 1854, 1868], [16, 1214, 1220], [17, 227, 236], [17, 1694, 1700], [17, 2162, 2170], [17, 3290, 3298], [18, 627, 635], [18, 1909, 1916]],
    # Subject 17
    [['a_03', 2282, 2372], ['a_04', 3025, 3140], ['b_63', 3136, 3224]],
    # Subject 18
    [[29, 3477, 3527], [30, 541, 571], [31, 2087, 2155], [32, 1908, 1963], [35, 2196, 2264], [36, 463, 509]],
    # Subject 19
    [[28, 299, 377], [29, 2964, 3041], [30, 3159, 3240]],
    # Subject 20
    [[12, 94, 123], [13, 1440, 1470], [13, 2498, 2537], [14, 1971, 2009], [15, 390, 425], [15, 1689, 1738], [16, 2226, 2261], [68, 1393, 1432]],
    # Subject 21
    [[19, 1288, 1344], [20, 2627, 2677], [21, 2003, 2084], [22, 2553, 2565]],
    # Subject 22
    [[20, 3367, 3425], [25, 3139, 3213], [38, 1263, 1335]],
    # Subject 23
    [[6, 3962, 4075], [8, 325, 345], [8, 5104, 5151], [9, 2589, 2660], [9, 6885, 6947], [9, 8505, 8532], [9, 9580, 9664]],
    # Subject 24
    [[1, 480, 505], [1, 2451, 2476], [3, 231, 260], [3, 2883, 2908], [4, 1088, 1120], [4, 1411, 1438], [4, 1745, 1764], [6, 1229, 1253], [7, 38, 60], [9, 1745, 1764], [11, 3527, 3597], [13, 3288, 3304], [14, 1939, 1966], [15, 3552, 3569], [17, 3515, 3581], [21, 2804, 2872]]
  ]




  path=f'{this.CONTENT_FOLDER}/physionet.org/files/chbmit/1.0.0/'
  url='https://physionet.org/files/chbmit/1.0.0/'


  ictal=[]
  for subject in range(1,25):
      ictal.insert(subject-1,[])
      file_path  = path + f'chb{subject:02d}/chb{subject:02d}-summary.txt'
      file_url = url + f'chb{subject:02d}/chb{subject:02d}-summary.txt'
      get_ipython().system(f'wget -c -q --show-progress {file_url} -P {os.path.dirname(file_path)}')

      with open(file_path, 'r') as file:
          lines = [line.strip() for line in file.readlines()]
      i=-1
      while True:
          i+=1
          if(i>=len(lines)): break
          if(lines[i].find("File Name:")>=0):
              file_name = re.search(r'chb.+edf', lines[i]).group()
              while True:
                  i+=1
                  if(lines[i].find("Number of Seizures in File:")>=0): break
              num = int(lines[i].split()[-1])
              if num > 0:
                  for s in range(num):
                      i+=1
                      start = int(re.search(r':\s*(\d+)', lines[i]).group(1))
                      i+=1
                      end = int(re.search(r':\s*(\d+)', lines[i]).group(1))
                      ictal[subject-1].append([file_name, start, end])
              else: continue
          else: continue










  # appending space for preictal data
  data=[]
  for i in range(len(ictal)):
    # Insert elements from list1 and list2 at the same index
    data.insert(i,[])
    data[i].append(ictal[i])
    data[i].append([])


    # generating preictal data
  for subject in range(len(data)):
    previous_end = 0
    previous_file = ""
    event = 0
    index = 0
    while event < len(data[subject][0]):
      index=index+1
      # Assuming the third dimension always has at least 3 elements
      file = data[subject][0][event][0]
      start = data[subject][0][event][1] - gap
      end = data[subject][0][event][2] - gap
      if file != previous_file: previous_end = 0
      previous_file = file
      if start < previous_end:
        print (f"Excluding: Ictal[{index}] Subject[{subject+1}] No enough time for preictal")
        if index==len(data[subject][0]) and event==0:
          raise Exception("\n"+f" No remaining ictals for subject[{subject+1}]")
        del data[subject][0][event]
        previous_end = data[subject][0][event][2]
        continue
      else:
        data[subject][1].insert(event,[file, start, end])
        previous_end = data[subject][0][event][2]
        event+=1

  return data
########################## Function End ######################################


######################### Function Start #####################################
def get_chbmit_interictal():

  # interictal data are taken from file groups with (at least) a separation of one file from files with seizure and two or less files from each other.






# Note: add interictal for subject 2 (file 16)





  included_data=[
    # Subject 1
    [[7, 2996, 3036], [9, 1467, 1494], [11, 1732, 1772], [31, 1015, 1066], [34, 1720, 1810], [38, 327, 420], [41, 1862, 1963]],
    # Subject 2
    [[8, 2972, 3053], [28, 3369, 3378]],
    # Subject 3
    [[8, 362, 414], [11, 731, 796], [14, 432, 501], [17, 2162, 2214], [20, 1982, 2029], [23, 2592, 2656], [26, 1725, 1778]],
    # Subject 4
    [[12, 7804, 7853], [15, 6446, 6557], [18, 1679, 1781], [22, 3782, 3898]],
    # Subject 5
    [[10, 417, 532], [26, 1086, 1196], [29, 2317, 2413], [32, 2451, 2571], [35, 2348, 2465]],
    # Subject 6
    [[6, 1724, 1738], [6, 7461, 7476], [6, 13525, 13540], [6, 327, 347], [7, 6211, 6231], [7, 12500, 12516], [7, 10833, 10845], [7, 506, 519], [15, 7799, 7811], [15, 9387, 9403]],
    # Subject 7
    [[4, 4920, 5006], [8, 3285, 3381], [16, 13688, 13831]],
    # Subject 8
    [[16, 2670, 2841], [17, 2856, 3046], [18, 2988, 3122], [24, 2417, 2577], [18, 2083, 2347]],
    # Subject 9
    [[3, 12231, 12295], [11, 2951, 3030], [13, 9196, 9267], [15, 5299, 5361]],
    # Subject 10
    [[3, 6313, 6348], [4, 6888, 6958], [5, 2382, 2447], [6, 3021, 3079], [15, 3801, 3877], [16, 4618, 4707], [17, 1383, 1437]],
    # Subject 11
    [[6, 298, 320], [9, 2695, 2727], [12, 1454, 2206]],
    # Subject 12
    [[20, 50, 111], [20, 226, 258], [20, 402, 415], [20, 578, 601], [20, 754, 774], [20, 930, 956], [20, 1106, 1138], [20, 1282, 1314], [20, 1458, 1490], [20, 1634, 1679], [20, 1810, 1847], [20, 1986, 2066], [20, 2162, 2259], [20, 2338, 2378], [20, 2514, 2549], [20, 2690, 2717], [20, 2866, 2891], [20, 3042, 3084], [20, 3218, 3270], [20, 3394, 3442], [40, 50, 84], [40, 226, 265], [40, 402, 440], [40, 578, 614], [40, 754, 800], [40, 930, 967], [40, 1106, 1133], [40, 1282, 1303], [40, 1458, 1481], [40, 1634, 1661], [40, 1810, 1835], [40, 1986, 2009], [40, 2162, 2205], [40, 2338, 2393], [40, 2514, 2560], [40, 2690, 2741], [40, 2866, 2894], [40, 3042, 3071], [40, 3218, 3243], [40, 3394, 3417]],
    # Subject 13
    [[4, 2077, 2121], [6, 934, 1004], [8, 142, 173], [8, 530, 594], [10, 458, 478], [10, 2436, 2454], [12, 2474, 2491], [14, 3339, 3401], [30, 638, 660], [37, 851, 916], [37, 1626, 1691], [37, 2664, 2721]],
    # Subject 14
    [[13, 1986, 2000], [14, 1372, 1392], [14, 2817, 2839], [22, 1911, 1925], [24, 1838, 1879], [32, 3239, 3259], [37, 1039, 1061], [37, 2833, 2849]],
    # Subject 15
    [[3, 50, 175], [3, 1112, 1143], [3, 2174, 2331], [3, 3236, 3271], [4, 50, 105], [4, 1112, 1317], [4, 2174, 2364], [4, 3236, 3356], [12, 50, 110], [12, 1112, 1231], [12, 2174, 2237], [13, 50, 157], [13, 1112, 1252], [13, 2174, 2245], [33, 50, 105], [33, 1112, 1289], [33, 2174, 2245], [35, 50, 121], [35, 1112, 1144], [35, 2174, 2282]],
    # Subject 16
    [[3, 2290, 2299], [3, 1120, 1129], [4, 1854, 1868], [5, 1214, 1220], [5, 227, 236], [6, 1694, 1700], [6, 2162, 2170], [6, 3290, 3298], [7, 627, 635], [7, 1909, 1916]],
    # Subject 17
    [['a_08', 2282, 2372], ['a_08', 3025, 3140], ['b_58', 3136, 3224]],
    # Subject 18
    [[6, 3477, 3527], [9, 541, 571], [12, 2087, 2155], [15, 1908, 1963], [18, 2196, 2264], [21, 463, 509]],
    # Subject 19
    [[9, 299, 377], [12, 2964, 3041], [15, 3159, 3240]],
    # Subject 20
    [[4, 94, 123], [6, 1440, 1470], [6, 2498, 2537], [23, 1971, 2009], [26, 390, 425], [26, 1689, 1738], [29, 2226, 2261], [34, 1393, 1432]],
    # Subject 21
    [[4, 1288, 1344], [7, 2627, 2677], [10, 2003, 2084], [13, 2553, 2565]],
    # Subject 22
    [[5, 3367, 3425], [8, 3139, 3213], [11, 1263, 1335]],
    # Subject 23
    [[17, 50, 163], [17, 557, 577], [17, 1064, 1111], [17, 1571, 1642], [17, 2078, 2140], [17, 2585, 2612], [17, 3092, 3176]],
    # Subject 24
    [[19, 50, 75], [19, 271, 296], [19, 492, 521], [19, 713, 738], [19, 934, 966], [19, 1155, 1182], [19, 1376, 1395], [19, 1597, 1621], [19, 1818, 1840], [19, 2039, 2058], [19, 2260, 2330], [19, 2481, 2497], [19, 2702, 2729], [19, 2923, 2940], [19, 3144, 3210], [19, 3365, 3433]]
  ]

  return data
########################## Function End ######################################



########################## Function Start ######################################
def read_raw(edf_filename, selected_ch_names):
  url_name = 'https://physionet.org/files/chbmit/1.0.0/' + edf_filename.split("/")[-2] + '/' + edf_filename.split("/")[-1]
  get_ipython().system(f'wget  -q -c --show-progress  {url_name} -P {os.path.dirname(edf_filename)}')


  raw = mne.io.read_raw_edf(edf_filename,stim_channel=None, exclude=('-'),verbose="ERROR")

  if 'T8-P8-1' in raw.info.ch_names:
    #X1 = raw.get_data(picks='T8-P8-0').astype(np.float32)
    #X2 = raw.get_data(picks='T8-P8-1').astype(np.float32)
    #if not np.array_equal(X1, X2):
      #print(f'Duplicates channels are not equals for file {edf_filename}')
    raw.drop_channels('T8-P8-1')
    raw.rename_channels({'T8-P8-0' : 'T8-P8'})

  if raw.info.ch_names!=selected_ch_names:
    raw.reorder_channels(selected_ch_names)

  return raw
########################## Function End ######################################


########################## Function Start ######################################
def stack_lists(list1, list2, list3=None):
  # Ensure the input lists have the same length
  if len(list1) != len(list2):
      raise ValueError("Input lists must have the same length")

  # Create a new list to hold the stacked lists
  stacked_list = []

  # Iterate over the indices and insert elements from both lists
  for i in range(len(list1)):
    # Insert elements from list1 and list2 at the same index
    stacked_list.insert(i,[])
    stacked_list[i].append(list1[i])
    stacked_list[i].append(list2[i])
    if list3 != None:
      stacked_list[i].append(list3[i])

  return stacked_list
########################## Function End ######################################




