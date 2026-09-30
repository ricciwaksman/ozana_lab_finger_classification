





import os
import re
import sys
import csv
import numpy as np
import scipy.io as sio
import h5py




from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d




from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
   accuracy_score,
   confusion_matrix,
   ConfusionMatrixDisplay,
   classification_report
)




from joblib import dump, load




import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog








# =========================
# SETTINGS
# =========================




FPS = 333.33




ROI_ROWS = 10
ROI_COLS = 10




CONTRAST_WINDOW = 7
SEARCH_WINDOW_SEC = (0.9, 1.8)




MOVEMENT_METHOD = "lowest_contrast"


# Temporal-window conditions, all defined relative to the same detected peak.
TEMPORAL_WINDOWS_SEC = {
    "full": (-0.35, 0.35),
    "medium": (-0.25, 0.25),
    "short": (-0.15, 0.15),
    "early": (-0.35, 0.00),
    "very_early": (-0.35, -0.15),
}

CURRENT_TEMPORAL_CONDITION = "full"
MOVEMENT_START_OFFSET_SEC = TEMPORAL_WINDOWS_SEC["full"][0]
MOVEMENT_END_OFFSET_SEC = TEMPORAL_WINDOWS_SEC["full"][1]




NPERSEG = 32
NOVERLAP = 24




FREQ_BANDS = [
  (10, 20),
  (20, 40),
  (40, 80),
]




TRAIN_FRACTION = 0.75
TEST_FRACTION = 0.25




# Use None for random split every run.
# Use an integer like 42 for repeatable results.
RANDOM_SEED = 42




# =========================
# ONLINE MODEL IMPROVEMENTS
# =========================




# The old code called partial_fit only once.
# For an online-capable model, we can still train incrementally,
# but we should show the training set to the model several times in
# a shuffled order so it actually has time to learn.
ONLINE_TRAINING_EPOCHS = 40




# When learning from future files/folders, retrain using the saved replay
# buffer instead of only updating once on the newest data. This reduces
# catastrophic forgetting.
FUTURE_UPDATE_EPOCHS = 20




# log_loss usually behaves more stably than hinge loss for small, noisy,
# online learning problems, and it also allows predict_proba if needed later.
SGD_LOSS = "log_loss"
SGD_ALPHA = 0.0001




# Keep this fixed so that the online training order is repeatable even when
# the train/test split is random.
ONLINE_RANDOM_SEED = 42




# Saved model base folder
BASE_MODEL_DIR = os.path.expanduser("~/VS Code Python/finger_model_versions_removed")
os.makedirs(BASE_MODEL_DIR, exist_ok=True)




DEFAULT_MODEL_VERSION = "reduced_spectrogram_classifier_v1"




MODEL_VERSION = DEFAULT_MODEL_VERSION
MODEL_DIR = os.path.join(BASE_MODEL_DIR, MODEL_VERSION)
MODEL_PATH = os.path.join(MODEL_DIR, f"reduced_finger_model_{MODEL_VERSION}.joblib")
LEARNING_LOG_PATH = os.path.join(MODEL_DIR, "future_learning_log.csv")




# Valid finger classes
CLASSES = np.array([1, 2, 3, 4, 5])








# =========================
# FILE SELECTION HELPERS
# =========================




app = QApplication.instance()
if app is None:
  app = QApplication(sys.argv)








def choose_folder(title):
  folder = QFileDialog.getExistingDirectory(None, title)
  if not folder:
      print("No folder selected.")
      return None
  return folder








def choose_mat_file(title):
  file_path, _ = QFileDialog.getOpenFileName(
      None,
      title,
      "",
      "MAT files (*.mat)"
  )




  if not file_path:
      print("No file selected.")
      return None




  return file_path








# =========================
# HELPERS
# =========================








def find_mat_files(parent):
  mat_files = []




  for root, dirs, files in os.walk(parent):
      for file in files:




          # Ignore hidden macOS metadata files from external drives
          if file.startswith("._"):
              continue




          # Ignore other hidden files
          if file.startswith("."):
              continue




          if file.lower().endswith(".mat"):
              mat_files.append(os.path.join(root, file))




  return sorted(mat_files)








def get_finger_label(path):
  name = os.path.splitext(os.path.basename(path))[0]




  patterns = [
      r"finger[_\- ]?([1-5])",
      r"\bf[_\- ]?([1-5])\b",
      r"\b([1-5])\.",
      r"^([1-5])[_\- ]",
      r"^([1-5])"
  ]




  for pattern in patterns:
      match = re.search(pattern, name, flags=re.IGNORECASE)
      if match:
          return int(match.group(1))




  return None








def load_mat_video(path):
  try:
      data = sio.loadmat(path)
      keys = [k for k in data.keys() if not k.startswith("__")]




      video = None




      for key in keys:
          arr = data[key]
          if arr.ndim == 3:
              video = arr
              break




      if video is None:
          raise ValueError("No 3D array found.")




  except NotImplementedError:
      with h5py.File(path, "r") as f:
          keys = list(f.keys())
          video = np.array(f[keys[0]])




  video = np.asarray(video)




  # Convert height x width x frames to frames x height x width
  if video.shape[-1] > 100 and video.shape[0] < 1000 and video.shape[1] < 1000:
      video = np.moveaxis(video, -1, 0)




  video = video.astype(np.float32)




  return video








def compute_frame_contrast(video):
  means = np.mean(video, axis=(1, 2))
  stds = np.std(video, axis=(1, 2))




  contrast = stds / (means + 1e-8)




  return np.nan_to_num(
      contrast,
      nan=0.0,
      posinf=0.0,
      neginf=0.0
  )








def find_movement_window(video):
  """
  Finds the movement window from the whole-frame contrast signal.




  This replaces the old method, which used:
      abs(gradient(smoothed contrast))




  The old method could choose an early noisy spike.
  The new default method ("lowest_contrast") chooses the bottom of the
  large sustained contrast drop, matching what you saw visually in the
  movement-window checker.
  """
  contrast = compute_frame_contrast(video)




  smooth = uniform_filter1d(
      contrast,
      size=CONTRAST_WINDOW
  )




  gradient = np.gradient(smooth)
  abs_gradient = np.abs(gradient)




  start_frame = int(SEARCH_WINDOW_SEC[0] * FPS)
  end_frame = int(SEARCH_WINDOW_SEC[1] * FPS)




  start_frame = max(0, start_frame)
  end_frame = min(len(smooth), end_frame)




  if end_frame <= start_frame:
      raise ValueError("Invalid movement search window.")




  if MOVEMENT_METHOD == "lowest_contrast":
      search_segment = smooth[start_frame:end_frame]
      local_peak = np.argmin(search_segment)




  elif MOVEMENT_METHOD == "negative_gradient":
      search_segment = gradient[start_frame:end_frame]
      local_peak = np.argmin(search_segment)




  elif MOVEMENT_METHOD == "absolute_gradient":
      search_segment = abs_gradient[start_frame:end_frame]
      local_peak = np.argmax(search_segment)




  else:
      raise ValueError(
          "Invalid MOVEMENT_METHOD. Use 'lowest_contrast', "
          "'negative_gradient', or 'absolute_gradient'."
      )




  peak_frame = start_frame + local_peak




  start_offset_frames = int(round(MOVEMENT_START_OFFSET_SEC * FPS))
  end_offset_frames = int(round(MOVEMENT_END_OFFSET_SEC * FPS))




  movement_start = max(0, peak_frame + start_offset_frames)
  movement_end = min(video.shape[0], peak_frame + end_offset_frames)




  if movement_end <= movement_start:
      raise ValueError(
          "Temporal movement window is empty or invalid. "
          f"Condition: {CURRENT_TEMPORAL_CONDITION}, "
          f"start: {movement_start}, end: {movement_end}, peak: {peak_frame}"
      )




  return movement_start, movement_end, peak_frame








def roi_contrast_timeseries(video, r0, r1, c0, c1):
  roi = video[:, r0:r1, c0:c1]




  means = np.mean(roi, axis=(1, 2))
  stds = np.std(roi, axis=(1, 2))




  contrast = stds / (means + 1e-8)




  return np.nan_to_num(
      contrast,
      nan=0.0,
      posinf=0.0,
      neginf=0.0
  )












def extract_spectrogram_features_from_signal(signal):
  """
  Extracts the five retained spectral features from one ROI:




  1. Spectral centroid
  2. Log total power
  3. Log power from 10-20 Hz
  4. Log power from 20-40 Hz
  5. Log power from 40-80 Hz




  Removed:
  - dominant frequency
  - spectral entropy
  - 0-5 Hz power
  - 5-10 Hz power
  - 80-150 Hz power
  """
  signal = np.asarray(signal)
  signal = signal - np.mean(signal)




  n_features = 2 + len(FREQ_BANDS)




  if np.std(signal) < 1e-12:
      return np.zeros(n_features)




  actual_nperseg = min(NPERSEG, len(signal))
  actual_noverlap = min(NOVERLAP, actual_nperseg - 1)




  f, t, Sxx = spectrogram(
      signal,
      fs=FPS,
      nperseg=actual_nperseg,
      noverlap=actual_noverlap,
      scaling="density",
      mode="psd"
  )




  Sxx = np.nan_to_num(
      Sxx,
      nan=0.0,
      posinf=0.0,
      neginf=0.0
  )




  mean_power_per_freq = np.mean(Sxx, axis=1)
  total_power = np.sum(mean_power_per_freq)




  if total_power <= 1e-12:
      return np.zeros(n_features)




  spectral_centroid = (
      np.sum(f * mean_power_per_freq) / total_power
  )
  total_power_log = np.log1p(total_power)




  features = [
      spectral_centroid,
      total_power_log
  ]




  for low, high in FREQ_BANDS:
      band_mask = (f >= low) & (f < high)
      band_power = np.sum(mean_power_per_freq[band_mask])
      features.append(np.log1p(band_power))




  return np.array(features)








# =========================
# FEATURE EXTRACTION
# =========================




def extract_spectrogram_features(video):
  """
  Extracts only the retained ROI spectrogram features.




  The image is divided into an 8 x 8 grid. For each ROI, the movement-period
  contrast time series is converted into five spectral features.




  Total feature length:
      8 x 8 ROIs x 5 features = 320 features
  """
  movement_start, movement_end, peak_frame = find_movement_window(video)




  movement_video = video[movement_start:movement_end]




  n_frames, h, w = movement_video.shape




  roi_h = h // ROI_ROWS
  roi_w = w // ROI_COLS




  spectrogram_features = []




  for r in range(ROI_ROWS):
      for c in range(ROI_COLS):
          r0 = r * roi_h
          r1 = (r + 1) * roi_h if r < ROI_ROWS - 1 else h




          c0 = c * roi_w
          c1 = (c + 1) * roi_w if c < ROI_COLS - 1 else w




          movement_signal = roi_contrast_timeseries(
              movement_video,
              r0,
              r1,
              c0,
              c1
          )




          spec_features = extract_spectrogram_features_from_signal(
              movement_signal
          )




          spectrogram_features.extend(spec_features)




  spectrogram_features = np.array(spectrogram_features)




  spectrogram_features = np.nan_to_num(
      spectrogram_features,
      nan=0.0,
      posinf=0.0,
      neginf=0.0
  )




  return spectrogram_features, movement_start, movement_end, peak_frame








def extract_features_from_mat_file(path):
  video = load_mat_video(path)




  features, movement_start, movement_end, peak_frame = extract_spectrogram_features(
      video
  )




  return features, movement_start, movement_end, peak_frame








# =========================
# DATASET BUILDING
# =========================




def build_dataset(folder, dataset_name):
  mat_files = find_mat_files(folder)




  print("\n==============================")
  print(dataset_name)
  print("==============================")
  print("Folder:", folder)
  print("Found mat files:", len(mat_files))




  X = []
  y = []
  names = []




  for path in mat_files:
      label = get_finger_label(path)




      if label is None:
          print("Skipping, could not get label:", os.path.basename(path))
          continue




      try:
          features, movement_start, movement_end, peak_frame = extract_features_from_mat_file(
              path
          )




      except Exception as e:
          print("Failed:", os.path.basename(path), e)
          continue




      if np.isnan(features).any() or np.isinf(features).any():
          print("Skipping bad feature vector:", os.path.basename(path))
          continue




      X.append(features)
      y.append(label)
      names.append(os.path.basename(path))




      print(
          os.path.basename(path),
          "| finger:", label,
          "| movement:",
          movement_start,
          "-",
          movement_end,
          "| peak:",
          peak_frame
      )




  X = np.array(X)
  y = np.array(y)
  names = np.array(names)




  print("\nUsable samples:", len(X))




  for finger in CLASSES:
      print("Finger", finger, ":", np.sum(y == finger))




  return X, y, names








def build_combined_dataset_from_folders(folders):
  all_X = []
  all_y = []
  all_names = []




  for i, folder in enumerate(folders, start=1):
      X_day, y_day, names_day = build_dataset(
          folder,
          f"DAY {i} DATASET"
      )




      day_name = os.path.basename(folder)




      names_day = np.array([
          day_name + "/" + name
          for name in names_day
      ])




      all_X.append(X_day)
      all_y.append(y_day)
      all_names.append(names_day)




  X_all = np.concatenate(all_X, axis=0)
  y_all = np.concatenate(all_y, axis=0)
  names_all = np.concatenate(all_names, axis=0)




  return X_all, y_all, names_all






# =========================
# MODEL SAVE / LOAD
# =========================




def set_model_version(model_version):
  global MODEL_VERSION, MODEL_DIR, MODEL_PATH, LEARNING_LOG_PATH




  # Clean name so it is safe for folder/file names
  model_version = model_version.strip()
  model_version = re.sub(r"[^a-zA-Z0-9_\-]+", "_", model_version)




  if model_version == "":
      model_version = DEFAULT_MODEL_VERSION




  MODEL_VERSION = model_version
  MODEL_DIR = os.path.join(BASE_MODEL_DIR, MODEL_VERSION)
  os.makedirs(MODEL_DIR, exist_ok=True)




  MODEL_PATH = os.path.join(
      MODEL_DIR,
      f"reduced_finger_model_{MODEL_VERSION}.joblib"
  )




  LEARNING_LOG_PATH = os.path.join(
      MODEL_DIR,
      "future_learning_log.csv"
  )




  print("\nUsing model version:", MODEL_VERSION)
  print("Model will save/load from:", MODEL_PATH)




def save_model(scaler, clf, feature_length, replay_X=None, replay_y=None, replay_names=None):
  bundle = {
      "scaler": scaler,
      "clf": clf,
      "classes": CLASSES,
      "feature_length": feature_length,
      "replay_X": replay_X,
      "replay_y": replay_y,
      "replay_names": replay_names,
      "settings": {
          "FPS": FPS,
          "ROI_ROWS": ROI_ROWS,
          "ROI_COLS": ROI_COLS,
          "CONTRAST_WINDOW": CONTRAST_WINDOW,
          "SEARCH_WINDOW_SEC": SEARCH_WINDOW_SEC,
          "MOVEMENT_METHOD": MOVEMENT_METHOD,
          "CURRENT_TEMPORAL_CONDITION": CURRENT_TEMPORAL_CONDITION,
          "MOVEMENT_START_OFFSET_SEC": MOVEMENT_START_OFFSET_SEC,
          "MOVEMENT_END_OFFSET_SEC": MOVEMENT_END_OFFSET_SEC,
          "NPERSEG": NPERSEG,
          "NOVERLAP": NOVERLAP,
          "FREQ_BANDS": FREQ_BANDS,
          "SPECTRAL_FEATURES_PER_ROI": [
              "spectral_centroid",
              "log_total_power",
              "log_power_10_20_hz",
              "log_power_20_40_hz",
              "log_power_40_80_hz"
          ],
          "ONLINE_TRAINING_EPOCHS": ONLINE_TRAINING_EPOCHS,
          "FUTURE_UPDATE_EPOCHS": FUTURE_UPDATE_EPOCHS,
          "SGD_LOSS": SGD_LOSS,
          "SGD_ALPHA": SGD_ALPHA
      }
  }




  dump(bundle, MODEL_PATH)




  print("\nSaved model to:", MODEL_PATH)




def choose_existing_model_version():
  if not os.path.exists(BASE_MODEL_DIR):
      print("No model folder found yet.")
      return False




  model_versions = [
      name for name in os.listdir(BASE_MODEL_DIR)
      if os.path.isdir(os.path.join(BASE_MODEL_DIR, name))
  ]




  if len(model_versions) == 0:
      print("No saved model versions found.")
      return False




  model_versions = sorted(model_versions)




  print("\nAvailable saved models:")
  for i, name in enumerate(model_versions, start=1):
      print(f"{i}. {name}")




  choice = input("\nChoose model number: ").strip()




  try:
      choice = int(choice)
  except ValueError:
      print("Invalid choice.")
      return False




  if choice < 1 or choice > len(model_versions):
      print("Invalid choice.")
      return False




  selected_model = model_versions[choice - 1]




  set_model_version(selected_model)




  return True




def load_model(ask_user=True):
  if ask_user:
      ok = choose_existing_model_version()




      if not ok:
          raise FileNotFoundError("No saved model selected.")




  if not os.path.exists(MODEL_PATH):
      raise FileNotFoundError(
          f"Could not find {MODEL_PATH}. Train the model first."
      )




  bundle = load(MODEL_PATH)




  print("\nLoaded model:", MODEL_VERSION)
  print("Loaded from:", MODEL_PATH)




  return bundle








# =========================
# ONLINE TRAINING HELPERS
# =========================




def make_balanced_sample_weights(y):
  """
  Gives each finger class similar influence during training.
  This helps if one finger has more usable samples than another.
  """
  y = np.asarray(y)
  weights = np.ones(len(y), dtype=np.float32)




  for finger in CLASSES:
      count = np.sum(y == finger)
      if count > 0:
          weights[y == finger] = len(y) / (len(CLASSES) * count)




  return weights








def train_online_classifier(X_train_scaled, y_train, n_epochs=ONLINE_TRAINING_EPOCHS):
  """
  Trains the online-capable classifier properly.




  The original code used partial_fit once, which made the model very weak.
  This keeps the same online model idea, but trains for multiple shuffled
  passes over the available training samples.
  """
  rng = np.random.default_rng(ONLINE_RANDOM_SEED)




  clf = SGDClassifier(
      loss=SGD_LOSS,
      penalty="l2",
      alpha=SGD_ALPHA,
      learning_rate="optimal",
      max_iter=1,
      tol=None,
      random_state=ONLINE_RANDOM_SEED
  )




  sample_weights = make_balanced_sample_weights(y_train)




  for epoch in range(n_epochs):
      indices = rng.permutation(len(X_train_scaled))




      X_epoch = X_train_scaled[indices]
      y_epoch = y_train[indices]
      w_epoch = sample_weights[indices]




      if epoch == 0:
          clf.partial_fit(
              X_epoch,
              y_epoch,
              classes=CLASSES,
              sample_weight=w_epoch
          )
      else:
          clf.partial_fit(
              X_epoch,
              y_epoch,
              sample_weight=w_epoch
          )




  return clf








def print_training_diagnostics(X_train, X_train_scaled, y_train, clf):
  train_pred = clf.predict(X_train_scaled)
  train_acc = accuracy_score(y_train, train_pred)




  print("\n==============================")
  print("TRAINING DIAGNOSTICS")
  print("==============================")
  print("Online training epochs:", ONLINE_TRAINING_EPOCHS)
  print("SGD loss:", SGD_LOSS)
  print("SGD alpha:", SGD_ALPHA)
  print("Raw feature mean:", round(float(np.mean(X_train)), 6))
  print("Raw feature std:", round(float(np.std(X_train)), 6))
  print("Scaled feature mean:", round(float(np.mean(X_train_scaled)), 6))
  print("Scaled feature std:", round(float(np.std(X_train_scaled)), 6))
  print("Training accuracy:", round(train_acc * 100, 1), "%")




  for finger in CLASSES:
      print("Training samples for finger", finger, ":", np.sum(y_train == finger))








def retrain_from_replay_buffer(bundle, extra_X=None, extra_y=None, extra_names=None, n_epochs=FUTURE_UPDATE_EPOCHS):
  """
  Rebuilds the model from saved examples plus new labelled examples.




  This is safer than updating only on the newest file/folder because it keeps
  the older examples in memory and reduces forgetting.
  """
  replay_X = bundle.get("replay_X")
  replay_y = bundle.get("replay_y")
  replay_names = bundle.get("replay_names")




  if replay_X is None or replay_y is None:
      print("WARNING: This saved model has no replay buffer.")
      print("Falling back to ordinary partial_fit update.")
      return None




  X_parts = [replay_X]
  y_parts = [replay_y]
  name_parts = [replay_names if replay_names is not None else np.array(["old_sample"] * len(replay_y))]




  if extra_X is not None and extra_y is not None:
      X_parts.append(extra_X)
      y_parts.append(extra_y)
      if extra_names is None:
          extra_names = np.array(["new_sample"] * len(extra_y))
      name_parts.append(extra_names)




  X_all = np.concatenate(X_parts, axis=0)
  y_all = np.concatenate(y_parts, axis=0)
  names_all = np.concatenate(name_parts, axis=0)




  scaler = StandardScaler()
  X_all_scaled = scaler.fit_transform(X_all)




  clf = train_online_classifier(
      X_all_scaled,
      y_all,
      n_epochs=n_epochs
  )




  bundle["scaler"] = scaler
  bundle["clf"] = clf
  bundle["replay_X"] = X_all
  bundle["replay_y"] = y_all
  bundle["replay_names"] = names_all




  return bundle




# =========================
# INITIAL TRAINING
# =========================




def choose_training_folders():
  """
  Lets the user choose unlimited folders.




  Keep selecting folders one at a time.
  Press Cancel when finished.
  """
  selected_folders = []




  print("\nSelect training folders one at a time.")
  print("Press Cancel when you have selected all the folders you want.")




  folder_number = 1




  while True:
      folder = choose_folder(
          f"Select folder {folder_number} - press Cancel when done"
      )




      if folder is None:
          break




      selected_folders.append(folder)
      print("Added folder:", folder)




      folder_number += 1




  if len(selected_folders) == 0:
      print("No folders selected.")
      return None




  print("\nSelected folders:")
  for folder in selected_folders:
      print(folder)




  return selected_folders








def train_initial_model():


  model_name = input(
      "\nEnter a name for this model version "
      "(example: ricci_8_rounds): "
  ).strip()




  set_model_version(model_name)




  selected_folders = choose_training_folders()




  if selected_folders is None:
      return




  X_all, y_all, names_all = build_combined_dataset_from_folders(
      selected_folders
  )




  if len(X_all) == 0:
      print("Dataset is empty.")
      return
 
   # Train on all selected folders. A separate held-out folder will be used later for testing.


  X_train = X_all
  y_train = y_all
  train_names = names_all


  X_test = None
  y_test = None
  test_names = None
 
  do_test = False




  print("\n==============================")
  print("FINAL DATASET")
  print("==============================")
  print("Total usable samples:", len(X_all))
  print("Training samples:", len(X_train))




  if do_test:
      print("Testing samples:", len(X_test))




  print("Feature vector length:", X_train.shape[1])
  print(
      "Expected feature length:",
      ROI_ROWS * ROI_COLS * (2 + len(FREQ_BANDS))
  )




  print("\nTraining files:")
  for name, label in zip(train_names, y_train):
      print(name, "| finger:", label)




  # -------------------------
  # TRAIN ONLINE-CAPABLE MODEL
  # -------------------------




  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)




  clf = train_online_classifier(
      X_train_scaled,
      y_train,
      n_epochs=ONLINE_TRAINING_EPOCHS
  )




  print_training_diagnostics(
      X_train=X_train,
      X_train_scaled=X_train_scaled,
      y_train=y_train,
      clf=clf
  )




  save_model(
      scaler=scaler,
      clf=clf,
      feature_length=X_train.shape[1],
      replay_X=X_train,
      replay_y=y_train,
      replay_names=train_names
  )




  # -------------------------
  # TEST MODEL
  # -------------------------




  if do_test:
      X_test_scaled = scaler.transform(X_test)




      y_pred = clf.predict(X_test_scaled)




      acc = accuracy_score(y_test, y_pred)




      print("\n==============================")
      print("RESULTS")
      print("==============================")
      print("Accuracy:", round(acc * 100, 1), "%")




      print("\nPrediction counts:")
      for finger in CLASSES:
          print("Finger", finger, ":", np.sum(y_pred == finger))




      print("\nTrue labels:")
      print(y_test)




      print("\nPredicted labels:")
      print(y_pred)




      print("\nPer-file predictions:")
      for name, true, pred in zip(test_names, y_test, y_pred):
          print(name, "| true:", true, "| predicted:", pred)




      labels = list(CLASSES)




      cm = confusion_matrix(
          y_test,
          y_pred,
          labels=labels
      )




      disp = ConfusionMatrixDisplay(
          confusion_matrix=cm,
          display_labels=labels
      )




      disp.plot(
          cmap="Blues",
          values_format="d"
      )




      plt.title(
          f"Online-Capable Reduced Spectrogram Classifier\n"
          f"Accuracy = {acc * 100:.1f}%"
      )




      plt.xlabel("Predicted label")
      plt.ylabel("True label")
      plt.tight_layout()
      plt.show()








# =========================
# PREDICT FUTURE DATA
# =========================




def predict_new_file(file_path=None):
  bundle = load_model()




  scaler = bundle["scaler"]
  clf = bundle["clf"]
  feature_length = bundle["feature_length"]




  if file_path is None:
      file_path = choose_mat_file("Select future .mat file to predict")




  if file_path is None:
      return None




  try:
      features, movement_start, movement_end, peak_frame = extract_features_from_mat_file(
          file_path
      )




  except Exception as e:
      print("Failed to extract features:", e)
      return None




  x = features.reshape(1, -1)




  if x.shape[1] != feature_length:
      print("Feature length mismatch.")
      print("Model expected:", feature_length)
      print("New file has:", x.shape[1])
      print("Check ROI_ROWS, ROI_COLS, retained spectral features, and FREQ_BANDS.")
      return None




  x_scaled = scaler.transform(x)




  prediction = clf.predict(x_scaled)[0]




  print("\n==============================")
  print("FUTURE FILE PREDICTION")
  print("==============================")
  print("File:", os.path.basename(file_path))
  print("Movement window:", movement_start, "-", movement_end)
  print("Peak frame:", peak_frame)
  print("Predicted finger:", prediction)




  if hasattr(clf, "decision_function"):
      scores = clf.decision_function(x_scaled)
      print("Decision scores:", scores)




  return prediction








# =========================
# LEARN FROM ONE FUTURE FILE
# =========================




def learn_new_file(file_path=None, true_label=None):
  bundle = load_model()




  scaler = bundle["scaler"]
  clf = bundle["clf"]
  classes = bundle["classes"]
  feature_length = bundle["feature_length"]




  if file_path is None:
      file_path = choose_mat_file("Select future .mat file to learn from")




  if file_path is None:
      return




  if true_label is None:
      print("\nEnter the TRUE finger label.")
      print("Valid labels: 1, 2, 3, 4, 5")




      try:
          true_label = int(input("True finger label: "))
      except ValueError:
          print("Invalid label.")
          return




  if true_label not in classes:
      print("Invalid label. Must be one of:", classes)
      return




  try:
      features, movement_start, movement_end, peak_frame = extract_features_from_mat_file(
          file_path
      )




  except Exception as e:
      print("Failed to extract features:", e)
      return




  x = features.reshape(1, -1)




  if x.shape[1] != feature_length:
      print("Feature length mismatch.")
      print("Model expected:", feature_length)
      print("New file has:", x.shape[1])
      print("Check ROI_ROWS, ROI_COLS, retained spectral features, and FREQ_BANDS.")
      return




  x_scaled = scaler.transform(x)




  # Optional: prediction before learning
  old_prediction = clf.predict(x_scaled)[0]




  # Improved update: use the saved replay buffer if available.
  # This prevents the model from learning only the newest file and forgetting
  # what it previously learned.
  updated_bundle = retrain_from_replay_buffer(
      bundle,
      extra_X=x,
      extra_y=np.array([true_label]),
      extra_names=np.array([os.path.basename(file_path)]),
      n_epochs=FUTURE_UPDATE_EPOCHS
  )




  if updated_bundle is None:
      # Fallback for older saved models that do not contain replay_X/replay_y.
      clf.partial_fit(
          x_scaled,
          np.array([true_label]),
          classes=classes
      )
      bundle["clf"] = clf
  else:
      bundle = updated_bundle




  dump(bundle, MODEL_PATH)




  log_future_learning(
      file_path=file_path,
      true_label=true_label,
      old_prediction=old_prediction,
      movement_start=movement_start,
      movement_end=movement_end,
      peak_frame=peak_frame
  )




  print("\n==============================")
  print("MODEL UPDATED")
  print("==============================")
  print("File:", os.path.basename(file_path))
  print("Previous prediction:", old_prediction)
  print("True label used for learning:", true_label)
  print("Movement window:", movement_start, "-", movement_end)
  print("Peak frame:", peak_frame)
  print("Updated model saved to:", MODEL_PATH)








def log_future_learning(
  file_path,
  true_label,
  old_prediction,
  movement_start,
  movement_end,
  peak_frame
):
  file_exists = os.path.exists(LEARNING_LOG_PATH)




  with open(LEARNING_LOG_PATH, mode="a", newline="") as f:
      writer = csv.writer(f)




      if not file_exists:
          writer.writerow([
              "file_path",
              "file_name",
              "old_prediction",
              "true_label",
              "movement_start",
              "movement_end",
              "peak_frame"
          ])




      writer.writerow([
          file_path,
          os.path.basename(file_path),
          old_prediction,
          true_label,
          movement_start,
          movement_end,
          peak_frame
      ])








# =========================
# LEARN FROM A FUTURE FOLDER
# =========================




def learn_from_future_folder():
  bundle = load_model()




  scaler = bundle["scaler"]
  clf = bundle["clf"]
  classes = bundle["classes"]
  feature_length = bundle["feature_length"]




  folder = choose_folder("Select future folder to learn from")




  if folder is None:
      return




  X_new, y_new, names_new = build_dataset(
      folder,
      "FUTURE LEARNING DATASET"
  )




  if len(X_new) == 0:
      print("No usable future samples found.")
      return




  if X_new.shape[1] != feature_length:
      print("Feature length mismatch.")
      print("Model expected:", feature_length)
      print("New folder has:", X_new.shape[1])
      return




  X_new_scaled = scaler.transform(X_new)




  old_predictions = clf.predict(X_new_scaled)




  print("\n==============================")
  print("PREDICTIONS BEFORE LEARNING")
  print("==============================")




  for name, true, pred in zip(names_new, y_new, old_predictions):
      print(name, "| true:", true, "| predicted before update:", pred)




  # Improved update: retrain using old replay examples + new labelled folder.
  # This is slower than one partial_fit call, but much safer and usually more
  # stable for small datasets.
  updated_bundle = retrain_from_replay_buffer(
      bundle,
      extra_X=X_new,
      extra_y=y_new,
      extra_names=names_new,
      n_epochs=FUTURE_UPDATE_EPOCHS
  )




  if updated_bundle is None:
      # Fallback for older saved models that do not contain replay_X/replay_y.
      clf.partial_fit(
          X_new_scaled,
          y_new,
          classes=classes
      )
      bundle["clf"] = clf
  else:
      bundle = updated_bundle




  dump(bundle, MODEL_PATH)




  print("\n==============================")
  print("MODEL UPDATED FROM FUTURE FOLDER")
  print("==============================")
  print("New samples learned:", len(X_new))
  print("Updated model saved to:", MODEL_PATH)








# =========================
# TEST SAVED MODEL ON A LABELLED FOLDER
# =========================


def display_classification_results(
   y_true,
   y_pred,
   model_name,
   test_folder_name,
   title="Saved Model Test Results"
):
   """
   Displays the confusion matrix and classification report
   together in one compact Matplotlib pop-up window.
   """


   labels = list(CLASSES)
   finger_names = [f"Finger {finger}" for finger in labels]


   accuracy = accuracy_score(y_true, y_pred)


   # -------------------------
   # CONFUSION MATRIX
   # -------------------------


   cm = confusion_matrix(
       y_true,
       y_pred,
       labels=labels
   )


   # -------------------------
   # CLASSIFICATION REPORT
   # -------------------------


   report = classification_report(
       y_true,
       y_pred,
       labels=labels,
       target_names=finger_names,
       output_dict=True,
       zero_division=0
   )


   row_names = finger_names + [
       "Accuracy",
       "Macro avg",
       "Weighted avg"
   ]


   table_data = []


   # Per-finger rows
   for finger_name in finger_names:
       table_data.append([
           f"{report[finger_name]['precision']:.3f}",
           f"{report[finger_name]['recall']:.3f}",
           f"{report[finger_name]['f1-score']:.3f}",
           f"{int(report[finger_name]['support'])}"
       ])


   # Accuracy row
   table_data.append([
       "",
       "",
       f"{report['accuracy']:.3f}",
       f"{len(y_true)}"
   ])


   # Macro-average row
   table_data.append([
       f"{report['macro avg']['precision']:.3f}",
       f"{report['macro avg']['recall']:.3f}",
       f"{report['macro avg']['f1-score']:.3f}",
       f"{int(report['macro avg']['support'])}"
   ])


   # Weighted-average row
   table_data.append([
       f"{report['weighted avg']['precision']:.3f}",
       f"{report['weighted avg']['recall']:.3f}",
       f"{report['weighted avg']['f1-score']:.3f}",
       f"{int(report['weighted avg']['support'])}"
   ])


   # -------------------------
   # COMBINED POP-UP WINDOW
   # -------------------------


   fig, axes = plt.subplots(
       1,
       2,
       figsize=(13, 6),
       gridspec_kw={
           "width_ratios": [1, 1.15],
           "wspace": 0.16
       }
   )


   # Main figure heading
   fig.suptitle(
       f"{title}\n"
       f"Model: {model_name}   |   Testing folder: {test_folder_name}\n"
       f"Accuracy: {accuracy * 100:.1f}%",
       fontsize=14,
       fontweight="bold",
       y=0.97
   )


   # -------------------------
   # LEFT: CONFUSION MATRIX
   # -------------------------


   disp = ConfusionMatrixDisplay(
       confusion_matrix=cm,
       display_labels=labels
   )


   disp.plot(
       ax=axes[0],
       cmap="Blues",
       values_format="d",
       colorbar=False
   )


   axes[0].set_title(
       "Confusion Matrix",
       pad=8
   )
   axes[0].set_xlabel("Predicted finger")
   axes[0].set_ylabel("True finger")


 # -------------------------
 #  RIGHT: RESULTS TABLE
 # -------------------------


   axes[1].axis("off")


   axes[1].text(
     0.5,
     0.81,
     "Classification Report",
     ha="center",
     va="bottom",
     fontsize=13,
     transform=axes[1].transAxes)


   table = axes[1].table(
     cellText=table_data,
     rowLabels=row_names,
     colLabels=[
       "Precision",
       "Recall",
       "F1-score",
       "Support"],
     cellLoc="center",
     rowLoc="center",
     loc="center",
     colWidths=[0.20, 0.18, 0.19, 0.16])


   table.auto_set_font_size(False)
   table.set_fontsize(10)
   table.scale(1.0, 1.55)


   # Make the row-label column fit its text more closely
   for row_index in range(1, len(row_names) + 1):
       table[(row_index, -1)].set_width(0.24)


   fig.subplots_adjust(
   left=0.06,
   right=0.98,
   bottom=0.10,
   top=0.82,
   wspace=0.16)


   plt.show()










def test_saved_model_on_folder():
  bundle = load_model()




  scaler = bundle["scaler"]
  clf = bundle["clf"]
  feature_length = bundle["feature_length"]




  folder = choose_folder("Select labelled folder to test saved model")
  test_folder_name = os.path.basename(os.path.normpath(folder))




  if folder is None:
      return




  X_test, y_test, names_test = build_dataset(
      folder,
      "SAVED MODEL TEST DATASET"
  )




  if len(X_test) == 0:
      print("No usable test samples found.")
      return




  if X_test.shape[1] != feature_length:
      print("Feature length mismatch.")
      print("Model expected:", feature_length)
      print("Test folder has:", X_test.shape[1])
      return




  X_test_scaled = scaler.transform(X_test)




  y_pred = clf.predict(X_test_scaled)




  acc = accuracy_score(y_test, y_pred)




  print("\n==============================")
  print("SAVED MODEL TEST RESULTS")
  print("==============================")
  print("Accuracy:", round(acc * 100, 1), "%")




  print("\nPrediction counts:")
  for finger in CLASSES:
      print("Finger", finger, ":", np.sum(y_pred == finger))




  print("\nPer-file predictions:")
  for name, true, pred in zip(names_test, y_test, y_pred):
      print(name, "| true:", true, "| predicted:", pred)




  display_classification_results(
   y_true=y_test,
   y_pred=y_pred,
   model_name=MODEL_VERSION,
   test_folder_name=test_folder_name,
   title="Saved Model Test Results")









# =========================
# TEMPORAL-WINDOW EXPERIMENT
# =========================


def set_temporal_window(condition_name):
  global CURRENT_TEMPORAL_CONDITION
  global MOVEMENT_START_OFFSET_SEC
  global MOVEMENT_END_OFFSET_SEC

  if condition_name not in TEMPORAL_WINDOWS_SEC:
      raise ValueError(f"Unknown temporal-window condition: {condition_name}")

  CURRENT_TEMPORAL_CONDITION = condition_name
  MOVEMENT_START_OFFSET_SEC, MOVEMENT_END_OFFSET_SEC = TEMPORAL_WINDOWS_SEC[condition_name]

  print("\nTemporal condition:", CURRENT_TEMPORAL_CONDITION)
  print("Window:", MOVEMENT_START_OFFSET_SEC, "to", MOVEMENT_END_OFFSET_SEC,
        "seconds relative to detected peak")


def choose_temporal_experiment_folders():
  print("\nSelect all TRAINING folders one at a time.")
  print("Press Cancel after the final training folder.")

  training_folders = choose_training_folders()
  if training_folders is None:
      return None, None

  test_folder = choose_folder("Select the single HELD-OUT TEST folder")
  if test_folder is None:
      return None, None

  return training_folders, test_folder


def save_temporal_experiment_results(results, output_folder):
  csv_path = os.path.join(output_folder, "temporal_window_results.csv")
  fieldnames = [
      "condition", "start_offset_sec", "end_offset_sec",
      "window_duration_sec", "accuracy", "macro_precision",
      "macro_recall", "macro_f1", "weighted_f1",
      "training_samples", "test_samples"
  ]

  with open(csv_path, "w", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()
      for result in results:
          writer.writerow({key: result[key] for key in fieldnames})

  print("\nSaved summary CSV to:", csv_path)
  return csv_path


def plot_temporal_experiment_results(results, output_folder):
  condition_names = [result["condition"] for result in results]
  accuracies = [result["accuracy"] * 100 for result in results]
  macro_f1_scores = [result["macro_f1"] * 100 for result in results]

  x = np.arange(len(condition_names))
  width = 0.36
  fig, ax = plt.subplots(figsize=(10, 6))

  accuracy_bars = ax.bar(x - width / 2, accuracies, width, label="Accuracy")
  f1_bars = ax.bar(x + width / 2, macro_f1_scores, width, label="Macro F1")
  ax.axhline(y=20, linestyle="--", linewidth=2, label="Chance level (20%)")

  ax.set_ylim(0, 100)
  ax.set_ylabel("Performance (%)")
  ax.set_xlabel("Temporal-window condition")
  ax.set_title("Temporal-Window Comparison")
  ax.set_xticks(x)
  ax.set_xticklabels(condition_names)
  ax.legend()
  ax.bar_label(accuracy_bars, fmt="%.1f", padding=3)
  ax.bar_label(f1_bars, fmt="%.1f", padding=3)
  fig.tight_layout()

  figure_path = os.path.join(output_folder, "temporal_window_results.png")
  fig.savefig(figure_path, dpi=300, bbox_inches="tight")
  print("Saved summary graph to:", figure_path)
  plt.show()
  return figure_path


def run_temporal_window_experiment():
  training_folders, test_folder = choose_temporal_experiment_folders()
  if training_folders is None or test_folder is None:
      print("Temporal-window experiment cancelled.")
      return

  output_folder = choose_folder(
      "Select a folder in which to save the temporal experiment results"
  )
  if output_folder is None:
      print("No output folder selected.")
      return

  test_folder_name = os.path.basename(os.path.normpath(test_folder))
  results = []

  print("\n==========================================")
  print("STARTING TEMPORAL-WINDOW EXPERIMENT")
  print("==========================================")
  print("Training folders:")
  for folder in training_folders:
      print(" -", folder)
  print("Held-out test folder:", test_folder)
  print("Conditions:", list(TEMPORAL_WINDOWS_SEC.keys()))

  for condition_name in TEMPORAL_WINDOWS_SEC:
      print("\n\n" + "#" * 60)
      print("RUNNING CONDITION:", condition_name)
      print("#" * 60)

      set_temporal_window(condition_name)

      X_train, y_train, train_names = build_combined_dataset_from_folders(
          training_folders
      )
      X_test, y_test, test_names = build_dataset(
          test_folder,
          f"TEST DATASET — {condition_name}"
      )

      if len(X_train) == 0 or len(X_test) == 0:
          print("Skipping empty condition:", condition_name)
          continue

      if X_train.shape[1] != X_test.shape[1]:
          print("Skipping feature-length mismatch:", condition_name)
          continue

      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      clf = train_online_classifier(
          X_train_scaled,
          y_train,
          n_epochs=ONLINE_TRAINING_EPOCHS
      )

      y_pred = clf.predict(X_test_scaled)
      accuracy = accuracy_score(y_test, y_pred)
      report = classification_report(
          y_test,
          y_pred,
          labels=list(CLASSES),
          output_dict=True,
          zero_division=0
      )

      start_offset, end_offset = TEMPORAL_WINDOWS_SEC[condition_name]
      result = {
          "condition": condition_name,
          "start_offset_sec": start_offset,
          "end_offset_sec": end_offset,
          "window_duration_sec": end_offset - start_offset,
          "accuracy": accuracy,
          "macro_precision": report["macro avg"]["precision"],
          "macro_recall": report["macro avg"]["recall"],
          "macro_f1": report["macro avg"]["f1-score"],
          "weighted_f1": report["weighted avg"]["f1-score"],
          "training_samples": len(y_train),
          "test_samples": len(y_test),
          "y_true": y_test.copy(),
          "y_pred": y_pred.copy()
      }
      results.append(result)

      print("\n==========================================")
      print("CONDITION RESULT:", condition_name)
      print("==========================================")
      print("Accuracy:", round(accuracy * 100, 1), "%")
      print("Macro F1:", round(result["macro_f1"], 3))

  if len(results) == 0:
      print("No temporal-window conditions completed successfully.")
      return

  print("\n\n" + "=" * 78)
  print("FINAL TEMPORAL-WINDOW RESULTS")
  print("=" * 78)
  print(f"{'Condition':<15}{'Window (s)':<20}{'Accuracy':>12}{'Macro F1':>12}")
  print("-" * 78)

  for result in results:
      window_text = (
          f"{result['start_offset_sec']:+.2f} "
          f"to {result['end_offset_sec']:+.2f}"
      )
      print(
          f"{result['condition']:<15}"
          f"{window_text:<20}"
          f"{result['accuracy'] * 100:>11.1f}%"
          f"{result['macro_f1']:>12.3f}"
      )

  best_result = max(results, key=lambda result: result["accuracy"])
  print("-" * 78)
  print(
      "Best condition by accuracy:",
      best_result["condition"],
      f"({best_result['accuracy'] * 100:.1f}%)"
  )

  save_temporal_experiment_results(results, output_folder)
  plot_temporal_experiment_results(results, output_folder)

  for result in results:
      display_classification_results(
          y_true=result["y_true"],
          y_pred=result["y_pred"],
          model_name="Temporal experiment — " + result["condition"],
          test_folder_name=test_folder_name,
          title="Temporal-Window Experiment"
      )


# =========================
# MENU
# =========================




def main_menu():
  while True:
      print("\n==============================")
      print("FINGER CLASSIFICATION SYSTEM")
      print("==============================")
      print("1 = Make a new model")      
      print("2 = Use a saved model to guess one unknown file")
      print("3 = Teach a saved model from one new file")
      print("4 = Teach a saved model from a whole new folder")
      print("5 = Test a saved model on a labelled folder")
      print("6 = Run complete temporal-window experiment")
      print("7 = Exit")




      choice = input("\nChoose an option: ").strip()




      if choice == "1":
          train_initial_model()




      elif choice == "2":
          predict_new_file()




      elif choice == "3":
          learn_new_file()




      elif choice == "4":
          learn_from_future_folder()




      elif choice == "5":
          test_saved_model_on_folder()




      elif choice == "6":
          run_temporal_window_experiment()




      elif choice == "7":
          print("Exiting.")
          break




      else:
          print("Invalid choice. Choose 1, 2, 3, 4, 5, 6, or 7.")








# =========================
# RUN PROGRAM
# =========================




if __name__ == "__main__":
  main_menu()










