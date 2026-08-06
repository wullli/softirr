# REFLACX – reports and eye-tracking data for localization of abnormalities in chest x-rays

This dataset, named REFLACX, provides eye-tracking data collected while radiologists dictated reports for frontal chest x-rays from the MIMIC-CXR, paired with the timestamped transcription of the dictation. 
For each case, we also provide other labels for validating algorithms derived from this dataset. These labels include image-level labels of anomalies found during the reading, ellipses localizing these anomalies, and bounding boxes around lungs and heart. We also provide, in a [GitHub repository](https://github.com/ricbl/eyetracking), the MATLAB interface code used for displaying and collecting data, the code used to postprocess the data, and examples on the use of the dataset.

This dataset is separated into two folders, main_data, containing the main files for the intended use of the dataset and metadata tables, and gaze_data, which is more significant in data size and contains the complete recorded eye-tracking data in 1000 Hz. Both folders contain several subfolders, one for each reading of a chest x-ray, named after an ID assigned to each reading. All IDs and some of their metadata are listed in the metadata tables. The dataset has three metadata tables, one for each of the phases of data collection: two preliminary phases, collected from November 11, 2020, to January 4, 2021, and from March 1, 2021, to March 11, 2021, when radiologists read a shared set of 109 chest x-rays, and the main phase, collected from March 24, 2021, to June 7, 2021, with readings of 2,507 chest x-rays. Each subfolder contains several comma-separated tables, one for each type of data collected: fixations, localization ellipses, chest bounding boxes, and the timestamped transcriptions. An example of the folder structure for one of the cases is:

```
main_data/
    metadata_phase_1.csv
    P102R009922/
        fixations.csv
        anomaly_location_ellipses.csv
        chest_bounding_box.csv
        timestamps_transcription.csv
        transcription.txt
gaze_data/
    P102R009922/
        gaze.csv
```

For the total 3,052 IDs, eye-tracking data and reports are not provided for 20 of them, which had their eye-tracking data discarded for quality but were kept in the dataset to calculate agreement scores for the manual labels. These readings represent 2,616 unique chest x-rays and 2,199 unique subjects. The columns of each of the tables are described below. All timestamp columns are counted from the start of the audio recording in seconds. All columns representing pixel coordinates have the origin at the top left corner, with x coordinates representing the horizontal axis and y coordinates representing the vertical axis.

- `main_data/metadata_phase_<phase>.csv`:
  * `id` (string): used to identify each chest x-ray reading.
  * `split` (string): the split (train, validate or test) given by the MIMIC-CXR dataset.
  * `eye_tracking_data_discarded` (Boolean): this column is True in the small subset of 20 images from phases 1 and 2 that had their eye-tracking data discarded, but the validation labels were kept to allow us to calculate the variability scores for these phases. The fixations.csv, gaze.csv, timestamps_transcription.csv, and transcription.txt files are not provided for cases for which this column is True.
  * `image` (string): folder location of the used chest x-ray in the MIMIC-CXR dataset.
  * `dicom_id` (string): id identifying the read chest x-ray that can be used to link this table with the MIMIC-CXR tables.
  * `subject_id` (string): id of the patient of the chest x-ray, from the MIMIC-CXR and MIMIC-IV datasets.
  * `image_size_x` , `image_size_y` (int): horizontal and vertical sizes of the chest x-ray, in pixels.
  * `Airway wall thickening`, `Atelectasis`, `Consolidation`, `Emphysema`, `Enlarged cardiac silhouette`, `Fibrosis`, `Fracture`, `Groundglass opacity`, `Mass`, `Nodule`, `Pleural effusion`, `Pleural thickening`, `Pneumothorax`, `Pulmonary edema`, `Wide mediastinum` (phase 1); `Abnormal mediastinal contour`, `Acute fracture`, `Atelectasis`, `Consolidation`, `Enlarged cardiac silhouette`, `Enlarged hilum`, `Groundglass opacity`, `Hiatal hernia`, `High lung volume / emphysema`, `Interstitial lung disease`, `Lung nodule or mass`, `Pleural abnormality`, `Pneumothorax`, `Pulmonary edema` (phases 2 and 3) (int): columns for the certainty of image-level labels, with values from 0 to 5, representing the maximum certainty selected over all ellipses drawn for the label, with the following representation: 0: not selected by radiologist, 1: Unlikely (<10%), 2: Less Likely (\~25%), 3: Possibly (\~50%), 4: Suspicious for/Probably (~75%), 5: Consistent with (>90%).
  * `Quality issue`, `Support devices` (phase 1); `Support devices` (phases 2 and 3) (Boolean): image-level labels with only presence indicated.
  * `Other` (string): written additional labels, separated by a "|" symbol.
- `main_data/<id>/fixations.csv`: list of fixations, i.e., stabilizations of the gaze of the radiologist in a specific location, for the <id> reading.
  * `timestamp_start_fixation`, `timestamp_end_fixation` (float): time when the fixation started and ended. The difference between these two values may be used as a weighting of the importance of that fixation.
  * `x_position`, `y_position` (int): average position for the fixation, in image space.
  * `pupil_area_normalized` (float): area of the pupil, normalized by the pupil area measured during calibration at the beginning of the data collection session.
  * `window_level`, `window_width` (float): variables representing the average windowing state over the fixation. Images were shown according to
```
image_shown=(original_image-window_level)/window_width+0.5
```
, where image_shown was then trimmed to 0 to 1, original_image was the loaded DICOM normalized to the [0, 1] range (usually by a division by 4096), window_level varied from 0 to 1, and window_width from 1.5e-5 to 2. 
  * `angular_resolution_x_pixels_per_degree`, `angular_resolution_y_pixels_per_degree` (int): the number of image pixels per visual angle in degrees for each image axis, depending on the screen region where the fixation was and the level of zoom applied to the image. 
  * `xmin_shown_from_image`, `ymin_shown_from_image`, `xmax_shown_from_image`, `ymax_shown_from_image` (int): the part of the image that was shown on the screen.
  * `xmin_in_screen_coordinates`, `ymin_in_screen_coordinates`, `xmax_in_screen_coordinates`, `ymax_in_screen_coordinates` (int): the coordinates on the screen where the image was shown. Together with the <…>_shown_from_image columns, these columns represent the zooming and panning state at the start of the fixation.
- `gaze_data/<id>/gaze.csv`: list of gaze locations at 1000 Hz. This table has the same columns as the fixations table, except for the timestamp columns, which were replaced by a single timestamp_sample column. Rows with empty values for positions, pupil area, and angular resolutions represent moments when the radiologist's eye was not found, e.g., when radiologists blinked.
- `main_data/<id>/timestamps_transcriptions.csv`:
  * `word` (string): the transcribed word, after correction by radiologists and by another person. The symbols for periods, commas, and slashes were also considered words since the radiologists dictated them.
  * `timestamp_start_word`, `timestamp_end_word` (float): time of start and end of the dictation of the word.
- `main_data/<id>/transcription.txt`: full corrected transcription in text format.
- `main_data/<id>/anomaly_location_ellipses.csv`: 
  * `xmin`, `ymin`, `xmax`, `ymax` (int): extreme points of each axis of the drawn ellipse in image space.
  * `certainty` (int): certainty of the presence of the highlighted finding, following the same 0-5 representation as in the metadata table.
  * (Boolean) The rest of the columns, one for each of the possible labels, represent which labels were found to have the possibility of representing the highlighted finding. Radiologists were asked not to highlight the labels "Support devices," "Quality issue," and "Other," while the rest of the labels were mandatorily drawn.
- `main_data/<id>/chest_bounding_box.csv`: table with one row representing the drawn chest bounding box around the lungs and heart.
  * `xmin`, `ymin`, `xmax`, `ymax` (int): extreme coordinates of the bounding box in image space.

The main_data folder is around 45MB (13 MB zipped).
The gaze_data is around 10 GB (635 MB zipped).
