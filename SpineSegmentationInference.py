#!/usr/bin/env python3
'''
   Do inference on a DICOM dataset.
   ./SpineSegmentationInference.py data/Prediction/input data/Prediction/output
'''

import argparse
import os
from fastMONAI.vision_all import *
from imagedata.series import Series
from skimage import measure
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    prog='SpineSegmentation',
    description='SpineSegmentation inference script.'
)
parser.add_argument('fn', type=str, help='Directory name of the input folder')
parser.add_argument('on', type=str, help='Directory name for the output')
args = parser.parse_args()

datafolder = args.fn + '/'
output = args.on

if not os.path.exists(output):
    try:
        os.mkdir(output, 0o777)
    except OSError as error:
        print(error)


_, reorder, resample = load_variables(pkl_fn='data/SpineSegmentation/vars.pkl')
learn_inf = load_learner('data/SpineSegmentation/spine_model.pkl',cpu=True)

from scipy.ndimage import label
from skimage.morphology import remove_small_objects

def pred_postprocess(pred_mask, avg_disc=10437, percentage=0.2):
    small_objects = avg_disc * percentage
    labeled_mask, ncomponents = label(pred_mask)
    labeled_mask = remove_small_objects(labeled_mask, min_size=small_objects)
    return np.where(labeled_mask>0, 1.0, 0.0)

# alternative prediction
# mask = inference(learn_inf, reorder, resample, datafolder)
org_img, input_img, org_size = med_img_reader(datafolder, reorder, resample, only_tensor=False)
# This is the only difference to the previous version, specify input as filename - not as input_img
df = pd.Series(data={'image': datafolder}, index=['image'])
pred, *_ = learn_inf.predict(df)
pred_data = pred.data
m = pred_data.numpy()
m = m.squeeze()
m = np.transpose(m, (-1,0,1))
m = np.flip(m,axis=1)
pred_data = m.copy()
pred_mask = do_pad_or_crop(pred_data[None], input_img.shape[1:], padding_mode=0, mask_name=None)
input_img.set_data(pred_mask)

from torchio import Resize
resize = Resize(
        org_size, 
        image_interpolation='nearest',
        label_interpolation='nearest'
)
input_img = resize(input_img)

from SimpleITK import DICOMOrient, GetArrayFromImage
orientation_itk = DICOMOrient(input_img.as_sitk(), ('').join(org_img.orientation))
reoriented_array =  GetArrayFromImage(orientation_itk).transpose()
reoriented_array = reoriented_array[None]
org_img.set_data(reoriented_array)

mask_data = org_img.data
mask_data = torch.where(mask_data > 0.5, 1.0, 0.0)
mask_data = torch.Tensor(pred_postprocess(mask_data))
org_img.set_data(mask_data) # Here is the new mask data

def save_series_pred(series_obj, save_dir, val='1234'):
    '''Make sure we get derived UIDs to allow for overwrite of image objects in PACS'''
    # the following changed in imagedata between the older 0.2 and the latest 0.3 version
    my_seriesInstanceUID = series_obj.seriesInstanceUID[:-4] + val
    series_obj.seriesInstanceUID = my_seriesInstanceUID
    my_studyID = series_obj.patientID[3:]
    series_obj.studyID = my_studyID

    for slice_idx in range(series_obj.slices):
        my_SOPInstanceUID = series_obj.SOPInstanceUIDs[0,slice_idx][:-4] + val
        #my_SOPInstanceUID = series_obj.getDicomAttribute(
        #    'SOPInstanceUID', slice=slice_idx)[:-4] + val
        series_obj.setDicomAttribute(
            'SOPInstanceUID', my_SOPInstanceUID, slice=slice_idx)

    series_obj.write(save_dir, opts={'keep_uid': True}, formats=['dicom'])

mask = org_img
mask_obj = Series(datafolder)
new_mask = mask.numpy()
new_mask = new_mask.squeeze()
new_mask = np.transpose(new_mask, (-1, 1, 0))
# Attention, this is just not nice. DICOMOrient could have done something?
new_mask = np.flip(new_mask, axis=0)
new_mask = np.flip(new_mask, axis=1)
new_mask = new_mask.copy()
new_mask = new_mask.astype(np.uint16)
mask_obj[:] = new_mask

if not("patientID" in mask_obj) or mask_obj.patientID == None or mask_obj.patientID == "":
    mask_obj.patientID = "MeMyselfAndI"

save_series_pred(mask_obj, output + '/mask')
