#!/usr/bin/env python3
'''
   Do inference on a DICOM dataset.
'''

import argparse
import os
from fastMONAI.vision_all import *
from imagedata.series import Series
from skimage import measure
import numpy as np

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

# load variables
_, reorder, resample = load_variables(pkl_fn='data/SpineSegmentation/vars.pkl')

# load learner
learn = load_learner('data/SpineSegmentation/spine_model.pkl')

mask = inference(learn, reorder, resample, datafolder)

mask_obj = Series(datafolder)
new_mask = mask.numpy()
new_mask = new_mask.squeeze()
new_mask = np.transpose(new_mask, (-1, 1, 0))
new_mask = new_mask.copy()
new_mask = new_mask.astype(np.uint16)
mask_obj[:] = new_mask

def save_series_pred(series_obj, save_dir, val='1234'):
    '''Make sure we get derived UIDs to allow for overwrite of image objects in PACS'''
    my_seriesInstanceUID = series_obj.seriesInstanceUID[:-4] + val
    series_obj.seriesInstanceUID = my_seriesInstanceUID
    my_studyID = series_obj.patientID[3:]
    series_obj.studyID = my_studyID

    for slice_idx in range(series_obj.slices):
        my_SOPInstanceUID = series_obj.getDicomAttribute(
            'SOPInstanceUID', slice=slice_idx)[:-4] + val
        series_obj.setDicomAttribute(
            'SOPInstanceUID', my_SOPInstanceUID, slice=slice_idx)

    series_obj.write(save_dir, opts={'keep_uid': True}, formats=['dicom'])


save_series_pred(mask_obj, output + '/mask')
