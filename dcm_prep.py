import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom
import scipy.ndimage

import glob
from skimage import measure 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import disk, opening, closing
# from tqdm import tqdm

# from IPython.display import HTML
from PIL import Image
import os


basepath = "/Users/dangriffith/Documents/SDS_Final/osic-pulmonary-fibrosis-progression/"
train = pd.read_csv(basepath + "/train.csv")
test = pd.read_csv(basepath + "/test.csv")

#print(train.shape)
train.head()

#/Users/dangriffith/Library/Mobile Documents/com~apple~CloudDocs/Universtiy Winter 2024/COMP 4900 SDS/Final Project /osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430

train["dcm_path"] = basepath + "train/" + train.Patient 


def load_scans(dcm_path):
    
    files = os.listdir(dcm_path)
    file_nums = [np.int64(file.split(".")[0]) for file in files]
    sorted_file_nums = np.sort(file_nums)[::-1]
    slices = [pydicom.dcmread(dcm_path + "/" + str(file_num) + ".dcm" ) for file_num in sorted_file_nums]
    return slices

def set_outside_scanner_to_air(raw_pixelarrays):
    # in OSIC we find outside-scanner-regions with raw-values of -2000. 
    # Let's threshold between air (0) and this default (-2000) using -1000
    raw_pixelarrays[raw_pixelarrays <= -1000] = 0
    return raw_pixelarrays


def transform_to_hu(slices):
    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    images = set_outside_scanner_to_air(images)
    
    # convert to HU
    for n in range(len(slices)):
        
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        
        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)
            
        images[n] += np.int16(intercept)
    
    return np.array(images, dtype=np.int16)

def get_window_value(feature):
    if type(feature) == pydicom.multival.MultiValue:
        return np.int64(feature[0])
    else:
        return np.int64(feature)

def set_manual_window(hu_image, custom_center, custom_width):
    w_image = hu_image.copy()
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    return w_image


def plot_3d(image, threshold=700, color="navy"):
    # Don't transpose the image since it's already in the correct orientation
    p = image
    
    # Select the desired channel for plotting (assuming it's the first channel)
    p = image.transpose(2,1,0)
    
    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.2)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    
    # recompute the resize factor and spacing such that we match the rounded new shape above
    rounded_resize_factor = new_shape / image.shape
    rounded_new_spacing = spacing / rounded_resize_factor
    
    # zoom with resize factor
    image = scipy.ndimage.zoom(image, rounded_resize_factor, mode='nearest')
    
    return image, rounded_new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def fill_lungs(binary_image):
    image = binary_image.copy()
    # For every slice we determine the largest solid structure
    for i, axial_slice in enumerate(image):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice)
        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None: #This slice contains some lung
            image[i][labeling != l_max] = 1
    return image

def segment_lung_mask(image):
    segmented = np.zeros(image.shape)   
    
    for n in range(image.shape[0]):
        binary_image = np.array(image[n] > -320, dtype=np.int8)+1
        labels = measure.label(binary_image)
        
        bad_labels = np.unique([labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])
        for bad_label in bad_labels:
            binary_image[labels == bad_label] = 2
    
        #We have a lot of remaining small signals outside of the lungs that need to be removed. 
        #In our competition closing is superior to fill_lungs 
        selem = disk(2)
        binary_image = opening(binary_image, selem)
    
        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1
        
        segmented[n] = binary_image.copy() * image[n]
    
    return segmented


def get_point_cloud(id):
    dp = []
    count = 0

    for i in range(len(train.dcm_path.values)):
        
        example = train.dcm_path.values[i]
       
        print(id)
   
        # print('hi')
        scans = load_scans(example)
        #print(scans[0])
        hu_scans = transform_to_hu(scans)
        # print(hu_scans)
        N = 1000


        pixelspacing_r = []
        pixelspacing_c = []
        slice_thicknesses = []
        patient_id = []
        patient_pth = []
        row_values = []
        column_values = []
        window_widths = []
        window_levels = []


        patients = train.Patient.unique()[0:N]


        for patient in patients:
            
            patient_id.append(patient)


            path = train[train.Patient == patient].dcm_path.values[0]
        
            example_dcm = os.listdir(path)[5]
            patient_pth.append(path)
            dataset = pydicom.dcmread(path + "/" + example_dcm)
            
            window_widths.append(get_window_value(dataset.WindowWidth))
            window_levels.append(get_window_value(dataset.WindowCenter))
            
            spacing = dataset.PixelSpacing
            slice_thicknesses.append(dataset.SliceThickness)
            
            row_values.append(dataset.Rows)
            column_values.append(dataset.Columns)
            pixelspacing_r.append(spacing[0])
            pixelspacing_c.append(spacing[1])
            
        scan_properties = pd.DataFrame(data=patient_id, columns=["patient"])
        scan_properties.loc[:, "rows"] = row_values
        scan_properties.loc[:, "columns"] = column_values
        scan_properties.loc[:, "area"] = scan_properties["rows"] * scan_properties["columns"]
        scan_properties.loc[:, "pixelspacing_r"] = pixelspacing_r
        scan_properties.loc[:, "pixelspacing_c"] = pixelspacing_c
        scan_properties.loc[:, "pixelspacing_area"] = scan_properties.pixelspacing_r * scan_properties.pixelspacing_c
        scan_properties.loc[:, "slice_thickness"] = slice_thicknesses
        scan_properties.loc[:, "patient_pth"] = patient_pth
        scan_properties.loc[:, "window_width"] = window_widths
        scan_properties.loc[:, "window_level"] = window_levels
        scan_properties.head()
        scan_properties["r_distance"] = scan_properties.pixelspacing_r * scan_properties.rows
        scan_properties["c_distance"] = scan_properties.pixelspacing_c * scan_properties["columns"]
        scan_properties["area_cm2"] = 0.1* scan_properties["r_distance"] * 0.1*scan_properties["c_distance"]
        scan_properties["slice_volume_cm3"] = 0.1*scan_properties.slice_thickness * scan_properties.area_cm2


        max_path = scan_properties[
            scan_properties.slice_volume_cm3 == scan_properties.slice_volume_cm3.max()].patient_pth.values[0]
        min_path = scan_properties[
            scan_properties.slice_volume_cm3 == scan_properties.slice_volume_cm3.min()].patient_pth.values[0]

        min_scans = load_scans(min_path)
        min_hu_scans = transform_to_hu(min_scans)

        max_scans = load_scans(max_path)
        max_hu_scans = transform_to_hu(max_scans)

        # plot_3d(max_hu_scans)
        old_distribution = max_hu_scans.flatten()
        example = train.dcm_path.values[0]
        scans = load_scans(example)
        hu_scans = transform_to_hu(scans)
        # plot_3d(hu_scans)
        img_resampled, spacing = resample(max_hu_scans, scans, [1,1,1])
    
        binary_image = np.array((hu_scans[20]>-320), dtype=np.int8) + 1
        np.unique(binary_image)
        labels = measure.label(binary_image)

        bad_labels = np.unique([labels[0,:], labels[-1,:], labels[:,0], labels[:,-1]])

        binary_image_2 = binary_image.copy()
        for bad_label in bad_labels:
            binary_image_2[labels == bad_label] = 2


        selem = disk(2)
        closed_binary_2 = closing(binary_image_2, selem)

        closed_binary_2 -= 1 #Make the image actual binary
        closed_binary_2 = 1-closed_binary_2 # Invert it, lungs are now 1

        filled_lungs_binary = fill_lungs(binary_image_2)

        air_pocket_binary = closed_binary_2.copy()
        # Remove other air pockets insided body
        labels_2 = measure.label(air_pocket_binary, background=0)
        l_max = largest_label_volume(labels_2, bg=0)
        if l_max is not None: # There are air pockets
            air_pocket_binary[labels_2 != l_max] = 0
    
        segmented_lungs = segment_lung_mask(np.array([hu_scans[20]]))

        # fig, ax = plt.subplots(1,2,figsize=(20,10))
        # ax[0].imshow(set_manual_window(hu_scans[20], -700, 255), cmap="Blues_r")
        # ax[1].imshow(set_manual_window(segmented[0], -700, 255), cmap="Blues_r")

        
        
        # dp.append(segmented_lungs)
        count+=1
        # print(segmented_lungs.shape)
        
    
        segmented_lungs = segment_lung_mask(hu_scans)
        print(segmented_lungs.shape)
        
        return segmented_lungs

dcm_path = '/Users/dangriffith/Personal Projects/Igneium_3D_Reconstruction/3D-Medical-Prediction/Data_Labels/Subject (1)/98.12.2/56364398.dcm'
pc = get_point_cloud(dcm_path)

print(pc.shape)
plot_3d(pc, threshold=-600)
# process_data()