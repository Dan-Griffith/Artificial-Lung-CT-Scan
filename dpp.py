import numpy as np 

import matplotlib.pyplot as plt
import pydicom
from skimage import measure 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import disk, opening
from PIL import Image
import os


# Use a raw string for the path
directory_labels = r'/Users/dangriffith/Personal Projects/Igneium_3D_Reconstruction/3D-Medical-Prediction/Data_Labels/Subject (10)/650458-'
directory_features = r'/Users/dangriffith/Personal Projects/Igneium_3D_Reconstruction/3D-Medical-Prediction/Data_features'


### ----- Handles the loading of the 3D point clould. -----###
    #Need to make labels dimesnions congruent
def load_scans(dcm_path):
    
    files = os.listdir(dcm_path)
    for file in files: 
        print(file)
    # file_nums = [np.int64(file.split(" ")[0]) for file in files]
    # sorted_file_nums = np.sort(file_nums)[::-1]
    slices = [pydicom.dcmread( directory_labels + '/'+ file) for file in files]
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
        
        intercept = slices[n].get((0x0028, 0x1052))
        slope = slices[n].get((0x0028, 0x1053))
        
        if slope != None and intercept != None:
            if slope != 1:
                images[n] = slope.value * images[n].astype(np.float64)
                images[n] = images[n].astype(np.int16)
                
            images[n] += np.int16(intercept.value)
    
    return np.array(images, dtype=np.int16)


def plot_3d(image, threshold=700, color="navy"):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces,_,_ = measure.marching_cubes(p, threshold)

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

### ---------- ###

### ----- Hangles the testing of loading a sinlge patient -----###
def run_label_test():

    scans = load_scans(directory_labels)
    print(len(scans))


    hu_scans = transform_to_hu(scans)
    print(hu_scans.shape)

    segmented_lungs = segment_lung_mask(hu_scans)
    plot_3d(segmented_lungs)

#run_label_test()
### ---------- ###

### ---- 2D image point cloud creation-----###

    #need to make feature arrays congruent.
def load_image(img_path): 
    images = []
    image_path =  os.listdir(directory_features)
    for file in image_path:
        im = Image.open(directory_features + '/' + file)
        images.append(im)
    return images
        

def create_lung_highlights(images):

    for image in images: 
        image = image.convert('L')
        highlited_xray = image.point(lambda p: p > 150 and 255)
       
        
        for i in range(highlited_xray.size[0]):
            for j in range(highlited_xray.size[1]):
                
                
                if highlited_xray.getpixel((i,j)) == 0: 
                    highlited_xray.putpixel((i,j), (1))
                if highlited_xray.getpixel((i,j)) == 255: 
                    highlited_xray.putpixel((i,j), (0))
        
        return highlited_xray           

def create_2D_point_cloud(image):

    point_cloud = np.array(image)
    point_cloud =np.argwhere(point_cloud == 1)

    return point_cloud
    
    ### View the point cloud created ###
    
    # plt.scatter(points_with_value_1[:, 1], points_with_value_1[:, 0], c='red', marker='o')
    # plt.gca().invert_yaxis()  
    # plt.title('Points with Value 1')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.grid(True)
    # plt.show()
    
    
def test_load_features():
    
    images = load_image(directory_features)
    highlighted_image = create_lung_highlights(images)
    create_2D_point_cloud(highlighted_image)

### ---------- ###

test_load_features()
run_label_test()