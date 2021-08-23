from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
import numpy as np
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy import wcs
from sklearn.cluster import KMeans
from DoG_maps import load_Data

def get_WCS_from_fits(clustername):
    """Takes in a string (clustername) as parameter, finds the associated fits files and returns the world coordinates"""
    hdu_map,img_map,weight_map= load_Data(clustername)
    w=wcs.WCS(hdu_map.header)
    return w

def maskOuterRing(img,radius):
    """Takes in a 2D array and a mask radius as parameters. Sets 2D array values to zero outside a circular region"""
    temp = np.copy(img)
    shape = temp.shape
    dist = np.zeros((shape))
    x_arr = np.arange(shape[0]) - (shape[0]/2)
    y_arr = np.arange(shape[1]) - (shape[1]/2)
    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            distance=np.sqrt(x_arr[i]**2 + y_arr[j]**2)
            dist[i,j] = distance
    temp[(dist>radius)]=0.0
    return temp

def maskByWeight(img,clustername):
    temp=np.copy(img)
    hdu_map,img_map,weight_map = load_Data(clustername)
    weight_map = weight_map / np.max(weight_map)
    temp[weight_map<0.2]=0
    return temp
    
def findPointSources_deprecated(filtered_img,num_src,mask=True,mask_rad=250):
    """"Takes in 4 parameters, a 2D array, number of sources, mask (Boolean) and radius of mask. Returns the pixel location of the max points in 2D array."""
    temp_data = np.copy(filtered_img)
    pointsrc_coords_x=[]
    pointsrc_coords_y=[]
    if mask == False:
        for i in range(num_src):
            center=np.where(temp_data==np.max(temp_data))
            pointsrc_coords_x=np.append(pointsrc_coords_x,center[0][0])
            pointsrc_coords_y=np.append(pointsrc_coords_y,center[1][0])
            xmin=center[0][0]-10
            xmax=center[0][0]+10
            ymin=center[1][0]-10
            ymax=center[1][0]+10
            temp_data[xmin:xmax,ymin:ymax]=0
    else:
        temp = maskOuterRing(temp_data,mask_rad)
        for i in range(num_src):
            center=np.where(temp==np.max(temp))
            pointsrc_coords_x=np.append(pointsrc_coords_x,center[0][0])
            pointsrc_coords_y=np.append(pointsrc_coords_y,center[1][0])
            xmin=center[0][0]-10
            xmax=center[0][0]+10
            ymin=center[1][0]-10
            ymax=center[1][0]+10
            temp[xmin:xmax,ymin:ymax]=0
    return pointsrc_coords_x,pointsrc_coords_y

def findPointSources(filtered_img,clustername,num_src):
    """"Takes in 4 parameters, a 2D array, number of sources, mask (Boolean) and radius of mask. Returns the pixel location of the max points in 2D array."""
    pointsrc_coords_x=[]
    pointsrc_coords_y=[]
    temp = maskByWeight(filtered_img,clustername)
    for i in range(num_src):
        center=np.where(temp==np.max(temp))
        pointsrc_coords_x=np.append(pointsrc_coords_x,center[0][0])
        pointsrc_coords_y=np.append(pointsrc_coords_y,center[1][0])
        xmin=center[0][0]-10
        xmax=center[0][0]+10
        ymin=center[1][0]-10
        ymax=center[1][0]+10
        temp[xmin:xmax,ymin:ymax]=0
    return pointsrc_coords_x,pointsrc_coords_y

def get_filenames(clustername):
    """"Obtains the list of the DoG maps generated with DoG_loop"""
    file_str=clustername+"_files"
    f = open(file_str,"r")
    filename_str_arr = np.genfromtxt(file_str,delimiter=',',dtype=str)
    return filename_str_arr

def Point_Srcs(clustername,dir_str,num_src):
    """"Takes in a cluster name (string), directory, number of sources (integer). Returns a dictionary of point sources of all DoG maps associated with cluster, and a dictionary of all the DoG maps."""
    DoG_diction={}
    filenames_arr=get_filenames(clustername)
    Point_src_diction={}
    for i in filenames_arr:
        full_name_str=dir_str+i
        array_data=np.loadtxt(full_name_str)
        DoG_diction[i]=array_data
        arr_ps_x,arr_ps_y=findPointSources(array_data,clustername,num_src)
        Point_src_diction[i]=[arr_ps_x,arr_ps_y]
    return DoG_diction,Point_src_diction

def kMeans_point_sources(clustername,dir_str,num_centers):
    """"Takes in a dictionary of point sources (keys are names for each DoG map, values are the pixel location of point sources) and returns the kmeans centers (point source location in )"""
    DoG_diction,Point_src_diction = Point_Srcs(clustername,dir_str,num_centers)
    keys=list(Point_src_diction.keys())
    X=[0,0]
    for i in range(len(keys)):
        arr_i=np.array(list(zip(Point_src_diction[keys[i]][0],Point_src_diction[keys[i]][1])))
        X=np.vstack((X,arr_i)) 
    X=X[1:]
    kmeans= KMeans(n_clusters=num_centers,random_state=0).fit(X)
    point_src_centers=kmeans.cluster_centers_
    ##############
    w=get_WCS_from_fits(clustername)
    coords_ra_dec=w.wcs_pix2world(point_src_centers,1)
    return DoG_diction,Point_src_diction,coords_ra_dec,point_src_centers

