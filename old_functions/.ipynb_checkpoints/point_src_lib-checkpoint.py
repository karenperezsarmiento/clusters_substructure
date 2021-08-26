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

def get_WCS_from_fits(hdu_map):
    """Takes in a string (clustername) as parameter, finds the associated fits files and returns the world coordinates"""
    hdu_map,img_map,weight_map= load_Data(clustername)
    w=wcs.WCS(hdu_map.header)
    return w

def maskByWeight(img,clustername):
    temp=np.copy(img)
    hdu_map,img_map,weight_map = load_Data(clustername)
    weight_map = weight_map / np.max(weight_map)
    temp[weight_map<0.2]=0
    return temp

def get_filenames(clustername):
    """"Obtains the list of the DoG maps generated with DoG_loop"""
    file_str=clustername+"_files"
    filename_str_arr = np.genfromtxt(file_str,delimiter=',',dtype=str)
    return filename_str_arr

def psrc_finder(dog_map,clustername):
    masked_map = maskByWeight(dog_map,clustername)
    max_mag = np.max(masked_map)
    current_std = np.std(masked_map.flatten())
    current_avg = np.mean(masked_map.flatten())
    rel_mag = max_mag/current_std
    pointsrc_coords_x=[]
    pointsrc_coords_y=[]
    center=np.where(masked_map==np.max(masked_map))
    pointsrc_coords_x=np.append(pointsrc_coords_x,center[0][0])
    pointsrc_coords_y=np.append(pointsrc_coords_y,center[1][0])
    xmin=center[0][0]-10
    xmax=center[0][0]+10
    ymin=center[1][0]-10
    ymax=center[1][0]+10
    masked_map[xmin:xmax,ymin:ymax]=0
    new_avg = np.mean(masked_map.flatten())
    new_std = np.std(masked_map.flatten())
    new_rel_mag = max_mag/new_std
    num_src = 1
    std_arr=[]
    std_arr=np.append(std_arr,new_std)
    source_mag = []
    source_mag = np.append(source_mag,max_mag)
    while num_src<10:
        max_mag = np.max(masked_map)
        current_std = np.std(masked_map.flatten())
        current_avg = np.mean(masked_map.flatten())
        rel_mag = max_mag/current_std
        center = np.where(masked_map==np.max(masked_map))
        pointsrc_coords_x=np.append(pointsrc_coords_x,center[0][0])
        pointsrc_coords_y=np.append(pointsrc_coords_y,center[1][0])
        xmin=center[0][0]-10
        xmax=center[0][0]+10
        ymin=center[1][0]-10
        ymax=center[1][0]+10
        masked_map[xmin:xmax,ymin:ymax]=0
        new_avg = np.mean(masked_map.flatten())
        new_std = np.std(masked_map.flatten())
        new_rel_mag = max_mag/new_std
        num_src+=1
        std_arr = np.append(std_arr,new_std)
        source_mag = np.append(source_mag,max_mag)
    src = np.arange(len(std_arr)-2)
    src = src[np.diff(np.diff(std_arr))>0]
    src = src[0]+2
    return src,pointsrc_coords_x[0:src],pointsrc_coords_y[0:src]

def all_Point_Srcs(clustername,dir_str):
    """"Takes in a cluster name (string), directory, number of sources (integer). Returns a dictionary of point sources of all DoG maps associated with cluster, and a dictionary of all the DoG maps."""
    DoG_diction={}
    filenames_arr= get_filenames(clustername)
    Point_src_diction={}
    num_src_arr = []
    for i in filenames_arr:
        full_name_str=dir_str+i
        array_data=np.loadtxt(full_name_str)
        DoG_diction[i]=array_data
        num_src,arr_ps_x,arr_ps_y=psrc_finder(array_data,clustername)
        Point_src_diction[i]=[arr_ps_x,arr_ps_y]
        num_src_arr = np.append(num_src_arr,num_src)
    return num_src_arr, DoG_diction,Point_src_diction

def kMeans_point_sources(clustername,dir_str):
    """"Takes in a dictionary of point sources (keys are names for each DoG map, values are the pixel location of point sources) and returns the kmeans centers (point source location in )"""
    num_src_arr,DoG_diction,Point_src_diction = all_Point_Srcs(clustername,dir_str)
    num_centers = int(max(num_src_arr))
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
