from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
import numpy as np
from astropy.convolution import convolve

def diffOfGaussians(img,theta1,theta2):
    kernel_1 = Gaussian2DKernel(x_stddev=theta1)
    kernel_2 = Gaussian2DKernel(x_stddev=theta2)
    scipy_conv_theta1 = scipy_convolve(img, kernel_1, mode='same', method='direct')
    scipy_conv_theta2 = scipy_convolve(img, kernel_2, mode='same', method='direct')
    diff_gaussians=scipy_conv_theta1-scipy_conv_theta2
    return diff_gaussians

def load_Data(clustername):
    if clustername=="MOO1142":
        snr=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1142/MOO_1142_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_snr_one_q.fits")
        noise=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1142/MOO_1142_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_noise_one_qv.fits")
        mapD=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1142/MOO_1142_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_map_one_qv.fits")
    elif clustername=="MOO1014":  
        snr=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1014/MOO_1014_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_snr.fits")
        noise=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1014/MOO_1014_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_noise.fits")
        mapD=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1014/MOO_1014_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_map.fits")
    elif clustername=="M0717":
        snr=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/M0717/Kelvin_M0717_2asp_pca5_qm2_0f08_41Hz_qc_1p2rr_L_FebCals_dt20_snr_iter1.fits")
        noise=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/M0717/Kelvin_M0717_2asp_pca5_qm2_0f08_41Hz_qc_1p2rr_L_FebCals_dt20_noise_iter1.fits")
        mapD=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/M0717/Kelvin_M0717_2asp_pca5_qm2_0f08_41Hz_qc_1p2rr_L_FebCals_dt20_map_iter1.fits")
    elif clustername=="MOO1506":
        snr=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1506/snr_map.fits")
        noise=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1506/noise.fits")
        mapD=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1506/map.fits")
    hdu_snr = fits.open(snr)[0]
    img_snr = hdu_snr.data
    hdu_noise = fits.open(noise)[0]
    img_noise = hdu_noise.data
    hdu_map = fits.open(mapD)[0]
    img_map = hdu_map.data
    hdu_weight_map = fits.open(mapD)[1]
    weight_map = hdu_weight_map.data
    return hdu_map,img_map,weight_map

def DoG_loop(clustername,theta1_arr,theta2_arr):
    filenames_thetas_arr=[]
    fname= clustername+"_files"
    text_file=open(fname,"w")
    hdu_map,img_map,weight_map=load_Data(clustername)
    for i in theta1_arr:
        for j in theta2_arr:
            if i<j:
                filename="diff_Gauss_img_map_"+clustername+"_theta1_"+str(i)+"__theta2_"+str(j)
                n_str="_theta1_"+str(i)+"__theta2_"+str(j)
                result=diffOfGaussians(img_map,i,j)
                np.savetxt(filename,result)
                filenames_thetas_arr = np.append(filenames_thetas_arr,n_str)
                text_file.write("%s\n" % n_str)
    text_file.close()
    return filenames_thetas_arr
