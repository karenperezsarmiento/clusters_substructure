
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

snrMOO1142=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1142/MOO_1142_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_snr_one_q.fits")
noiseMOO1142=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1142/MOO_1142_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_noise_one_qv.fits")
mapMOO1142=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1142/MOO_1142_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_map_one_qv.fits")
hdu_snr_MOO1142 = fits.open(snrMOO1142)[0]
img_snr_MOO1142 = hdu_snr_MOO1142.data
hdu_noise_MOO1142 = fits.open(noiseMOO1142)[0]
img_noise_MOO1142 = hdu_noise_MOO1142.data
hdu_map_MOO1142 = fits.open(mapMOO1142)[0]
img_map_MOO1142 = hdu_map_MOO1142.data

snrM0717=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/M0717/Kelvin_M0717_2asp_pca5_qm2_0f08_41Hz_qc_1p2rr_L_FebCals_dt20_snr_iter1.fits")
noiseM0717=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/M0717/Kelvin_M0717_2asp_pca5_qm2_0f08_41Hz_qc_1p2rr_L_FebCals_dt20_noise_iter1.fits")
mapM0717=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/M0717/Kelvin_M0717_2asp_pca5_qm2_0f08_41Hz_qc_1p2rr_L_FebCals_dt20_map_iter1.fits")
hdu_snr_M0717 = fits.open(snrM0717)[0]
img_snr_M0717 = hdu_snr_M0717.data
hdu_noise_M0717 = fits.open(noiseM0717)[0]
img_noise_M0717 = hdu_noise_M0717.data
hdu_map_M0717 = fits.open(mapMOO1142)[0]
img_map_M0717 = hdu_map_M0717.data


snr_MOO1014=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1014/MOO_1014_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_snr.fits")
noise_MOO1014=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1014/MOO_1014_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_noise.fits")
map_MOO1014=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1014/MOO_1014_cmsub_cmscale_ffilt0.08_41_deglitch-7.5_map.fits")
hdu_snr_MOO1014 = fits.open(snr_MOO1014)[0]
img_snr_MOO1014 = hdu_snr_MOO1014.data
hdu_noise_MOO1014 = fits.open(noise_MOO1014)[0]
img_noise_MOO1014 = hdu_noise_MOO1014.data
hdu_map_MOO1014 = fits.open(map_MOO1014)[0]
img_map_MOO1014 = hdu_map_MOO1014.data

snr_MOO1506=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1506/snr_map.fits")
noise_MOO1506=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1506/noise.fits")
map_MOO1506=get_pkg_data_filename("/users/ksarmien/Documents/clusters_substructure/MOO1506/map.fits")
hdu_snr_MOO1506 = fits.open(snr_MOO1506)[0]
img_snr_MOO1506 = hdu_snr_MOO1506.data
hdu_noise_MOO1506 = fits.open(noise_MOO1506)[0]
img_noise_MOO1506 = hdu_noise_MOO1506.data
hdu_map_MOO1506 = fits.open(map_MOO1506)[0]
img_map_MOO1506 = hdu_map_MOO1506.data

theta1_arr=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5]
theta2_arr=[8.75,10,15,20,25,30,35,40,45,50]
cluster_list=[img_map_MOO1142,img_map_M0717,img_map_MOO1014,img_map_MOO1506]
cluster_list_names=["img_map_MOO1142","img_map_M0717","img_map_MOO1014","img_map_MOO1506"]
for k in range(len(cluster_list_names)):
    for i in theta1_arr:
        for j in theta2_arr:
            filename="diff_Gauss_"+cluster_list_names[k]+"_theta1_"+str(i)+"__theta2_"+str(j)
            result=diffOfGaussians(cluster_list[k],i,j)
            np.savetxt(filename,result)

