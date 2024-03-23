import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from skimage import io
from skimage.color import rgb2gray
from skimage.io import imsave
from skimage import exposure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dpc_algorithm import DPCSolver

data_path  = "C:/Users/Zaid/Desktop/DPC_Code/sample_data/" #INSERT YOUR DATA PATH HERE
image_list = listdir(data_path)
image_list = [image_file for image_file in image_list if image_file.endswith(".jpg")]
image_list.sort()
#dpc_images = np.array([io.imread(data_path+image_list[image_index]) for image_index in range(len(image_list))])
dpc_images = np.array([rgb2gray(io.imread(data_path+image_list[image_index])) for image_index in range(len(image_list))])

#plot first set of measured DPC measurements
f, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
for plot_index in range(4):
    plot_row = plot_index//2
    plot_col = np.mod(plot_index, 2)
    ax[plot_row, plot_col].imshow(dpc_images[plot_index], cmap="gray",\
                                  extent=[0, dpc_images[0].shape[-1], 0, dpc_images[0].shape[-2]])
    ax[plot_row, plot_col].axis("off")
    ax[plot_row, plot_col].set_title("DPC {:02d}".format(plot_index))

wavelength     =  0.52 #micron
mag            =   2.5
na             =   0.28 #numerical aperture
na_in          =    0.0
pixel_size_cam =    1.4 #pixel size of camera
dpc_num        =      4 #number of DPC images captured for each absorption and phase frame
pixel_size     = pixel_size_cam/mag
rotation       = [0, 180, 90, 270] #degree


dpc_solver_obj = DPCSolver(dpc_images, wavelength, na, na_in, pixel_size, rotation, dpc_num=dpc_num)

#parameters for Tikhonov regurlarization [absorption, phase] ((need to tune this based on SNR)
dpc_solver_obj.setRegularizationParameters(reg_u = 1e-1, reg_p = 5e-7)
dpc_result = dpc_solver_obj.solve()

# Assuming dpc_result is the output with shape (n, height, width) where n is the number of sets of images processed
# And each set contains a complex number where the real part is absorption and the imaginary part is phase

for i, result in enumerate(dpc_result):
    # Normalize the real and imaginary parts separately
    absorption = exposure.rescale_intensity(result.real, out_range=(0, 1))
    phase = exposure.rescale_intensity(result.imag, out_range=(0, 1))

    # Convert to 8-bit (this is what imshow does implicitly)
    absorption_8bit = exposure.rescale_intensity(absorption, out_range=(0, 255)).astype(np.uint8)
    phase_8bit = exposure.rescale_intensity(phase, out_range=(0, 255)).astype(np.uint8)

    # Save the images
    imsave(f"C:/Users/Zaid/Desktop/DPC_Code/output/absorption_{i:02d}.png", absorption_8bit)
    imsave(f"C:/Users/Zaid/Desktop/DPC_Code/output/phase_{i:02d}.png", phase_8bit)
