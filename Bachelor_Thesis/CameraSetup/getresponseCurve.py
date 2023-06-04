#!/usr/bin/python3

#%matplotlib widget
import os
from typing import List, Tuple

import cv2
import imageio 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
#from flt_reader import read_flt_image
from tqdm import tqdm
import json

from numba import jit

def readfile(filename):

    try:

        filename = str(filename.encode("unicode_escape").decode('utf-8'))

        image = imageio.imread(filename)

    except:

        print("Datei " , filename , " kann nicht geoeffnet werden.")

        raise 

        sys.exit()

    return image




# load the ldr image list
#with open("max.list", "r") as exposure_list_file:
#    file_content = exposure_list_file.readlines()
#exposures = [s.rstrip().split(" ") for s in file_content]
# load the images and extract exposure time (ms)
#images = [imageio.imread(os.path.join("input_images", exp[0])) for exp in exposures]
#exposure_times = [1 / float(exp[1]) for exp in exposures]

#images = [0] * 3 # initialize array
images = [0] * 21 # initialize array


#exposure_times = [3001, 24999,149999]
exposure_times = [ 7, 98 , 197 , 499 , 998 , 1997 , 4999 , 9997 , 20001 , 40002 , 99999 , 149999 , 199998 , 299997 , 500002 , 899998 , 1.2e+06 , 1.8e+06 , 2.5e+06 , 3e+06 , 7e+06 , 1e+07]

#my load ldr images
def readldr(): 
    for i in range(len(images)): #j, i in enumerate([2,5,8]):
        readoutpath = f"ExpCurve_{i}.exr"
        pic = readfile(readoutpath)
        #import pdb;pdb.set_trace()
        pic[pic > 1] = 1
        pic = (pic * (2**12 - 1)).astype(np.uint64)
        #print(f"pic min = {np.min(pic)} , pic max = {np.max(pic)}")
        images[i] = pic
        
readldr()

### load ground truth result image
##hdr_gt = readfile("reconstructed.jpg")
##
##
##print("LDR input images:")
##fig, axs = plt.subplots(2, len(images) // 2, figsize=[8, 2])
##for j in range(2):
##    for i in range(len(images) // 2):
##        idx = j * len(images) // 2 + i
##        
##        axs[j][i].imshow(images[idx])
##        axs[j][i].set_axis_off()
##        axs[j][i].set_title("{:.3f} s".format(exposure_times[idx]))
##
##fig.set
##print("Tonemapped ground truth hdr image (expected outcome):")
##fig, ax = plt.subplots(1, 1, figsize=[4, 3])
##ax.imshow(hdr_gt)
##ax.set_axis_off()
##ax.set_title("Reconstructed HDR image (tonemapped)")
##fig.show()
###plt.show()

#@jit(nopython=True)
def weight(pixel_values: np.ndarray) -> np.ndarray:
    """Gaussian weighting function Equation (5) in Paper.
    Args:
        pixel_values: Input values of arbitrary shape.
    Returns:
        Gaussian weights for given input according to gaussian function.
        The output array will have the same shape as the input.
    """
    #This was changed from 127.5 to 2047.5 due to the fact that we have 12 bit instead of 8
    
    result = np.zeros_like(pixel_values)
    mask = pixel_values >= (2**12) - 1
    result = np.exp(-4 * (pixel_values - 2047.5) ** 2 / (2047.5) ** 2)
    result[mask] = 0
    return result


fig_weight, ax = plt.subplots(1, 1, figsize=[4, 3])
ax.plot(weight(np.arange(4096))) #does this need to be 2048?
ax.set_title("Weight Function")
ax.set_xlabel("Pixel Value")
ax.set_ylabel("Weight")
fig_weight.tight_layout()
#fig_weight.show()
#plt.show()
fig_weight.savefig("Weighting Function")

@jit(nopython=True)
def normalize(response: np.ndarray) -> np.ndarray:
    """Normalizes the response curve values such that the median is 1.
    See Paper text after Equation 9
    Args:
        response: Input response values of shape [N]. (Monotonically increasing values)
    Returns:
        Normalized response of shape [N],
        such that result[median_idx] == 1.
    """


    median = np.median(response)
    return response / median


# Uncomment this assertion to check if your implementation is correct. Check your code if you get an Assertion error!
assert np.all(normalize(np.array([1, 2, 3])) == np.array([0.5, 1.0, 1.5]))

def show_I(response_curve: np.ndarray, response_curve2: np.ndarray, hdr: np.ndarray, imgname) -> None:
    
    hdr = hdr.astype(np.float32)
    tonemap = cv2.createTonemap(2.2)
    ldr = tonemap.process(hdr)
    
    figure, axis = plt.subplots(1, 3, figsize=(19,8))
    
    
    axis[0].plot(np.arange(response_curve.shape[1]), response_curve[0], "r-")
    axis[0].plot(np.arange(response_curve.shape[1]), response_curve[1], "g-")
    axis[0].plot(np.arange(response_curve.shape[1]), response_curve[2], "b-")
    axis[0].set_title("I")

    axis[1].plot(np.arange(response_curve.shape[1]), response_curve2[0], "r-")
    axis[1].plot(np.arange(response_curve.shape[1]), response_curve2[1], "g-")
    axis[1].plot(np.arange(response_curve.shape[1]), response_curve2[2], "b-")
    axis[1].set_title("y in log")
    
    axis[2].imshow(ldr)
    axis[2].set_axis_off()
    axis[2].set_title("HDR (tonemapped)")
    
    figure.savefig(imgname)

def show_hdr(response_curve: np.ndarray, hdr: np.ndarray, imgname) -> None:
    """Utility function for showing a HDR image on a LDR screen.

    This function uses a tonemapping function implemented in the open-cv library.
    Args:
        response_curve:
            The response curves (I) that were used to reconstruct the HDR image.
            Shape is [C, 4096] - one response-curve per channel.
        hdr:
            The HDR iamge that is supposed to be shown.
    """
    if not np.all(response_curve.shape == np.array([3, 4096])):
        raise ValueError(
            "Response curve must have shape [3, 4096]! One response curve for each channel."
        )
    hdr = hdr.astype(np.float32)
    tonemap = cv2.createTonemap(2.2)
    ldr = tonemap.process(hdr)
    
    fig_hdr, axs = plt.subplots(1, 2, figsize=[8, 3])
    axs[0].semilogx(response_curve[0], np.arange(response_curve.shape[1]), "r-")
    axs[0].semilogx(response_curve[1], np.arange(response_curve.shape[1]), "g-")
    axs[0].semilogx(response_curve[2], np.arange(response_curve.shape[1]), "b-")
    axs[0].set_title("Response Curves (rgb)")
    axs[1].imshow(ldr)
    axs[1].set_axis_off()
    axs[1].set_title("HDR (tonemapped)")
    fig_hdr.savefig(imgname)
    
#    plt.show()

@jit(nogil=True)
def robertson_apply_response(
    ldr_images: List[np.ndarray], times: List[float], I: np.ndarray
) -> np.ndarray:
    """Computes HDR image from ldr images by applying the given response curve I for one channel.
    See paper Equation (8).
    Args:
        ldr_images:
            A list of ldr images of length N and shape [H, W]. <-- should be for all 3 channels
            Expected type: np.uint8
            Expected value range: [0, 255] <-- this must be 2^12
        times:
            A list of exposure times of length N.
        I:
            The response curve lookup table for one channel.
            Array of shape [256].
        channel:
            The chanel to reconstruct. In range [0, C[
    Returns:
        The hdr image of shape [H, W]
    """
    image_shape = ldr_images[0].shape
    nominator = np.zeros([image_shape[0], image_shape[1]], dtype=np.float32)
    denominator = np.zeros([image_shape[0], image_shape[1]], dtype=np.float32)
    
    for i in range(len(images)):
        time = times[i]
        image = ldr_images[i]
        print(f"max = {np.max(image)} , min= {np.min(image)}")
        print(f"for {i} time is {time}")
        weighted_image = weight(image)
        nominator += time * weighted_image * I[image]# I[image]???
        denominator += time ** 2 * weighted_image
    hdr = nominator / denominator
    return hdr


# Lets test your code. This is a linear response curve with values between 0 and 2
I = np.linspace(0, 2, 4096)
hdr_linear = robertson_apply_response(images, exposure_times, I)
# hdr_linear_g = robertson_apply_response(images, exposure_times, I, 1)
# hdr_linear_b = robertson_apply_response(images, exposure_times, I, 2)
show_hdr(
    np.stack([I, I, I], 0), np.stack([hdr_linear, hdr_linear, hdr_linear], -1) , "fig_linear.png"
) # STACK RESPONSE CURVE __________ STACK HDR PICTURE




@jit(nogil=True)
def robertson_get_response(
    ldr_images: List[np.ndarray],
    times: List[float],
    hdr_image: np.ndarray,
) -> np.ndarray:
    """Estimate the camera response curve given a ldr-sequence with exposure times and a corresponding hdr_image.
    See Equation (10)
    Args:
        ldr_images:
            List of LDR images captured by camera. Length N, image shape [H, W, C].
            Expected type: np.uint8
            Expected value range: [0, 255]
        times:
            Exposure times corresponding to each LDR-image. Length N.
        hdr_image:
            A HDR image for the current channel that was reconstructed from the LDR-images. Shape [H, W].
        channel:
            The channel whose response curve is supposed to be estimated.
    Returns:
        An estimate for the response curve for the given channel.
        The returned curve is guaranteed to be monotone.
    """
    cardinality = np.zeros(2**12, dtype=np.int64) 
    nominator = np.zeros(2**12, dtype=np.float64) 

    for i in range(len(ldr_images)): #print progress takes the longest
        image = ldr_images[i]
        time = times[i]
        #print("i:", i)
        for j in range(2**12):
            #print("j:", j)
            selected_pixels = image == j 
            #print(f"selected pixels : {selected_pixels}")
            #print(f"image : {image}")
            cardinality[j] += np.sum(selected_pixels)
            #import pdb;pdb.set_trace()
            nominator[j] += time * np.sum(hdr_image[selected_pixels])
    
    mask_card = cardinality == 0
    I = nominator / cardinality
    I[mask_card] = 0
    print(f"I after I[mask_card] = {I}")
    print(f"sum(mask_card) {np.sum(mask_card)}")
    
    #for i in range(2**12):  
    #    if cardinality[i] == 0 or I[i - 1] > I[i]:
    #        I[i] = I[i - 1] #evtl fehlerquelle, muss vielleicht raus, verbessert werden
    print(f"Cardinality is: {cardinality}")
    return I

def robertson_unknown_response(
    ldr_images: List[np.ndarray], times: List[float], I: np.ndarray, 
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimizes the response curve by iteratively reconstructing the hdr with
    the current response curve and then refining the response curve values.
    Args:
        ldr_images:
            A list of ldr images of length N and shape [H, W, C].
            Expected type: np.uint8,
            Expected value range: [0, 4095]
        times:
            A list of exposure times of length N.
        I:
            Initial response curve for the current channel of shape [4096].
    Returns:
        A tuple of the optimized response curve and the reconstructed hdr image:
        (response_curves, hdr_image), where response_curve has shape [4096]
        and hdr_image has shape [H, W].
    """
    # Optimization stops after this many iterations
    max_iterations = 100
    # if the sum of squared differences between new and previous response curve are smaller than this stop optimization
    max_delta = 5e-8

    # initialize
    I = normalize(I)
    # Do a deep copy of the initial response curve.
    # Note: a simple I_previous = I will not copy the array!!
    # We keep track of the old response curve to check the stopping condition in the end.
    I_previous = I.copy()
    hdr_image = robertson_apply_response(ldr_images, times, I)

    # optimization
    delta = max_delta + 1
    print(f"delta is: {delta}")
    
    bar = tqdm(range(max_iterations), desc="optimization")
    
    for iteration in bar:
        # step 1: Minimize with respect to I
        I = robertson_get_response(ldr_images, times, hdr_image)

        # step 2: normalize I
        I = normalize(I)    

        #smooth the curve
        xs = np.arange(len(I_result))
        
       # I = sp.signal.savgol_filter(I, len(I), 5)
       
       # I = sp.signal.wiener(I, 64)
       # I = sp.signal.medfilt(I, 9)
        I = sp.ndimage.gaussian_filter1d(I , 5, mode='nearest')
        
        I_old = I.copy()
        
        
        # step 3: apply new response (minimize wrt. hdr_image)
        hdr_image = robertson_apply_response(ldr_images, times, I)

        show_I(np.stack([I_old, I_old, I_old] , 0), np.stack([I, I, I], 0), np.stack([hdr_image, hdr_image, hdr_image], -1), f"I_{iteration}")
        
        # step 4: check stopping condition
        delta = np.sum((I - I_previous) ** 2)
        bar.postfix = "delta = {:.2e}".format(delta)
        if delta < max_delta:
            print(f"delta followed by break: {delta}")
            break
        I_previous = I.copy()
        print(f"delta is: {delta}")

    return (I, hdr_image)


# We again start with a linear response curve, but this time we optimize for the correct respone curve.
I_init = np.linspace(0, 2, 4096) #does this need to be 2048?
# Call your robertson_unknown_response for each channel to reconstruct both the color hdr image as well as the response curves for each channe.
I_result = np.zeros(4096, np.float32) 
hdr = np.zeros(images[0].shape, np.float32)

I_result, hdr = robertson_unknown_response(
    images, exposure_times, I_init
)

#I_result = sp.signal.savgol_filter(I_result, len(I_result), 14)
#I_result = sp.signal.wiener(I_result, 128)
#I_result = sp.signal.savgol_filter(I_result , len(I_result), 21)

with open("I_resultg.txt", 'w') as g:
    json.dump(I_result.tolist(), g, allow_nan=True)

show_hdr(np.stack([I_result, I_result, I_result], 0), np.stack([hdr, hdr, hdr], -1), "fig_result.png")
#new step to improve the curve 

xs = np.arange(len(I_result))
#xs = xs
#ys = I_result 
#p = np.polynomial.polynomial.polyfit(xs, ys, [1,2,3,4,5,6,7])
#polynom_fit = np.poly1d(p)
#I = polynom_fit(np.arange(2**12))



hdr_image = robertson_apply_response(images, exposure_times, I)

imageio.imwrite("resultingimg.exr" , hdr_image)

show_hdr(np.stack([I, I, I], 0), np.stack([hdr_image, hdr_image, hdr_image], -1), "fig_smooth.png")

