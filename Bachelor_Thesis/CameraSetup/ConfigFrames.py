import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2 as cv
import sys
import imageio as iio
iio.plugins.freeimage.download()

def readfile(filename):
    try:
        filename = str(filename.encode("unicode_escape").decode('utf-8'))
        image = iio.imread(filename)
    except:
        print("Datei " , filename , " kann nicht geoeffnet werden.")
        raise 
        sys.exit()
    return image



# ============== READOUT NOISE ==================== 1.Step
def get_mu_r(path):
    pic = readfile(path)
    return np.mean(pic)

def get_variance(path):
    pic = readfile(path)
    return np.var(pic)

# =============== AVERAGE(OMEGA) of formula (16) ===========
#average all ff for each gain 
def averagepic(pic):       
    return np.mean(pic) #average / (height * width)

#================== Subtract ff_1-ff_2 ======================
def subtractff(pic1,pic2):
    height, width = pic1.shape
    for i in range(height):
        for j in range(width):
                pic1[i][j] = pic1[i][j] - pic2[i][j]
    return pic1          
# =======================CAMERA GAIN 4.Step  =======================    
def CameraGain(rangepictures):
    gaincounter = 0
    avgcounter = 0     
    result = 0
    f = open("/home/stratman/Documents/code_public/light_stage_test/configframesresults/CG.txt", "w")
    f.write("{'CG_Values' : [")
    for i in range(11):
        pathreadout = f"BiasFrame_{gaincounter}.exr"
        mu_r = get_mu_r(pathreadout)
        sigma_r = get_variance(pathreadout)
        avg_omega_ff = averageGoP(grouppics("FF", gaincounter) , None)
        for j in range(rangepictures):
            path1 = f"gain_{gaincounter}_{j}.exr"
            for k in range(rangepictures):
                if (k == j):
                    continue
                avgcounter += 1
                path2 = f"gain_{gaincounter}_{k}.exr"
                pic1 = readfile(path1)
                pic2 = readfile(path2)
                pic = 0.5 * np.var(pic1 - pic2) - sigma_r 
                print(f"Variance of Subtraction is {np.var(pic1 - pic2)}")
                print("sigmaR is : " , sigma_r)
                print("mu_r is : " , mu_r)
                print(f"Pic is {pic}")
                print(f"avg_omega_ff {avg_omega_ff}")
                pic = pic / (avg_omega_ff - mu_r)
                print("Pic nach division ist : ", pic)
                
                
                
                print(f"Result of pic = {pic}")
                if(j == 0 and k == 0):
                    result = pic

                else:
                    result = result + pic
                
                
        print("averagecounter = ",avgcounter)       
        result = result / avgcounter 
        avgcounter = 0
        writeout = "{ 'GainLvL' : " + str(gaincounter) + " ,\n 'CG_Value' : " + str(result) + "}"

        if(gaincounter != 20):
            f.write(",")
            
        f.write(writeout)
        print(f"====================== \n Gain {gaincounter} result is = {result} \n ========================= ")
        gaincounter += 2
        result= 0 
    f.write("]}")
    f.close()
    print("CameraGain is done and was written")
#=================================================================

#Use FF or DF do get Darkframes and Flatfield Images for Pictures
def grouppics(name, gain):
    pics = []
    if (name == "DF"):
        gainlevels = 0
        for i in range(20):
            path =  f"df_gain_{gain}_{i}.exr"
            pic = readfile(path)
            pics.append(pic)
    if (name == "FF"):
        gainlevels = 0
        for i in range(20):
            path = f"gain_{gain}_{i}.exr"
            pic = readfile(path)
            pics.append(pic)
    return pics                
      


def averageGoP(pictures, axis=0):
    pic = pictures[0]

    toaverage = np.array(pictures)
    result = np.mean(toaverage, axis=axis)       
    return result   
# ================ PRNU Equation 14 ================
def prnu():
    gainlvl = 0
    f = open("/home/stratman/Documents/code_public/light_stage_test/configframesresults/prnu_paths.txt", "w")
    f.write("{ 'PRNU_PATHS': [")
    for i in range(11):
        df = averageGoP(grouppics("DF" , gainlvl))
        ff = averageGoP(grouppics("FF" , gainlvl))
        height, width= ff.shape
        subtracted = subtractff(df, ff)
        omega = height * width
        
        
        result = subtracted / np.mean(subtracted)
               
        print(f"PRNU for Gain {gainlvl} is done!")
        cv.imwrite(f"/home/stratman/Documents/code_public/light_stage_test/configframesresults/a_j_{gainlvl}.exr" , result)
        mu_r = np.mean(result)
        sigma_r = np.var(result)
        if(gainlvl != 0):
            f.write(",")
        f.write("{'gainlevel' : " + str(gainlvl) + " ,\n  'mu_r' : " + str(mu_r) + " ,\n 'sigma_r' : " + str(sigma_r) + ",\n 'prnu_path': '/home/stratman/Documents/code_public/light_stage_test/configframesresults/a_j_{" + str(gainlvl) + "}.exr'}")
        gainlvl += 2
    f.write("]}")   
    f.close()

if __name__=="__main__":   
    CameraGain(20) #how many pictures are being considered for the calculation? max is 20
    prnu()
