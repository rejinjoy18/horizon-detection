
import numpy as np
import cv2
import math
from tqdm import tqdm
from scipy import stats


def horizon_detect(path, scale = 50):

    #### function that returns the y values of a line given slope and intercept ####
    def line(m,b,x):

        y = m*x + b
        return y    
    
    #### reading image
    image = (cv2.imread(path))
	
	#### shape of image    
    y_axis = image.shape[0]
    x_axis = image.shape[1]

	#### Scaling factor to downsample the image (for faster processing; n and n2 can be increased to 
	#### decrease processing time, at a relatively negligible cost of accuracy

	   

    n = int(x_axis/scale)
    n2 = int(y_axis/scale)

    img = cv2.resize(image, (int(x_axis/n), int(y_axis/n2)))

	#### shape of image after downsampling
    y_axis = img.shape[0]
    x_axis = img.shape[1]

	#### range of slope and intercept values considered. Resolution of this scan can be changed by
	#### altering values here
    m = np.linspace(-1,1,40)
    b = np.linspace(0, y_axis,40)
	

	#### initializing variables
    maximum = [];
    J1 = 0

	#### two for loops running for every combination of slope and y-intercept
    for mm in tqdm(range(len(m))):
        for bb in range(len(b)):
            y = []
            x = []
            xs = []    #### array of pixel values containing sky
            xg = []	   #### array of pixel values containing ground
            CovS = []  #### covariance of xs
            CovG = []  #### covariance of xg
            
	#### for a given slope and intercept, finding points along that line
            for i in range(x_axis):
                y.append((line(m[mm],b[bb],i)))
                x.append(i)
			
	### value used for cross product
            v1 = [x[2-1]-x[0], y[2-1]-y[0]]


	#### for loop running along every pixel value across the image
            for i in range(x_axis):
                for j in range(y_axis):
	#### cross product to find every pixel value above and below that line
                    v2 = [x[2-1]-i, y[2-1]-j]
                    xp = v1[0]*v2[1] - v1[1]*v2[0]
                    if xp>0:
                        xs.append([img[j,i,0], img[j,i,1], img[j,i,2]])
                    else:
                        xg.append([img[j,i,0], img[j,i,1], img[j,i,2]])
            

	### finding covariance and eigenvalues of xs (pixels containng sky)
	### and xg (pixels containing ground)
            xs = np.transpose(xs)
            xg = np.transpose(xg)
            try:

                lol = np.cov(xs)
                lol1 = np.cov(xg)


                CovS = np.linalg.det(lol)
                CovG = np.linalg.det(lol1)

                eig_vs, _ = np.linalg.eig(lol)
                eig_vg, _ = np.linalg.eig(lol1)
            
            
            
                J = 1/(CovS + CovG + pow((eig_vs[0]+eig_vs[1]+eig_vs[2]),2) +  pow((eig_vg[0]+eig_vg[1]+eig_vg[2]),2))
                
	### finding maximum value of J across all slopes and intercepts
                if J > J1:
                    maximum = [m[mm],b[bb]]
                    J1 = J
                    
            except Exception:
                pass
  
	##### end of first part
	
	#### drawing line across given original image at point of horizon
    
	### slope and intercept obtained from earlier
    m = maximum [0]
    b = maximum [1]
	
	### scaling the intercept back upto original size of image
    b = b*n2
	
	### width of original image
    x_axis = image.shape[1]
    y=[]
    x=[]

    for i in range(x_axis):
        y.append((line(m,b,i)))
        x.append(i)

    cv2.line(image, (int(x[0]), int(y[0])), (int(x[x_axis-1]), int(y[x_axis-1])), color =2, thickness = 2)

    cv2.imwrite('horizon_'+path, image)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("path", help = "path to image file")
    parser.add_argument("--scale", help = "size you want to downgrade image to. Larger value->faster processing (>20)")
    args = parser.parse_args()
    
    path = args.path
    scale = int(args.scale)
    horizon_detect(path, scale)

