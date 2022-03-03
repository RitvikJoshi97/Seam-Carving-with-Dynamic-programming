from numpy.lib.function_base import append, delete
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from operator import index, itemgetter, length_hint
import sys
from PIL import Image as im

## Get aguments
input_image = sys.argv[1]
output_image = sys.argv[2]
n = int(sys.argv[3])

## Import file
img = cv.imread(input_image)
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("title",grayImage)
gray_img_data = np.asarray(grayImage)
gray_img_data = np.ndarray.tolist(gray_img_data)


# laplacian = cv.Laplacian(img,cv.CV_64F)
# laplacianOfGreyImage = cv.Laplacian(grayImage,cv.CV_64F) 
# lap_img_data = np.asarray(laplacianOfGreyImage)
# lap_img_data = np.ndarray.tolist(lap_img_data)






## Find the edges 
def findEdge(data, i, j, power = 0.0, edge=[]):
    if i > height_of_img:
        final_power.append(power)
        return 
    else:
        if j+1 > (len(data[0])-1):
            least = min(data[i][j-1], data[i][j]) 
        elif j-1 < 0:
            least = min(data[i][j], data[i][j+1])
        else:
            least = min(data[i][j-1], data[i][j], data[i][j+1])



        if least == data[i][j-1]:
            power += data[i][j-1]
            edge.append([i,j-1])
            findEdge(data, i+1, j-1, power, edge)
        elif least == data[i][j]: 
            power += data[i][j]
            edge.append([i,j])
            findEdge(data, i+1, j, power, edge)
        else:
            power += data[i][j+1]
            edge.append([i,j+1])
            findEdge(data, i+1, j+1, power, edge)


def reorder(data):
    return sorted(data, key=itemgetter(0))

def removeEdge(data, edge):
    for i in range(len(edge)):
        data[edge[i][0]].pop(edge[i][1])
    


def removeEdgeData(data, index):
    data = data[:index-1]
    return data


def checkForIndex(index, all_edges):
    for i in range(len(all_edges)):
        if all_edges[i][2] == index:
            return True
        else:
            return False


## Run for n number of seam/edge carvings 
all_edges = []
while n > 0:
    index = 0
    i = 0
    height_of_img = len(gray_img_data) - 1
    width_of_img = len(gray_img_data[0]) -1
    for j in range(width_of_img):
        final_power = []
        edge = []
        if checkForIndex(index, all_edges): #Only those indexes for which edge data is not available should be updated
            index += 1 
        else: # new edge data is calculated if edge is not present in all_edges
            findEdge(gray_img_data, i, j, edge= edge)
            all_edges.append([final_power[0], edge, index]) # stores the edges in order to use in next iteration
            index += 1

    ## order the edges after the new edges are added
    sorted_list = reorder(all_edges)
    
    ## Carves out the seam in the image
    index = sorted_list[0][2]
    removeEdge(gray_img_data, sorted_list[0][1])

    ## Removes the edges from the storage after the index because those will be affected by the seam which is carved out
    all_edges =  removeEdgeData(all_edges,index)

    
    n -= 1



plt.imsave(output_image, gray_img_data)
