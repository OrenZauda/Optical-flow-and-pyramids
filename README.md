# Optical-flow-and-pyramids

## methods:

1. Lucas kanade - this function consume two consecutive images, window size
and step size and return the optical flow between the two puctures

2. laplaceianReduce - this function consume image and number of levels and create the 
laplacian pyramid of the image

3. laplaceianExpand - this function retrieve the original image from laplacian list

4. gaussianPyr - this function consume image and number of levels and create the 
gaussian pyramid of the image

5. gaussExpand - this function retrieve the original image from gaussian list

6. pyrBlend - this function blending two images using pyramids
