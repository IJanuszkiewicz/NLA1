## Numerical Linear Algebra A.A. 2026/2027

Lecturer: Prof. P. F. Antonietti

TA: Dott. M. Botti

## Challenge 1 - Image filtering and denoising

Goal: Apply image filters and find the approximate solution of linear system to process a

greyscale image.

Data: Download the file deer.jpg from webeep and move it to your working directory.

## 1 Overview

First, we will look at the applications of matrix multiplication to the filtering of images. The most common operations are blurring/smoothing, sharpening, and edge detection. All of these are examples of so-called convolution filters, that work by changing the color of the pixels in the image using the colors of nearby pixels.

Consider an image described by the matrix F = (fij) where (i; j) are integer coordinates of the pixel in an image, and the value fij returns the color of the image. Consider H = (hkl) to be a convolution matrix (filter matrix) of size n ×m. These matrices might have (and usually do have) different dimensions since H is usually relatively small. Then a new matrix G = F H (i.e. the convolution of F and H) can be found by the following formula:

where the summation runs over all the indices which make sense. In particular, if an element with a certain index does not exist, we simply assume the corresponding term to be zero. You can think of convolution being obtained by the filter matrix H sliding over the original image F. Two of the most common operations on images done with convolution filters include smoothing, sharpening, and edge detection. In the following we provide some examples:

- Two average smoothing filters are given by the matrices:

- An opposite action to blurring is the sharpening of an image with filters such as:

- Finally, it is possible to use convolution to detect edges in the image. Edge detection is used for image segmentation and data extraction in areas such as image processing, computer


*Figure 1: From left to right, original image, image with noise, sharpening filter, and edge detec- tion filters applied to the original image. These are just examples and do not correspond to the tasks solutions.*

vision, and machine vision. Two of the most common filters used for these are the Sobel filters:

The kernel Hed1 is a discrete analogue of a continuous Laplacian and may be interpreted as a sum of two discrete second order partial derivatives.

## 2 Tasks

The tasks of this challenge are also reported in https://forms.cloud.microsoft/e/EfrNXMAGYj.

- Load the image as an Eigen matrix with size m×n. Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white). Report the size of the matrix.

- Introduce a noise signal into the loaded image by adding random fluctuations of color ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.

- Reshape the original and noisy images as vectors v and w, respectively. Verify that each vector has mn components. Report here the Euclidean norm of v.

- Write the convolution operation corresponding to the smoothing kernel Hav1 as a matrix vector multiplication between a matrix A1 having size mn × mn and the image vector. Report the number of non-zero entries in A1.

- Apply the previous smoothing filter to the noisy image by performing the matrix vector multiplication A1w. Export and upload the resulting image.

- Write the convolution operation corresponding to the sharpening kernel Hsh1 as a matrix vector multiplication by a matrix A2 having size mn×mn. Report the number of non-zero entries in A2. Is A2 symmetric?

- Apply the previous sharpening filter to the original image by performing the matrix vector multiplication A2v. Export and upload the resulting image.


- Export the Eigen matrix A2 and vector w in the .mtx format. Using a suitable iterative solver and preconditioning technique available in the LIS library compute the approximate solution to the linear system A2x = w prescribing a tolerance of 10−12. Report here the iteration count and the final residual.

- Convert the previous approximate solution vector x into a .png image. Upload the resulting file here.

- Write the convolution operation corresponding to the detection kernel Hed2 as a matrix vector multiplication by a matrix A3 having size mn×mn. Is matrix A3 symmetric?

- Apply the previous edge detection filter to the original image by performing the matrix vector multiplication A3 v. Export and upload the resulting image.

- Using a suitable iterative solver available in the Eigen library compute the approximate so- lution of the linear system (4I+A3)y = w, where I denotes the identity matrix, prescribing a tolerance of 10−10. Report here the iteration count and the final residual.

- Import the previous approximate solution vector y in Eigen and convert it into a .png image and upload it.
