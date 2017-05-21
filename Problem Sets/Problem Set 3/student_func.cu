/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/
#include "device_launch_parameters.h"
#include "utils.h"

__global__ void
inclusive_min_max_scan(
    const float* const d_minReadBuffer,
    const float* const d_maxReadBuffer,
    float* d_minWriteBuffer,
    float* d_maxWriteBuffer,
    int neighborOffset,
    int numRows,
    int numCols)
{
    int ny = numRows;
    int nx = numCols;
    int2 imageIndex2d = make_int2(
        (blockIdx.x * blockDim.x) + threadIdx.x,
        (blockIdx.y * blockDim.y) + threadIdx.y);

    if (imageIndex2d.x < nx && imageIndex2d.y < ny)
    {
        int  imageIndex1d = (nx * imageIndex2d.y) + imageIndex2d.x;
        int neighborIndex1d = imageIndex1d - neighborOffset;
        if (neighborIndex1d < 0)
        {
            // copy data from read buffer to write buffer
            d_minWriteBuffer[imageIndex1d] = d_minReadBuffer[imageIndex1d];
            d_maxWriteBuffer[imageIndex1d] = d_maxReadBuffer[imageIndex1d];
        }
        else
        {
            float neighborValue = d_minReadBuffer[neighborIndex1d];
            float myValue = d_minReadBuffer[imageIndex1d];
            float minValue = neighborValue < myValue ? neighborValue : myValue;
            d_minWriteBuffer[imageIndex1d] = minValue;

            neighborValue = d_maxReadBuffer[neighborIndex1d];
            myValue = d_maxReadBuffer[imageIndex1d];
            float maxValue = neighborValue > myValue ? neighborValue : myValue;
            d_maxWriteBuffer[imageIndex1d] = maxValue;
        }
    }
}

__global__ void
compute_histogram(
    const float* const d_logLuminance,
    int numRows, int numCols,
    unsigned int* d_histogramBuffer,
    int numBins,
    float lumMin, float lumRange)
{
    int ny = numRows;
    int nx = numCols;
    int2 imageIndex2d = make_int2(
        (blockIdx.x * blockDim.x) + threadIdx.x,
        (blockIdx.y * blockDim.y) + threadIdx.y);

    if (imageIndex2d.x < nx && imageIndex2d.y < ny)
    {
        int  imageIndex1d = (nx * imageIndex2d.y) + imageIndex2d.x;
        float lumValue = d_logLuminance[imageIndex1d];
        size_t bin = static_cast<size_t>((lumValue - lumMin) / lumRange * numBins);
        atomicAdd(&d_histogramBuffer[bin], 1);
    }
}

__global__ void
exclusive_sum_scan(
    unsigned int* const d_buffer,
    int bufSize,
    int neighborOffset)
{
    int writeIndex1d = ((blockIdx.x * blockDim.x) + threadIdx.x) * neighborOffset;

    if (writeIndex1d >= bufSize)
    {
        int readIndex1 = writeIndex1d - neighborOffset;
        int readIndex2 = writeIndex1d;
        int read1 = d_buffer[readIndex1];
        int read2 = d_buffer[readIndex2];
        d_buffer[writeIndex1d] = read1 + read2;
    }
}

__global__ void
exclusive_downsweep_sum_scan(
    unsigned int* const d_buffer,
    int bufSize,
    int neighborOffset)
{
    int threadIndex = ((blockIdx.x * blockDim.x) + threadIdx.x) * neighborOffset;

    if (threadIndex >= bufSize)
    {
        int backIndex = threadIndex - neighborOffset;
        int frontIndex = threadIndex;
        int backValue = d_buffer[backIndex];
        int frontValue = d_buffer[frontIndex];
        d_buffer[backIndex] = frontValue;
        d_buffer[frontIndex] = backValue + frontValue;
    }
}

void your_histogram_and_prefixsum(
    const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float &min_logLum,
    float &max_logLum,
    const size_t numRows,
    const size_t numCols,
    const size_t numBins)
{
    //TODO
    /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
    store in min_logLum and max_logLum*/
    const dim3 blockSize(32, 16, 1);
    const dim3 gridSize(
        (numCols + blockSize.x - 1) / blockSize.x,
        (numRows + blockSize.y - 1) / blockSize.y);
    float* d_minScanBuffer1 = nullptr;
    cudaMalloc(&d_minScanBuffer1, (numRows * numCols) * sizeof(float));
    float* d_minScanBuffer2 = nullptr;
    cudaMalloc(&d_minScanBuffer2, (numRows * numCols) * sizeof(float));
    float* d_minReadBuffer = const_cast<float*>(d_logLuminance);
    float* d_minWriteBuffer = d_minScanBuffer1;
    float* d_maxScanBuffer1 = nullptr;
    cudaMalloc(&d_maxScanBuffer1, (numRows * numCols) * sizeof(float));
    float* d_maxScanBuffer2 = nullptr;
    cudaMalloc(&d_maxScanBuffer2, (numRows * numCols) * sizeof(float));
    float* d_maxReadBuffer = const_cast<float*>(d_logLuminance);
    float* d_maxWriteBuffer = d_maxScanBuffer1;
    int neighborOffset = 1;
    int numItems = (numRows * numCols);
    while (neighborOffset < numItems - 1)
    {
        inclusive_min_max_scan << <gridSize, blockSize >> >(
            d_minReadBuffer, d_maxReadBuffer,
            d_minWriteBuffer, d_maxWriteBuffer,
            neighborOffset,
            numCols, numRows);

        //cudaDeviceSynchronize();
        //checkCudaErrors(cudaGetLastError());

        if (neighborOffset == 1)
        {
            d_minReadBuffer = d_minScanBuffer2;
            d_maxReadBuffer = d_maxScanBuffer2;
        }

        float* curReadBuffer = d_minReadBuffer;
        d_minReadBuffer = d_minWriteBuffer;
        d_minWriteBuffer = curReadBuffer;

        curReadBuffer = d_maxReadBuffer;
        d_maxReadBuffer = d_maxWriteBuffer;
        d_maxWriteBuffer = curReadBuffer;

        neighborOffset <<= 1;
    }

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // read back the last item from the read buffer
    // (note that the current read buffer was the most recent write buffer)
    cudaMemcpy(&min_logLum, &d_minReadBuffer[numItems - 1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_logLum, &d_maxReadBuffer[numItems - 1], sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_minReadBuffer);
    cudaFree(d_minWriteBuffer);
    cudaFree(d_maxReadBuffer);
    cudaFree(d_maxWriteBuffer);

    //2) subtract them to find the range
    float lumRange = max_logLum - min_logLum;

    /*3) generate a histogram of all the values in the logLuminance channel using
    the formula: bin = (lum[i] - lumMin) / lumRange * numBins*/

    unsigned int* d_histogramBuffer = nullptr;
    cudaMalloc(&d_histogramBuffer, sizeof(unsigned int) * numBins);

    compute_histogram << <gridSize, blockSize >> >(
        d_logLuminance, numCols, numRows,
        d_histogramBuffer, numBins,
        min_logLum, lumRange);

    //cudaDeviceSynchronize();
    //checkCudaErrors(cudaGetLastError());

    cudaMemcpy(d_cdf, d_histogramBuffer,
        sizeof(unsigned int) * numBins, cudaMemcpyDeviceToDevice);

    /*4) Perform an exclusive scan (prefix sum) on the histogram to get
    the cumulative distribution of luminance values (this should go in the
    incoming d_cdf pointer which already has been allocated for you)       */
    const dim3 exclusiveBlockSize(32, 1, 1);
    dim3 exclusiveGridSize(numBins / exclusiveBlockSize.x, 1, 1);

    neighborOffset = 1;
    while (neighborOffset < numBins - 1)
    {
        exclusiveGridSize.x >>= 1;

        exclusive_sum_scan << <exclusiveBlockSize, exclusiveGridSize >> >(
            d_cdf, numBins, neighborOffset);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        neighborOffset <<= 1;
    }

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // reset right most element to identity/zero.
    cudaMemset(&d_cdf[numBins - 1], 0, sizeof(int));

    while (neighborOffset > 1)
    {
        exclusive_downsweep_sum_scan << <exclusiveBlockSize, exclusiveGridSize >> >(
            d_cdf, numBins, neighborOffset);

        //cudaDeviceSynchronize();
        //checkCudaErrors(cudaGetLastError());

        exclusiveGridSize.x <<= 1;
        neighborOffset >>= 1;
    }
}