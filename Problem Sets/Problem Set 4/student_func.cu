//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <cstdio>
#include <algorithm>

#define makePow2(v) \
	v--;			\
	v |= v >> 1;	\
	v |= v >> 2;	\
	v |= v >> 4;	\
	v |= v >> 8;	\
	v |= v >> 16;	\
	v++;			\


/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
 
static const unsigned int k_numBitsPerPass = 2;
static const unsigned int k_numBins = (1 << k_numBitsPerPass);

__global__
void compute_digit_histogram(const unsigned int* const d_inputBuf, unsigned int* const d_outputBuf, 
				const size_t itemsPerThread, int leastSigBitIndex, int bufSize)
{
	// itemsPerThread = 64
	// leastSigBitIndex = 0, 2, 4
	// bufSize = num items in d_inputBuf/d_outputBuf
	// k_numBins = 4
	int blockId = blockIdx.x
		+ (gridDim.x * blockIdx.y)
		+ (gridDim.x * gridDim.y * blockIdx.z);
	int threadId = (blockId * blockDim.x) + threadIdx.x;
	if(threadId >= bufSize)
		return;
	
	unsigned int localBins[k_numBins];
	memset(localBins, 0, sizeof(int)*k_numBins);
	// each thread processes 64 items from input
	// each pass 2 bits are evaluated from the input, starting with the least
	// significant and progressing to the most significant
	int baseIndex = threadId * itemsPerThread;
	unsigned int andBins = k_numBins - 1;
	// for each input item that this thread is assigned
	//  1. shift input by leastSigBitIndex parameter to get current set of 2 bits.
	//  2. compute bin index via "and" of shifted input and k_numBins-1
	//    add one to bin 3 if shifted number is 3: 3 & 3 = 3
	//    add one to bin 2 if shifted number is 2: 2 & 3 = 2
	//    add one to bin 1 if shifted number is 1: 1 & 3 = 1
	//    add one to bin 0 if shifted number is 0: 0 & 3 = 0
	for(int i = 0; i < itemsPerThread; ++i)
	{
		int offsetIndex = baseIndex + i;
		if(offsetIndex >= bufSize)
			break;
		unsigned int inputBits = d_inputBuf[offsetIndex] >> leastSigBitIndex;
		unsigned int binIndex = inputBits & andBins;
		localBins[binIndex]++;
	}
	// add the totals from the local bins to the global bins
	for(int i = 0; i < k_numBins; ++i)
		atomicAdd(d_outputBuf + i, localBins[i]);
}


__global__
void blockwise_exclusive_sum_scan(
	const unsigned int* const d_in,
	unsigned int* const d_out,
	unsigned int* const d_block_out,
	int bufSize)
{
	extern __shared__ unsigned int sharedBuf[];
	
	int blockId = blockIdx.x
		+ (gridDim.x * blockIdx.y)
		+ (gridDim.x * gridDim.y * blockIdx.z);
	int threadId = (blockId * blockDim.x) + threadIdx.x;
	int localThreadIndex = threadIdx.x;
	unsigned int tmp;
	
	if(threadId >= n || blockId > (bufSize / blockDim.x))
		return;

	// copy my input element to the shared memory
	sharedBuf[localThreadIndex] = d_in[threadId];
	__syncthreads();
	
	// each thread reads the first data value needed for the sum scan from the
	// shared memory, then synchronizes with other threads, so that second data
	// value can be read, added to the first data value, and then written back
	// to shared memory (without clobbering the first read).
	for(unsigned int myOffset = 1;
		myOffset < blockDim.x;
		myOffset <<= 1)
	{
		unsigned int first = sharedBuf[localThreadIndex];
		__syncthreads();

		if(localThreadIndex + myOffset < blockDim.x)
		{
            unsigned int second = sharedBuf[localThreadIndex + myOffset];
			sharedBuf[localThreadIndex + myOffset] = second + first;
		}
		__syncthreads();
	}

	// write result of first stage (i.e. scan) to output buffer
	d_out[threadId] = localThreadIndex > 0 ? sharedBuf[localThreadIndex - 1] : 0;

	// write last thread of block's output to d_block_out
	if(localThreadIndex == 1023)
		d_block_out[blockId] = sharedBuf[localThreadIndex];
}

__global__
void add_blockwise_output(
	const unsigned int* const d_in,
	unsigned int* const d_out, 
	const unsigned int* d_block_out,
	int numElems)
{
	int blockId = blockIdx.x
		+ (gridDim.x * blockIdx.y)
		+ (gridDim.x * gridDim.y * blockIdx.z);
	int threadId = (blockId * blockDim.x) + threadIdx.x;
	if(threadId >= numElems)
		return;

	d_out[threadId] = d_in[threadId] + d_block_out[blockId];
}

void exclusive_sum_scan(unsigned int* d_arr, unsigned int* d_out, int numElems)
{
	int numElems2 = numElems;
	makePow2(numElems2);
	int threads = 1024;
	dim3 blocks(
		numElems2/threads,
		numElems2/threads/65535+1,
		numElems2/threads/65535/65535+1);

	blocks.x = std::max(1u, std::min(blocks.x, 65535u));
	blocks.y = std::min(blocks.y, 65535u);
	blocks.z = std::min(blocks.z, 65535u);

	unsigned int* d_block_out;
	checkCudaErrors(cudaMalloc(
		&d_block_out,
		sizeof(unsigned int) * blocks.x * blocks.y * blocks.z));

	blockwise_exclusive_sum_scan<<<blocks, threads, threads * sizeof(unsigned int)>>>(
		d_arr, d_out, d_block_out, numElems);

	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());

	if(blocks.x > 1)
	{
		exclusive_sum_scan(d_block_out, d_block_out, blocks.x * blocks.y * blocks.z);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		// add output of blockwise exclusive sum scan to original output to get final
		// sum result
		add_blockwise_output<<<blocks, threads>>>(d_out, d_out, d_block_out, size);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	cudaFree(d_block_out);
}

__global__
void compute_is_bin_member(
	unsigned int* d_in, unsigned int* d_out,
	 const size_t size, int histoBinIndex, int leastSigBitIndex)
{
	int blockId = blockIdx.x +
		(gridDim.x * blockIdx.y) +
		(gridDim.x * gridDim.y * blockIdx.z);
	int threadId = (blockId * blockDim.x) + threadIdx.x;
	if(threadId >= size)
		return;
	
	// left shift the bits so that current set of least significan bits are first
	// bitwise-and it with k_numBins-1 (i.e. 3) to zero out other bits
	unsigned int input_bits = (d_in[threadId] >> leastSigBitIndex) & (k_numBins - 1);
	// then check if left over bits are equal to the histoBinIndex
	d_out[threadId] = input_bits == histoBinIndex ? 1 : 0;
}

__global__
void copy_input_to_sorted_output_pos(
	unsigned int* const d_inputVals, unsigned int* const d_inputPos,
	unsigned int* const d_outputVals, unsigned int* const d_outputPos,
	unsigned int* const d_binMemberBuf, unsigned int* d_binMemberBufSumScan,
	unsigned int* digitHistoSumScan, const size_t size, int binIndex)
{
	int blockId = blockIdx.x + 
		(gridDim.x * blockIdx.y) +
		(gridDim.x * gridDim.y * blockIdx.z);
	int threadId = (blockId * blockDim.x) + threadIdx.x;
	// if this input item is not a member of the current bin then skip it
	if(threadId >= size || !d_binMemberBuf[threadId])
		return;

	int outputPos = digitHistoSumScan[binIndex] + d_binMemberBufSumScan[threadId];

	d_outputVals[outputPos] = d_inputVals[threadId];
	d_outputPos[outputPos] = d_inputPos[threadId];
}

void your_sort(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               const size_t numElems)
{
	unsigned int numElems2 = numElems;
	makePow2(numElems2);

	const int histSerialization=64;
	int threads=1024;
	dim3 blocksHist(numElems2/threads/histSerialization,
					numElems2/threads/histSerialization/65535+1,
					numElems2/threads/histSerialization/65535/65535+1);
	blocksHist.x = std::max(1u, std::min(blocksHist.x, 65535u));
	blocksHist.y = std::min(blocksHist.y, 65535u);
	blocksHist.z = std::min(blocksHist.z, 65535u);

	dim3 blocksMap(numElems2/threads,
				   numElems2/threads/65535+1,
				   numElems2/threads/65535/65535+1);
	blocksMap.x = std::max(1u, std::min(blocksMap.x, 65535u));
	blocksMap.y = std::min(blocksMap.y, 65535u);
	blocksMap.z = std::min(blocksMap.z, 65535u);
	
	unsigned int* d_histoBuf;
	checkCudaErrors(
		cudaMalloc(&d_histoBuf, sizeof(unsigned int) * k_numBins));
	unsigned int* d_binMemberBuf;
	checkCudaErrors(
		cudaMalloc(&d_binMemberBuf, sizeof(unsigned int) * numElems));
	unsigned int* d_binMemberBufSumScan;
	checkCudaErrors(
		cudaMalloc(&d_binMemberBufSumScan, sizeof(unsigned int) * numElems));
	// iterate across the bits of the input values 2 bits at a time
	int numBits = 8 * static_cast<int>(sizeof(unsigned int));
	for(int leastSigBit = 0; leastSigBit < numBits; leastSigBit += k_numBitsPerPass)
	{
		checkCudaErrors(
			cudaMemset(d_histoBuf, 0, sizeof(unsigned int) * k_numBins));

		compute_digit_histogram<<<blocksHist, threads>>>(
			d_inputVals, d_histoBuf, histSerialization, leastSigBit, numElems);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		// compute an offset table from the d_histoBuf and store it
		// back in the d_histoBuf, this will be used as digit-specific
		// offset table
		exclusive_sum_scan(d_histoBuf, d_histoBuf, k_numBins);
		// for each bin compute the per-digit offset and copy the input to
		// the output
		for(int binIndex = 0; binIndex < k_numBins; ++binIndex)
		{
			// this kernel will output a 1 for each item in the input
			// if the current set of bits would go into the binIndex
			// we will use this (in the next step) to count up the number
			// of ocurrences of this digit in input
			compute_is_bin_member<<<blocksMap, threads>>>(
				d_inputVals, d_binMemberBuf, numElems, binIndex, leastSigBit);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			// do exclusive sum scan on d_binMemberBuf in order to get the
			// count of each digit, for example if input is:
			// [0, 1, 0, 1]
			// Then we want to know that the first zero is the first zero and
			// that the second zero is the second zero.
			// so the compute_is_bin_member function produces:
			// [1, 0, 1, 0]
			// and this exclusive_sum_scan produces:
			// [0, 1, 1, 2]
			exclusive_sum_scan(
				d_binMemberBuf, d_binMemberBufSumScan, numElems);
			// copy each member of this bin into the correct output position
			// using the previously computed digit-offset table stored in
			// d_histoBuf and the per-digit offset table stored in
			// d_binMemberBufSumScan
			// For example, if the input is:
			// [0, 2, 2, 3, 1, 0]
			// then d_histoBuf is initially:
			// [2, 1, 2, 1]
			// and the sum scan of d_histoBuf which we are sticking back into
			// d_histBuf is:
			// [0, 2, 3, 5]
			// Then for bin 0 the per-digit offset table is:
			// [0, 1, 1, 1, 1, 1]
			// (which comes from the exclusive-sum-scan of d_binMemberBuf)
			// so the final position of the first zero is:
			// d_histoBuf[0] + d_binMemberBufSumScan[0]
			// and the final position of the second zero is:
			// d_histoBuf[0] + d_binMemberBufSumScan[5]
			copy_input_to_sorted_output_pos<<<blocksMap, threads>>>(
				d_inputVals, d_inputPos,
				d_outputVals, d_outputPos,
				d_binMemberBuf, d_binMemberBufSumScan,
				d_histoBuf, numElems, binIndex);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
		}
		// now move to the next set of bits and sort the output of the last step
		std::swap(d_inputPos, d_outputPos);
		std::swap(d_inputVals, d_outputVals);
	}
	cudaMemcpy(d_outputVals, d_inputVals,
		numElems*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos, d_inputPos,
		numElems*sizeof(int), cudaMemcpyDeviceToDevice);

	cudaFree(d_binMemberBufSumScan);
	cudaFree(d_binMemberBuf);
	cudaFree(d_histoBuf);
}