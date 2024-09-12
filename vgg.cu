/* Includes, system */
#include <cstdio>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cstdio>
#include <stdio.h>
/* Includes, cuda */
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16

#define BK 16
#define BM 128
#define BN 128

#define TM 4
#define TN 4


__global__ void matrixMultiplication(const float* A, const float* B, float* C, int M, int N, int K) {

	const int cRow = blockIdx.y;
	const int cCol = blockIdx.x;

	const int totalResultsBlocktile = BM * BN;

	const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

	const int threadCol = threadIdx.x % (BN / TN);
	const int threadRow = threadIdx.x / (BN / TN);

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	const int innerRowA = threadIdx.x / BK;
	const int innerColA = threadIdx.x % BK;

	const int strideA = numThreadsBlocktile / BK;
	const int innerRowB = threadIdx.x / BN;
	const int innerColB = threadIdx.x % BN;

	const int strideB = numThreadsBlocktile / BN;

	float threadResults[TM * TN] = { 0.0 };

	float regM[TM] = { 0.0 };
	float regN[TN] = { 0.0 };

	for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
		for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
			As[(innerRowA + loadOffset) * BK + innerColA] =
				A[(innerRowA + loadOffset) * K + innerColA];
		}
		for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
			Bs[(innerRowB + loadOffset) * BN + innerColB] =
				B[(innerRowB + loadOffset) * N + innerColB];
		}
		__syncthreads();


		A += BK;
		B += BK * N;

		// calculate per-thread results
		for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
			// block into registers
			for (int i = 0; i < TM; ++i) {
				regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
			}
			for (int i = 0; i < TN; ++i) {
				regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
			}
			for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
				for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[resIdxM * TN + resIdxN] +=
						regM[resIdxM] * regN[resIdxN];
				}
			}
		}
		__syncthreads();
	}

	for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
		for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
			C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
				threadResults[resIdxM * TN + resIdxN] + C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
		}
	}

}

__global__ void maxPool(const float* input, float* output, int w, int h, int c, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n * c * h / 2 * w / 2)
	{
		int y = idx % (h / 2);
		int x = (idx / (h / 2)) % (w / 2);
		int k = (idx / (h / 2 * w / 2)) % c;
		int i = idx / (c * h / 2 * w / 2);

		output[idx] = fmaxf(fmaxf(input[i * c * h * w + k * h * w + (2 * y) * w + (2 * x)],
			input[i * c * h * w + k * h * w + (2 * y) * w + (2 * x + 1)]),
			fmaxf(input[i * c * h * w + k * h * w + (2 * y + 1) * w + (2 * x)],
				input[i * c * h * w + k * h * w + (2 * y + 1) * w + (2 * x + 1)]));
	}
}

__global__ void convolution(const float* input, float* output, const float* filter,
	int w, int h, int c, int n, int filter_w, int filter_h,
	int dilation_w, int dilation_h, int pad_w, int pad_h,
	int wstride, int hstride) {
	// Calculate the row and column index of the element
	int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	int out_y = blockIdx.y * blockDim.y + threadIdx.y;
	int out_n = blockIdx.z * blockDim.z + threadIdx.z;

	// Calculate output dimensions
	int out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) / wstride + 1;
	int out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) / hstride + 1;

	if (out_x < out_w && out_y < out_h && out_n < n) {
		float value = 0.0f;
		for (int kw = 0; kw < filter_w; ++kw) {
			for (int kh = 0; kh < filter_h; ++kh) {
				for (int cc = 0; cc < c; ++cc) {
					int in_x = out_x * wstride - pad_w + kw * dilation_w;
					int in_y = out_y * hstride - pad_h + kh * dilation_h;
					if (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h) {
						int filter_idx = ((out_n * c + cc) * filter_h + kh) * filter_w + kw;
						int input_idx = (cc * h + in_y) * w + in_x;
						value += input[input_idx] * filter[filter_idx];
					}
				}
			}
		}
		int output_idx = (out_n * out_h + out_y) * out_w + out_x;
		output[output_idx] = value;
	}
}

float* initializeRandomFilter(int filterWidth, int filterHeight, int c_in, int c_out) {
	float* filter = (float*)malloc(filterWidth * filterHeight * c_in * c_out * sizeof(float));

	srand(time(0)); // Seed the random number generator

	for (int i = 0; i < filterWidth * filterHeight * c_in * c_out; ++i) {
		filter[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random value between 0 and 1
	}

	return filter;
}

void Conv2D(float* input, float* output, int w, int h, int c, int n, int k, int filter_w, int filter_h, int dilation_w, int dilation_h, int pad_w, int pad_h, int wstride, int hstride) {
	// Calculate the dimensions of the output
	int out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) / wstride + 1;
	int out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) / hstride + 1;

	float* filter = initializeRandomFilter(filter_w, filter_h, c, k);

	float* d_input, * d_filter, * d_output;
	cudaMalloc(&d_input, w * h * c * n * sizeof(float));
	cudaMalloc(&d_filter, filter_w * filter_h * c * k * sizeof(float));
	cudaMalloc(&d_output, out_w * out_h * k * sizeof(float));

	cudaMemcpy(d_input, input, w * h * c * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter, filter_w * filter_h * c * k * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridDim((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE, n);


	for (size_t i = 0; i < n; i++)
		convolution << <gridDim, blockDim >> > (d_input + i * (w * h * c), d_output, d_filter, w, h, c, k, filter_w, filter_h,
			dilation_w, dilation_h, pad_w, pad_h, wstride, hstride);

	/// error handling
	cudaError_t err = cudaGetLastError(); ;
	if (err != cudaSuccess)
	{
		printf("::::%d,%d,%d::::%d,%d,%d::::errpr: %s\n", gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z, cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();

	cudaMemcpy(output, d_output, out_w * out_h * k * n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_filter);
	cudaFree(d_output);
}

void MaxPool(float* input, float* output, int w, int h, int c, int n)
{
	float* d_input, * d_output;

	cudaMalloc((void**)&d_input, n * c * h * w * sizeof(float));
	cudaMalloc((void**)&d_output, n * c * h / 2 * w / 2 * sizeof(float));

	cudaMemcpy(d_input, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);

	// Launch kernel
	int blockSize = 256;
	int numBlocks = (n * c * h / 2 * w / 2 + blockSize - 1) / blockSize;

	auto start = std::chrono::steady_clock::now();
	maxPool << <numBlocks, blockSize >> > (d_input, d_output, w, h, c, n);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
	std::cout << " " << fwd_time << " ms" << std::endl;

	/// error checking
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("::::%d::::%d::::errpr: %s\n", blockSize,
			blockSize, cudaGetErrorString(err));
	}

	cudaMemcpy(output, d_output, n * c * h / 2 * w / 2 * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void FC(float* input, float* output, int left, int right) {
	int m = 1, k = left, n = right;

	float* h_B = (float*)malloc(left * right * sizeof(float));
	for (int i = 0; i < left * right; i++) {
		h_B[i] = (float)std::rand() / RAND_MAX / 1000;
	}

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, k * sizeof(float));
	cudaMalloc(&d_B, left * right * sizeof(float));
	cudaMalloc(&d_C, right * sizeof(float));

	cudaMemcpy(d_A, input, k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, left * right * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

	auto start = std::chrono::steady_clock::now();

	matrixMultiplication << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, m, n, k);

	/// error handling
	cudaError_t err = cudaGetLastError(); ;
	if (err != cudaSuccess)
	{
		printf("::::%d::::%d::::errpr: %s\n", numBlocks,
			threadsPerBlock, cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());

	std::cout << " " << fwd_time << " ms" << std::endl;

	cudaMemcpy(output, d_C, right * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_B);
}

int main()
{
	std::srand(std::time(0));

	float* input;
	float* output;

	int data_size = 224 * 224 * 3 * 1;
	input = (float*)malloc(data_size * sizeof(float));
	for (int i = 0; i < data_size; i++)
	{
		input[i] = (float)std::rand() / RAND_MAX;
	}

	// ===============  1 =====================
	std::cout << "CONV 224x224x64";
	output = (float*)malloc(224 * 224 * 64 * 1 * sizeof(float));
	Conv2D(input, output, 224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "CONV 224x224x64";
	output = (float*)malloc(224 * 224 * 64 * 1 * sizeof(float));
	Conv2D(input, output, 224, 224, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "POOLMAX 112x112x64";
	output = (float*)malloc(112 * 112 * 64 * sizeof(float));
	MaxPool(input, output, 224, 224, 64, 1);
	std::swap(input, output);
	free(output);

	// ===============  2 =====================
	std::cout << "CONV 112x112x128";
	output = (float*)malloc(112 * 112 * 128 * 1 * sizeof(float));
	Conv2D(input, output, 112, 112, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "CONV 112x112x128";
	output = (float*)malloc(112 * 112 * 128 * 1 * sizeof(float));
	Conv2D(input, output, 112, 112, 128, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "POOLMAX 56x56x128";
	output = (float*)malloc(56 * 56 * 128 * sizeof(float));
	MaxPool(input, output, 112, 112, 128, 1);
	std::swap(input, output);
	free(output);

	// ===============  3 =====================
	std::cout << "CONV 56x56x256";
	output = (float*)malloc(56 * 56 * 256 * 1 * sizeof(float));
	Conv2D(input, output, 56, 56, 128, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "CONV 56x56x256";
	output = (float*)malloc(56 * 56 * 256 * 1 * sizeof(float));
	Conv2D(input, output, 56, 56, 256, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "POOLMAX 28x28x256";
	output = (float*)malloc(28 * 28 * 256 * sizeof(float));
	MaxPool(input, output, 56, 56, 256, 1);
	std::swap(input, output);
	free(output);

	// ===============  4 =====================
	std::cout << "CONV 28x28x512";
	output = (float*)malloc(28 * 28 * 512 * 1 * sizeof(float));
	Conv2D(input, output, 28, 28, 256, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "CONV 28x28x512";
	output = (float*)malloc(28 * 28 * 512 * 1 * sizeof(float));
	Conv2D(input, output, 28, 28, 512, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "POOLMAX 14x14x512";
	output = (float*)malloc(14 * 14 * 512 * sizeof(float));
	MaxPool(input, output, 28, 28, 512, 1);
	std::swap(input, output);
	free(output);

	// ===============  5 =====================
	std::cout << "CONV 14x14x1024";
	output = (float*)malloc(14 * 14 * 1024 * 1 * sizeof(float));
	Conv2D(input, output, 14, 14, 512, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "CONV 14x14x1024";
	output = (float*)malloc(14 * 14 * 1024 * 1 * sizeof(float));
	Conv2D(input, output, 14, 14, 1024, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
	std::swap(input, output);
	free(output);

	std::cout << "POOLMAX 7x7x1024";
	output = (float*)malloc(7 * 7 * 1024 * sizeof(float));
	MaxPool(input, output, 14, 14, 1024, 1);
	std::swap(input, output);
	free(output);

	// ===============  6 =====================
	std::cout << "FC 4096";
	output = (float*)malloc(4096 * sizeof(float));
	FC(input, output, 7 * 7 * 1024, 4096);
	std::swap(input, output);
	free(output);

	std::cout << "FC 4096";
	output = (float*)malloc(4096 * sizeof(float));
	FC(input, output, 4096, 4096);
	std::swap(input, output);
	free(output);

	std::cout << "FC 1000";
	output = (float*)malloc(1000 * sizeof(float));
	FC(input, output, 4096, 1000);

	free(input);
	free(output);

	return 0;
}
