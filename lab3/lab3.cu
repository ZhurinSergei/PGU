#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdio.h>

#define N 512
#define NELEMS (N * N)
#define TRANSP 1
#define BLOCK 32

#define CUDA_CHECK_RETURN(value)                                    \
  {                                                                 \
    cudaError_t _m_cudaStat = value;                                \
    if (_m_cudaStat != cudaSuccess)                                 \
    {                                                               \
      fprintf(stderr, "Error %s at line %d in file %s\n",           \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
      exit(1);                                                      \
    }                                                               \
  }

double wtime()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

__global__ void tr1(float *a, float *b, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if ((i < n) && (j < n))
    b[j * n + i] = a[i * n + j];
}

__global__ void tr2(float *a, float *b, int n)
{
  __shared__ float smem[BLOCK][BLOCK];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  smem[threadIdx.y][threadIdx.x] = a[i + j * n];
  __syncthreads();
  i = threadIdx.x + blockIdx.y * blockDim.x;
  j = threadIdx.y + blockIdx.x * blockDim.y;
  b[i + j * n] = smem[threadIdx.x][threadIdx.y];
}

__global__ void tr3(float *a, float *b, int n)
{
  __shared__ float smem[BLOCK][BLOCK + 1];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  smem[threadIdx.y][threadIdx.x] = a[i + j * n];
  __syncthreads();
  i = threadIdx.x + blockIdx.y * blockDim.x;
  j = threadIdx.y + blockIdx.x * blockDim.y;
  b[i + j * n] = smem[threadIdx.x][threadIdx.y];
}

int main()
{
  size_t size = sizeof(float) * NELEMS;
  double tgpu = 0, tmem = 0;
  float elapsedTime = 0;
  cudaEvent_t start, stop;

  /* Allocate vectors on host */
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  if (h_A == NULL || h_B == NULL)
  {
    fprintf(stderr, "Allocation error.\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < NELEMS; ++i)
  {
    h_A[i] = rand() / (float)RAND_MAX;
  }

  /* Allocate vectors on device */
  float *d_A = NULL, *d_B = NULL;
  tmem = -wtime();
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_B, size));

  /* Copy the host vectors to device */
  CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice))
  tmem += wtime();
  /* Launch the kernel */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  tgpu = -wtime();
  int threadsPerBlockDim = BLOCK;
  dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim, 1);
  int blocksPerGridDimX = ceilf(N / (float)threadsPerBlockDim);
  int blocksPerGridDimY = ceilf(N / (float)threadsPerBlockDim);
  dim3 gridDim(blocksPerGridDimX, blocksPerGridDimY, 1);
  cudaEventRecord(start, 0);
  // #if TRANSP == 1
  tr1<<<gridDim, blockDim>>>(d_A, d_B, N);
  // #elif TRANSP == 2
  tr2<<<gridDim, blockDim>>>(d_A, d_B, N);
  // #else
  tr3<<<gridDim, blockDim>>>(d_A, d_B, N);
  // #endif
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  tgpu += wtime();
  CUDA_CHECK_RETURN(cudaGetLastError());
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  /* Copy the device vectors to host */
  tmem -= wtime();
  CUDA_CHECK_RETURN(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));
  tmem += wtime();

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
      if (fabs(h_A[i * N + j] - h_B[j * N + i]) > 1e-5)
      {
        fprintf(stderr,
                "Result verification failed at element %d , %d! Ex: %f, Real: %f\n",
                i, j, h_A[i * N + j], h_B[j * N + i]);
        exit(EXIT_FAILURE);
      }
  }

  printf("Transponse\n");
  printf("GPU version (sec.): %.6lf\n", tgpu);
  printf("Memory ops. (sec.): %.6lf\n", tmem);
  printf("Total time (sec.): %.6lf\n", tgpu + tmem);
  printf("Events Time %.6f\n", elapsedTime);

  cudaFree(d_A);
  cudaFree(d_B);
  free(h_A);
  free(h_B);
  cudaDeviceReset();
  return 0;
}