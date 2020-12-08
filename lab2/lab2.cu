#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdio.h>

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

enum
{
  NELEMS = 1 << 23
};

double wtime()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

__global__ void vadd(const float *a, const float *b, float *c, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
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
  float *h_C = (float *)malloc(size);
  if (h_A == NULL || h_B == NULL || h_C == NULL)
  {
    fprintf(stderr, "Allocation error.\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < NELEMS; ++i)
  {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  /* Allocate vectors on device */
  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  tmem = -wtime();
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_B, size));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_C, size));

  /* Copy the host vectors to device */
  CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice))
  CUDA_CHECK_RETURN(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice))
  tmem += wtime();

  /* Launch the kernel */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  tgpu = -wtime();
  int threadsPerBlock = 1024;
  int blocksPerGrid = (NELEMS + threadsPerBlock - 1) / threadsPerBlock;
  cudaEventRecord(start,0);
  vadd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, NELEMS);
  cudaEventRecord(stop,0); 
  cudaEventSynchronize(stop); 
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  tgpu += wtime();
  CUDA_CHECK_RETURN(cudaGetLastError());
  cudaEventElapsedTime(&elapsedTime,start,stop); 

  cudaEventDestroy(start); 
  cudaEventDestroy(stop); 
  /* Copy the device vectors to host */
  tmem -= wtime();
  CUDA_CHECK_RETURN(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
  tmem += wtime();

  for (int i = 0; i < NELEMS; ++i)
  {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaDeviceReset();

  printf("GPU version (sec.): %.6lf\n", tgpu);
  printf("Memory ops. (sec.): %.6lf\n", tmem);
  printf("Total time (sec.): %.6lf\n", tgpu + tmem);
  printf("Events Time %.6f\n", elapsedTime);

  return 0;
}
