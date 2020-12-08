#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdio.h>

enum
{
  NELEMS = 1 << 22
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
  
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  for (int i = 0; i < NELEMS; ++i)
  {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  tmem = -wtime();
  if (cudaMalloc((void **)&d_A, size) != cudaSuccess)
  {
    fprintf(stderr, "Allocation error\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **)&d_B, size) != cudaSuccess)
  {
    fprintf(stderr, "Allocation error\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **)&d_C, size) != cudaSuccess)
  {
    fprintf(stderr, "Allocation error\n");
    exit(EXIT_FAILURE);
  }

  if (cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Host to device copying failed\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Host to device copying failed\n");
    exit(EXIT_FAILURE);
  }
  tmem += wtime();

  tgpu = -wtime();
  int threadsPerBlock = 1024;
  int blocksPerGrid = (NELEMS + threadsPerBlock - 1) / threadsPerBlock;
  vadd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, NELEMS);
  cudaDeviceSynchronize();
  tgpu += wtime();

  if (cudaGetLastError() != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch kernel!\n");
    exit(EXIT_FAILURE);
  }
  
  tmem -= wtime();
  if (cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    fprintf(stderr, "Device to host copying failed\n");
    exit(EXIT_FAILURE);
  }
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

  printf("GPU time (sec.): %.6f\n", tgpu);
  printf("Memory ops. (sec.): %.6f\n", tmem);
  printf("Total time (sec.): %.6f\n", tgpu + tmem);

  return 0;
}
