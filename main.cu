#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>

#define L 512
#define THREADS_PER_BLOCK 16
#define BLOCKS_PER_GRID (L / THREADS_PER_BLOCK)
#define MC_STEPS 1000

__global__ void setupKernel(curandState *state, unsigned long seed)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * L;

    if (x < L && y < L)
    {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void metropolis(int *spin, curandState *state, float T, int parity)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= L || y >= L)
        return;

    if ((x + y) % 2 != parity)
        return;

    int idx = x + y * L;

    int left = ((x - 1 + L) % L) + y * L;
    int right = ((x + 1) % L) + y * L;
    int up = x + ((y - 1 + L) % L) * L;
    int down = x + ((y + 1) % L) * L;

    int sumNeighbors = spin[left] + spin[right] + spin[up] + spin[down];
    int deltaE = 2 * spin[idx] * sumNeighbors;

    curandState localState = state[idx];
    float randVal = curand_uniform(&localState);
    state[idx] = localState;

    if (deltaE <= 0 || randVal < expf(-deltaE / T))
    {
        spin[idx] = -spin[idx];
    }
}

int main()
{
    int *spinHost = (int *) malloc(L * L * sizeof(int));
    int *spinDev;

    curandState *devStates;

    srand(time(NULL));
    for (int i = 0; i < L * L; ++i)
    {
        spinHost[i] = (rand() % 2) * 2 - 1;
    }

    cudaMalloc((void **) &spinDev, L * L * sizeof(int));
    cudaMalloc((void **) &devStates, L * L * sizeof(curandState));

    cudaMemcpy(spinDev, spinHost, L * L * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS_PER_GRID, BLOCKS_PER_GRID);
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    setupKernel<<<blocks, threads>>>(devStates, time(NULL));

    float T = 2.269;

    for (int step = 0; step < MC_STEPS; ++step)
    {
        metropolis<<<blocks, threads>>>(spinDev, devStates, T, 0);
        cudaDeviceSynchronize();

        metropolis<<<blocks, threads>>>(spinDev, devStates, T, 1);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(spinHost, spinDev, L * L * sizeof(int), cudaMemcpyDeviceToHost);

    std::ofstream outFile("spins.txt");
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            outFile << spinHost[i * L + j] << " ";
        }
        outFile << std::endl;
    }
    outFile.close();

    int sum = 0;
    for (int i = 0; i < L * L; i++)
    {
        sum += spinHost[i];
    }

    float m_avg = (float) sum / (L * L);
    std::cout << "Average magnetization: " << m_avg << std::endl;

    cudaFree(spinDev);
    cudaFree(devStates);
    free(spinHost);

    return 0;
}
