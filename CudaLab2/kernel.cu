
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "wb.h"

//@@ Сложение векторов
__global__ void vecAdd(float* in1, float* in2, float* out, int len) 
{  
    int idx = threadIdx.x;             //Получаем id текущей нити.
    out[idx] = in1[idx] + in2[idx];    //Расчитываем результат.
}

int main(int argc, char** argv) {
    wbArg_t args;
    int inputLength;
    float* hostInput1;
    float* hostInput2;
    float* hostOutput;
    float* deviceInput1;
    float* deviceInput2;
    float* deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    //@@ Выделение памяти GPU
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc(&deviceInput1, inputLength * sizeof(float));
    cudaMalloc(&deviceInput2, inputLength * sizeof(float));
    cudaMalloc(&deviceOutput, inputLength * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    //@@ Копирование памяти на GPU
    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Инициализируйте размерности сетки и блоков
    int blockSize = 128;
    int gridSize = ceil(static_cast<double>(inputLength) / blockSize);

    //@@ Запустите ядро GPU
    wbTime_start(Compute, "Performing CUDA computation");
    vecAdd <<<gridSize, blockSize>>> (deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    //@@ Скопируйте память GPU обратно на хост
    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    //@@ Освободите память GPU
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
