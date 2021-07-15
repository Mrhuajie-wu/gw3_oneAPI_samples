// Convolution, DPCPP implemention
// by Yuxiao Hu 
// last change: 2021/7/2
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

const int seed = 100;

void Initialize(float* map, float* parallel_result, float* scala_result,float* kernel, const size_t n_rows, const size_t n_cols, const size_t slide_rows, const size_t slide_cols){
    cout << "Intializing ..." << "\n";
    srand(seed);
    for (size_t i = 0; i < n_rows; i++){
        size_t offset = i * n_cols;
        for(size_t j = 0; j < n_cols; j++){
            map[offset + j] = rand() % 256;
            parallel_result[offset + j] = 0;
            scala_result[offset + j] = 0;
        }
    }
    for (size_t i = 0; i < slide_rows; i++){
        size_t offset = i * slide_cols;
        for(size_t j = 0; j < slide_cols; j++){
            kernel[offset + j] = rand() % 2;
        }
    }
}

void PrintTargetInfo(queue &q){
    auto device = q.get_device();
    auto max_block_size = device.get_info<info::device::max_work_group_size>();

    auto max_EU_count = device.get_info<info::device::max_compute_units>();

    cout<< " Running on " << device.get_info<info::device::name>()<<"\n";
    cout<< " The Device Max Work Group Size is : "<< max_block_size<<"\n";
    cout<< " The Device Max EUCount is : " << max_EU_count<<"\n"; 
}

void PrintResult(float * map, size_t n_rows, size_t n_cols){
    for (int i = 0; i < n_rows; i++){
        for(int j = 0; j < n_cols; j++){
            cout << map[i * n_cols + j] << ", ";
        }
        cout << "\n";
    }
}

void ParallelConvolution(queue & q, float * map, float * parallel_result, float* kernel, const size_t n_rows, const size_t n_cols, const size_t slide_size_row, const size_t slide_size_col){
    size_t n_size = n_cols * n_rows;
    size_t kernel_size = slide_size_row * slide_size_col;
    buffer<float> map_buf(map, range(n_size));
    buffer<float> parallel_result_buf(parallel_result, range(n_size));
    buffer<float> kernel_buf(kernel, range(kernel_size));
    q.submit([&](auto &h){
        accessor map_a(map_buf, h, read_only);
        accessor parallel_result_a(parallel_result_buf, h, write_only);
        accessor kernel_a(kernel_buf, h, read_only);

        auto global_range = range<2>(n_rows, n_cols);

        h.parallel_for(global_range, [=](auto it){
            size_t gid_row = it[0];
            size_t gid_col = it[1];
            float sum = 0.0;
            for(size_t i = 0; (i < slide_size_row) && (i + gid_row < n_rows); i++){
                size_t offset = (i + gid_row) * n_cols + gid_col;
                for(size_t j = 0; (j < slide_size_col) && (j + gid_col < n_cols); j++){
                    sum += map_a[offset + j] * kernel_a[i * slide_size_col + j];
                }
            }
            parallel_result_a[gid_row * n_cols + gid_col] = sum / (slide_size_row * slide_size_col);
        });
    });
}

void ScalaConvolution(float * map, float * scala_result, float* kernel, const size_t n_rows, const size_t n_cols, const size_t slide_size_row, const size_t slide_size_col){
    for (size_t i = 0; i < n_rows; i++){
        for (size_t j = 0; j < n_cols; j++){
            size_t offset = i * n_cols + j;
            float sum = 0.0;
            for(size_t wi = 0; (wi < slide_size_row) && (wi + i < n_rows); wi++){
                for(size_t wj = 0; (wj < slide_size_col) && (wj + j < n_cols); wj++){
                    sum += map[(wi + i) * n_cols + wj + j] * kernel[wi * slide_size_col + wj];
                }
            }
            scala_result[offset] = sum / (slide_size_row * slide_size_col);
        }
    }
}

int main(int argc, char* argv[]) {
    float * map;
    float * parallel_result;
    float * scala_result;
    float * kernel;
    size_t n_rows, n_cols;
    size_t slide_size_row, slide_size_col;

    try{
        n_rows = stoi(argv[1]);
        n_cols = stoi(argv[2]);
        slide_size_row = stoi(argv[3]);
        slide_size_col = stoi(argv[4]);
    }
    catch(...){
        cout << "incorrect parameters of " << "gw3" << "\n";
        cout << "Usage: " << "\n";
        cout << "gw3" << " row_size col_size slide_size_row slide_size_col" << "\n";
        cout << "row_size: size of dim0" << "\n";
        cout << "col_size: size of dim1" << "\n";
        cout << "slide_size_row, col: size of slider" << "\n";
        return 1;
    }

    size_t n_size = n_rows * n_cols;
    size_t kernel_size = slide_size_row * slide_size_col;

    map = new float[n_size];
    parallel_result = new float[n_size];
    scala_result = new float[n_size];
    kernel = new float[kernel_size];

    Initialize(map, parallel_result, scala_result, kernel, n_rows, n_cols, slide_size_row, slide_size_col);
    cout << "Grid Sizes: " << n_rows << " " << n_cols << "\n";
    cout << "Slider Sizes: " << slide_size_row << " " << slide_size_col << "\n";

    default_selector device_selector;
    queue q(device_selector, dpc_common::exception_handler);

    cout << "Computing convolution on device ..." << "\n";
    // display device info 
    PrintTargetInfo(q);

    // Start Timer
    dpc_common::TimeInterval t_offload;

    ParallelConvolution(q, map, parallel_result, kernel, n_rows, n_cols, slide_size_row, slide_size_col);
    q.wait_and_throw();
    auto time = t_offload.Elapsed();
    cout << "offload time: " << time << "\n";
    cout << "offload result: " << "\n";
    PrintResult(parallel_result, n_rows, n_cols);

    // Compute on CPU
    dpc_common::TimeInterval t_cpu;

    ScalaConvolution(map, scala_result, kernel, n_rows, n_cols, slide_size_row, slide_size_col);
    time = t_cpu.Elapsed();
    cout << "CPU time: " << time << "\n";
    cout << "CPU result" << "\n";
    PrintResult(scala_result, n_rows, n_cols);

    // cleanup
    delete[] map;
    delete[] parallel_result;
    delete[] scala_result;
    return 0;
}