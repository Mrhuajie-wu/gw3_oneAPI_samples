//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include<array>
#include<iostream>
#include<cmath>
#include <math.h> 
#include<iomanip>
#include<limits>
#include <chrono>
using namespace sycl;
using namespace std;
static const int N = 6;

double pow(double base_unit, double power_unit){
    double result=base_unit;
    int i=1;
    while(i<power_unit){
        result=result*base_unit;
        i++;
    }
    
  return result;  
}

double cal_mean(queue &q, double* x_value, double* x_prob )
{
    double *mean_x=malloc_shared<double>(N,q);
    q.parallel_for(range<1>(N), [=] (id<1> i){
    mean_x[i]=x_value[i]*x_prob[i];
    }).wait();
    double mean=0;
    for(int i=0;i<N;i++)
    {
        mean+=mean_x[i];
    }
    cout<<"mean value is: "<<mean<<std::endl;
    return mean;
    
}

double cal_var(queue &q,double* x_value,double* x_prob )
{
    double *pow_mean=malloc_shared<double>(N,q);
    q.parallel_for(range<1>(N), [=] (id<1> i){
    pow_mean[i]=x_value[i]*x_value[i]*x_prob[i];
    }).wait();
    double var=0;
    for(int i=0;i<N;i++)
    {
        var+=pow_mean[i];
    }
    double mean=cal_mean(q,x_value,x_prob);
    var=var-pow(mean,2);
    cout<<"var value is: "<<var<<std::endl;
    return var;

}

double cal_covariance(queue &q,double* x_value,double* y_value,double* prob)
{
    double *mean_xy=malloc_shared<double>(N,q);
    q.parallel_for(range<1>(N), [=] (id<1> i)
    {
    mean_xy[i]=x_value[i]*y_value[i]*prob[i];
    }).wait();
    double cov_xy=0;
    for(int i=0;i<N;i++)
    {
        cov_xy+=mean_xy[i];
    }
    double meanx=cal_mean(q,x_value,prob);
    double meany=cal_mean(q,y_value,prob);
    cov_xy=cov_xy-meanx*meany;
    return cov_xy;
}


int main()
{
   
    //# Initialization
    double x[]={43,21,25,42,57,59};
    double y[]={99,65,79,75,87,81};
    double x_prob[]={0.1,0.1,0.1,0.1,0.1,0.5};
    double y_prob[]={0.1,0.1,0.1,0.1,0.1,0.5};

    double x_mean=0;
    double sum_x=0;  //sum of x values  
    double sum_y=0;  //sum of y values  
    double sum_xy=0; //sum of xy values  
    double sum_x_squared=0; //sum of x  squared values  
    double sum_y_squared=0; //sum of y  squared values  
    
    //# Unified Shared Memory Allocation enables data access on host and device
    queue q;
    cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
    double *mean_x=malloc_shared<double>(N,q);
    double *mean_x_pow=malloc_shared<double>(N,q);
    double *mean_y=malloc_shared<double>(N,q);
    double *mean_y_pow=malloc_shared<double>(N,q);
    double *mean_xy=malloc_shared<double>(N,q);
    double *xy=malloc_shared<double>(N, q); //to hold xy calculated values  
    double *x_squared=malloc_shared<double>(N, q); // to hold x_squared calculated values  
    double *y_squared=malloc_shared<double>(N, q); // to hold y_squared calculated values

    //# Offload parallel computation to device
    
    //we do calculation the mean and var of x in parallel
    q.parallel_for(range<1>(N), [=] (id<1> i)
    {
        mean_x[i]=x[i]*x_prob[i];
        mean_x_pow[i]=x[i]*x[i]*x_prob[i];
    }).wait();
    
    double mean=0;
    double var=0;
    for(int i=0;i<N;i++)
    {
        mean+=mean_x[i];
        var+=mean_x_pow[i];
    }
    var=var-pow(mean,2);
    
    //we do calculation the mean and var of y in parallel
    q.parallel_for(range<1>(N), [=] (id<1> i)
    {
        mean_y[i]=y[i]*y_prob[i];
        mean_y_pow[i]=y[i]*y[i]*y_prob[i];
    }).wait();
    double meany=0;
    double vary=0;
    for(int i=0;i<N;i++)
    {
        meany+=mean_y[i];
        vary+=mean_y_pow[i];
    }
    vary=vary-pow(meany,2);
    
    //we do calculation the covariance of x and y in parallel
    q.parallel_for(range<1>(N), [=] (id<1> i)
    {
    mean_xy[i]=x[i]*y[i]*y_prob[i];
    }).wait();
    double cov_xy=0;
    for(int i=0;i<N;i++)
    {
        cov_xy+=mean_xy[i];
    }
    cov_xy=cov_xy-mean*meany;
    
    //# Print Output
    std::cout<<"x value is: "<<std::endl;
    for(int i=0; i<N; i++) std::cout << x[i] << "  ";
    std::cout<<std::endl<<"y value is: "<<std::endl;
    for(int i=0; i<N; i++) std::cout << y[i] << "  ";
    std::cout<<std::endl<<"probability value is: "<<std::endl;
    for(int i=0; i<N; i++) std::cout << x_prob[i] << "  ";
    std::cout<<std::endl<<"x_mean: "<<mean<<std::endl;
    std::cout<<"x_var: "<<var<<std::endl;
    std::cout<<"y_mean: "<<meany<<std::endl;
    std::cout<<"y_var: "<<vary<<std::endl;
    std::cout<<"cov_xy: "<<cov_xy<<std::endl;
       
    //double mean_value=cal_mean(q,x,x_prob);
    //double var_value=cal_var(q,x,x_prob);
    //std::cout<<"x_mean_value"<<mean_value<<std::endl;
    //std::cout<<"x_var_value"<<var_value<<std::endl;
    
       
    std::cout << "calculate the regression of x and y : y = ax + b" << std::endl;
   
    //we do calculation of  x*y x_squared y_squared in parallel
    q.parallel_for(range<1>(N), [=](id<1> i) {
       xy[i]=x[i]*y[i];
       x_squared[i]=pow(x[i],2);
       y_squared[i]=pow(y[i],2);
    }).wait();
    
    //calculate the  sum_x ,sum_y , sum_x_squared ,sum_y_squared  sum_xy
    for (int i =0; i < N; i++) {
        sum_x+=x[i];
        sum_y+=y[i];   
        sum_x_squared+=x_squared[i];
        sum_y_squared+=y_squared[i]; 
        sum_xy+=xy[i];   
    }
    
    //calculate the  Slope coefficient 
    double a=(N*(sum_xy)-(sum_x * sum_y))/(N*(sum_x_squared)-pow(sum_x,2));
    //calculate the  Intercept coefficient
    double b=((sum_y * sum_x_squared)-(sum_x * sum_xy)) / (N * (sum_x_squared)-pow(sum_x,2)); 
    
    std::cout<<"regression parameter a is: "<<a<<std::endl;
    std::cout<<"regression parameter b is: "<<b<<std::endl;
    
    
    //free(data, q);
    return 0;
}
