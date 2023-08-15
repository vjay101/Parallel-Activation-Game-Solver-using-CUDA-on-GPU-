/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/


//This kernel is used to find level 0 nodes

__global__ void initializeLevel0(int currLevel, int *d_offset, int *d_csrList, int *d_vertexLevel, int level0lastnode){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
   
    if(id <= level0lastnode){
        d_vertexLevel[id]=0;
        for(int j = d_offset[id]; j<d_offset[id+1]; j++){
            atomicCAS(&d_vertexLevel[d_csrList[j]], -1, currLevel+1);
           
        }
    }
}

//this kernel will find the nodes of corresponding level from 1 to L-1

__global__ void find_level(int currLevel,int *d_offset, int *d_csrList, int *d_vertexLevel){
    int id=blockIdx.x*blockDim.x+threadIdx.x; 
    
    if(d_vertexLevel[id]==currLevel){
        for(int j = d_offset[id]; j<d_offset[id+1]; j++){
           atomicCAS(&d_vertexLevel[d_csrList[j]], -1, currLevel+1);
        }
    }
  

}

//this kernel will activate level no 0 and at same time update aid of level 1
__global__ void zerolevelactive(int *d_offset,int *d_csrList,int *d_apr,int *d_aid,int *d_activeVertex,int *d_isactiveNode,int *d_levelfind,int L,int V,int E){
 
 int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id <= d_levelfind[0]){
   
        d_isactiveNode[id]=1;
        
        for(int j=d_offset[id];j<d_offset[id+1];j++){
           atomicAdd(&d_aid[d_csrList[j]], 1);
        }
    }
}

//this kernel will activate from level no 1 to last level
  __global__ void activation(int count,int *d_offset,int *d_csrList,int *d_apr,int *d_aid,int *d_activeVertex,int *d_isactiveNode,int *d_levelfind,int L,int V,int E){
    
       int id=blockIdx.x*blockDim.x+threadIdx.x;
       id= id + d_levelfind[count]+1;
       if(id <= d_levelfind[count+1]){
        if(d_aid[id]>=d_apr[id]){
          d_isactiveNode[id]=1;
         
          for(int j=d_offset[id];j<d_offset[id+1];j++){
              atomicAdd(&d_aid[d_csrList[j]], 1);

           }
        }
       }
}

//this kernel will deactivate if a node doesnot satisfy the condition to remain active
__global__ void de_activation(int count,int *d_offset,int *d_csrList,int *d_apr,int *d_aid,int *d_activeVertex,int *d_isactiveNode,int *d_levelfind,int L,int V,int E){
    
         int id=blockIdx.x*blockDim.x+threadIdx.x;
          id= id + d_levelfind[count]+1;  
          if(id <= d_levelfind[count+1]){ 
           if(id>d_levelfind[count]+1 && id<d_levelfind[count+1] && d_isactiveNode[id]==1 && d_isactiveNode[id-1]==0 && d_isactiveNode[id+1]==0){
           d_isactiveNode[id]=0;
          
           for(int j=d_offset[id];j<d_offset[id+1];j++){
              atomicAdd(&d_aid[d_csrList[j]],-1);
           }
        }
    }

    
}


//this kernel will find the final ans in which all the nodes V will run parallel
__global__ void active_vertex(int *d_vertexLevel,int *d_offset,int *d_csrList,int *d_apr,int *d_aid,int *d_activeVertex,int *d_isactiveNode,int *d_levelfind,int L,int V,int E){
    
  
    int id=blockIdx.x*blockDim.x+threadIdx.x;
     if(id<V){
         if(d_isactiveNode[id]==1)
         {
             atomicAdd(&d_activeVertex[d_vertexLevel[id]], 1);
         }
    }
    
  
}
   
     
    
    
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement
    

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // active in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    
/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/
// variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
    cudaMalloc(&d_activeVertex, L*sizeof(int));
    //cudaMemcpy(d_activeVertex, h_activeVertex, L*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_activeVertex, 0, L*sizeof(int));
   

    int *h_aid;
    h_aid = (int *)malloc(V*sizeof(int));
    // setting initially all to zero
    memset(h_aid, 0, V*sizeof(int));

   
    cudaMalloc(&d_aid, V*sizeof(int));
    //cudaMemcpy(d_aid, h_aid, V*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_aid, 0, V*sizeof(int));
   
    int *h_isactiveNode;
    h_isactiveNode = (int*)malloc(V*sizeof(int));
    // setting initially all to zero
    memset(h_isactiveNode, 0, V*sizeof(int)); 

    int *d_isactiveNode;
    cudaMalloc(&d_isactiveNode, V*sizeof(int));
    
    cudaMemset(d_isactiveNode, 0, V*sizeof(int));

    int *h_levelfind;
    h_levelfind = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_levelfind, 0, L*sizeof(int)); 

    int *d_levelfind;
    cudaMalloc(&d_levelfind, L*sizeof(int));
    
    cudaMemset(d_levelfind,0,L*sizeof(int));

    int *h_vertexLevel;
    h_vertexLevel = (int*)malloc(V*sizeof(int));
    int *d_vertexLevel;
    cudaMalloc(&d_vertexLevel, V*sizeof(int));
    cudaMemset(d_vertexLevel,-1,V*sizeof(int));



//launching of kernel threads here
//calculating the level no of each node
    
    // Level 0 ka last nikal le 

    // numThread = Level 0 pe nodes hai 
    int lasLevel0Node; 
    for(int i=0;h_apr[i]==0;i++){
      lasLevel0Node=i;
    }
     
    //kernel ko launch krna 
    int numBlock = (V+1024-1)/1024;
    int numThread = 1024;
    initializeLevel0<<<numBlock, numThread>>>(0, d_offset, d_csrList, d_vertexLevel, lasLevel0Node);
    cudaMemcpy(h_vertexLevel,d_vertexLevel,V*sizeof(int),cudaMemcpyDeviceToHost);

    //for finding the level of nodes
    for(int i=1;i<L-1;i++){
        find_level<<<numBlock, numThread>>>(i,d_offset, d_csrList, d_vertexLevel);
     }
    cudaMemcpy(h_vertexLevel,d_vertexLevel,V*sizeof(int),cudaMemcpyDeviceToHost);
   //calculating end edges from above level node array
   int l=0;
   for(int i=0;i<V-1;i++){
      if(h_vertexLevel[i]!=h_vertexLevel[i+1])
        {
            h_levelfind[l]=i;
            l++;
        }
   }
    h_levelfind[L-1]=V-1; 
    cudaMemcpy(d_levelfind,h_levelfind,L*sizeof(int),cudaMemcpyHostToDevice);
  
    
    //kernel:checking activeness of nodes of level 0 and at same time update aid of level 1

     zerolevelactive<<<(h_levelfind[0]+1024)/1024,1024>>>(d_offset,d_csrList,d_apr,d_aid,d_activeVertex,d_isactiveNode,d_levelfind,L,V,E);
     cudaMemcpy(h_aid,d_aid,V*sizeof(int),cudaMemcpyDeviceToHost);
     cudaMemcpy(h_isactiveNode,d_isactiveNode,V*sizeof(int),cudaMemcpyDeviceToHost);


    //kernel:checking activeness of L1 to last level
    
    for(int i=1;i<L;i++){
        int num_block = (h_levelfind[i]-h_levelfind[i-1]+1024)/1024;
        activation<<<num_block, 1024>>>(i-1,d_offset,d_csrList,d_apr,d_aid,d_activeVertex,d_isactiveNode,d_levelfind,L,V,E);
        cudaDeviceSynchronize();
        de_activation<<<num_block, 1024>>>(i-1,d_offset,d_csrList,d_apr,d_aid,d_activeVertex,d_isactiveNode,d_levelfind,L,V,E);
        cudaDeviceSynchronize();
       
    }
     cudaMemcpy(h_aid,d_aid,V*sizeof(int),cudaMemcpyDeviceToHost);
   
    

    //kernel:to calculate the final result

    active_vertex<<<(V+1024)/1024,1024>>>(d_vertexLevel,d_offset,d_csrList,d_apr,d_aid,d_activeVertex,d_isactiveNode,d_levelfind,L,V,E);
    cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);
 //finally the result is in h_activeVertex
   

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}
}