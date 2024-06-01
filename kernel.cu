#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>

#define VERTICES 300
#define DENSITY 0.5
#define GRAPHS_NUM 10000
#define PRINT_COLOR_NUM false
#define PRINT_COLORING false
#define PRINT_EDGES false
#define SAME_SEED false

class Graph {
    int V;
    double density;
    int seed;
    std::vector<std::pair<int, int>> allEdges;

public:
    std::vector<std::vector<bool>> adjMatrix;
    Graph(int V, double density, int seed);
    void printEdges(int seed = 0);
private:
    void generateAllEdges();
    void generateRandomGraph();
};

Graph::Graph(int V, double density, int seed) {
    this->V = V;
    this->density = density;
    this->seed = seed;
    adjMatrix.resize(V, std::vector<bool>(V, false)); // Initialize adjacency matrix with false (no edges)
    generateAllEdges();
    generateRandomGraph();
}

void Graph::generateAllEdges() {
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            allEdges.push_back({ i, j });
        }
    }
}

void Graph::generateRandomGraph() {
    int maxEdges = V * (V - 1) / 2;
    int numEdges = (int)density * maxEdges;

    if (SAME_SEED) {
        std::mt19937 g(seed);
        std::shuffle(allEdges.begin(), allEdges.end(), g);
    }
    else {
        std::random_device rd;
        std::mt19937 g(seed);
        std::shuffle(allEdges.begin(), allEdges.end(), rd);
    }

    std::vector<std::pair<int, int>> selectedEdges(allEdges.begin(), allEdges.begin() + numEdges);
    for (const auto& edge : selectedEdges) {
        int i = edge.first;
        int j = edge.second;
        adjMatrix[i][j] = true;
        adjMatrix[j][i] = true;
    }
}

void Graph::printEdges(int index) {
    std::cout << "Graph " << index << std::endl;
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            if (adjMatrix[i][j]) {
                std::cout << "Edge: " << i << " - " << j << std::endl;
            }
        }
    } std::cout << std::endl;
}

auto greedyColoring(const std::vector<Graph>& graphs) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> result(VERTICES, -1);
    std::vector<bool> available(VERTICES, false);
    std::vector<std::vector<int>> results;

    for (int i = 0; i < GRAPHS_NUM; ++i) {
        auto& adjMatrix = graphs[i].adjMatrix;
        result.assign(VERTICES, -1);
        available.assign(VERTICES, false);

        result[0] = 0;
        for (int u = 1; u < VERTICES; ++u) {
            for (int v = 0; v < VERTICES; ++v) {
                if (adjMatrix[u][v] && result[v] != -1) {
                    available[result[v]] = true;
                }
            }

            int cr;
            for (cr = 0; cr < VERTICES; ++cr) {
                if (!available[cr]) {
                    break;
                }
            }

            result[u] = cr;
            for (int v = 0; v < VERTICES; ++v) {
                if (adjMatrix[u][v] && result[v] != -1) {
                    available[result[v]] = false;
                }
            }
        }
        results.push_back(result);
    }
    auto end = std::chrono::high_resolution_clock::now(); // End time measurement
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // Calculate duration


    if (PRINT_COLORING || PRINT_COLOR_NUM) {
        for (int i = 0; i < GRAPHS_NUM; ++i) {
            const std::vector<int>& graphResult = results[i]; // Get the result for this graph
            std::cout << "NORMAL - graph " << i << std::endl;

            if (PRINT_COLOR_NUM) {
                int maxColor = *std::max_element(graphResult.begin(), graphResult.end());
                std::cout << "Colorings: " << maxColor + 1 << std::endl;
            }

            if (PRINT_COLORING) {
                for (int j = 0; j < VERTICES; ++j) {
                    printf("%d ", graphResult[j]);
                }std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    return std::make_pair(results, duration);
}

__global__ void cudaGreedyColoringKernel(int* d_adjMatrix, int* d_results) {
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= GRAPHS_NUM) return;

    int* adjMatrix = d_adjMatrix + u * VERTICES * VERTICES;
    int* result = d_results + u * VERTICES;
    bool available[VERTICES];

    for (int i = 0; i < VERTICES; i++) {
        result[i] = -1;
        available[i] = false;
    }

    result[0] = 0;
    for (int vertex_number = 1; vertex_number < VERTICES; ++vertex_number) {
        for (int v = 0; v < VERTICES; ++v) {
            if (adjMatrix[vertex_number * VERTICES + v] && result[v] != -1) {
                available[result[v]] = true;
            }
        }

        int cr;
        for (cr = 0; cr < VERTICES; ++cr) {
            if (!available[cr]) {
                break;
            }
        }

        result[vertex_number] = cr;
        for (int v = 0; v < VERTICES; ++v) {
            if (adjMatrix[vertex_number * VERTICES + v] && result[v] != -1) {
                available[result[v]] = false;
            }
        }
    }
}

auto cudaGreedyColoring(const std::vector<Graph>& graphs) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> h_adjMatrices(GRAPHS_NUM * VERTICES * VERTICES);
    for (int i = 0; i < GRAPHS_NUM; ++i) {
        auto& adjMatrix = graphs[i].adjMatrix;
        for (int u = 0; u < VERTICES; ++u) {
            for (int v = 0; v < VERTICES; ++v) {
                h_adjMatrices[i * VERTICES * VERTICES + u * VERTICES + v] = adjMatrix[u][v];
            }
        }
    }

    int* d_results;
    cudaMalloc((void**)&d_results, GRAPHS_NUM * VERTICES * sizeof(int));

    int* d_adjMatrix;
    cudaMalloc((void**)&d_adjMatrix, GRAPHS_NUM * VERTICES * VERTICES * sizeof(int));
    cudaMemcpy(d_adjMatrix, &h_adjMatrices[0], GRAPHS_NUM * VERTICES * VERTICES * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (GRAPHS_NUM + blockSize - 1) / blockSize;
    cudaGreedyColoringKernel << < numBlocks, blockSize >> > (d_adjMatrix, d_results);

    std::vector<int> h_results(GRAPHS_NUM * VERTICES);
    cudaMemcpy(h_results.data(), d_results, GRAPHS_NUM * VERTICES * sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    cudaFree(d_adjMatrix);
    cudaFree(d_results);

    if (PRINT_COLORING || PRINT_COLOR_NUM) {
        std::cout << std::endl;
        for (int i = 0; i < GRAPHS_NUM; ++i) {
            int* graphResult = h_results.data() + i * VERTICES;
            std::cout << "CUDA - graph " << i << std::endl;

            if (PRINT_COLOR_NUM) {
                int maxColor = *std::max_element(graphResult, graphResult + VERTICES);
                std::cout << "Colorings: " << maxColor + 1 << std::endl;
            }

            if (PRINT_COLORING) {
                for (int j = 0; j < VERTICES; ++j) {
                    std::cout << graphResult[j] << ' ';
                }std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    return std::make_pair(h_results, duration);
}


auto ompGreedyColoring(const std::vector<Graph>& graphs) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> results(GRAPHS_NUM);

#pragma omp parallel for shared(results)
    for (int i = 0; i < GRAPHS_NUM; ++i) {
        std::vector<int> result(VERTICES, -1);
        std::vector<bool> available(VERTICES, false);
        auto& adjMatrix = graphs[i].adjMatrix;
        result[0] = 0;


        for (int u = 1; u < VERTICES; ++u) {
            for (int v = 0; v < VERTICES; ++v) {
                if (adjMatrix[u][v] && result[v] != -1) {
                    available[result[v]] = true;
                }
            }

            int cr;
            for (cr = 0; cr < VERTICES; ++cr) {
                if (!available[cr]) {
                    break;
                }
            }
            result[u] = cr;

            for (int v = 0; v < VERTICES; ++v) {
                if (adjMatrix[u][v] && result[v] != -1) {
                    available[result[v]] = false;
                }
            }
        }
        results[i] = result;
        //results.push_back(result);
    }

    auto end = std::chrono::high_resolution_clock::now(); // End time measurement
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // Calculate duration


    if (PRINT_COLORING || PRINT_COLOR_NUM) {
        for (int i = 0; i < GRAPHS_NUM; ++i) {
            const std::vector<int>& graphResult = results[i]; // Get the result for this graph
            std::cout << "OMP - graph " << i << std::endl;

            if (PRINT_COLOR_NUM) {
                int maxColor = *std::max_element(graphResult.begin(), graphResult.end());
                std::cout << "Colorings: " << maxColor + 1 << std::endl;
            }

            if (PRINT_COLORING) {
                for (int j = 0; j < VERTICES; ++j) {
                    printf("%d ", graphResult[j]);
                }std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    return std::make_pair(results, duration);
}

int main() {
    std::cout << "Graphs: " << GRAPHS_NUM
        << "     Vertices: " << VERTICES
        << "     Density: " << DENSITY
        << "     Edges: " << int(VERTICES * (VERTICES - 1) / 2 * DENSITY)
        << std::endl << std::endl;

    std::vector<Graph> graphs;
    graphs.reserve(GRAPHS_NUM);
    for (int i = 0; i < GRAPHS_NUM; i++) {
        graphs.emplace_back(VERTICES, DENSITY, i);
        if (PRINT_EDGES) graphs[i].printEdges(i);
    }

    auto startNormal = std::chrono::steady_clock::now();
    auto normalPair = greedyColoring(graphs);
    //auto normalDuration = normalPair.second;
    auto normalDuration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startNormal).count();
    std::cout << "Normal greedy: " << (int)normalDuration << " us" << std::endl;

    auto startCuda = std::chrono::steady_clock::now();
    auto cudaPair = cudaGreedyColoring(graphs);
    //auto cudaDuration = cudaPair.second;
    auto cudaDuration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startCuda).count();
    std::cout << "CUDA greedy: " << (int)cudaDuration << " us" << std::endl;

    auto startOmp = std::chrono::steady_clock::now();
    auto ompPair = ompGreedyColoring(graphs);
    //auto ompDuration = ompPair.second;
    auto ompDuration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startOmp).count();
    std::cout << "OMP greedy: " << (int)ompDuration << " us" << std::endl;

    return 0;
}
