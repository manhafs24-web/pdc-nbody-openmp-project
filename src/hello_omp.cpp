#include <omp.h>
#include <iostream>

int main() {
    std::cout << "Max threads: " << omp_get_max_threads() << "\n";

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        #pragma omp critical
        std::cout << "Hello from thread " << id << "\n";
    }

    return 0;
}