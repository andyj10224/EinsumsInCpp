#include "einsums/OpenMP.h"

int omp_get_max_threads() {
    return 1;
}

void omp_set_num_threads(int) {
}