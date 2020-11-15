#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// newton_G  -- Newton's gravitational constant (in metric units of m^3/(kg*s^2))
const double newton_G = 6.67384e-11;
const double TIME_STEP = 1;

double *get_new_vel(double arr[][2][3], const double mass[], int idx, int arr_size) {
    static double total_acc[3];
    for (int i = 0; i < arr_size; ++i) {
        double v[3];
        for (int p = 0; p < 3; ++p) {
            v[p] = arr[i][0][p] - arr[idx][0][p];
        }

        double norm = 0;
        for (double p : v)
            norm += pow(p, 2);

        double factor = newton_G * (mass[i] / norm);

        for (int a = 0; a < 3; ++a)
            total_acc[a] = factor * v[a];
    }
    return total_acc;
}

int main() {
    const int arr_size = 1 << 15; // NOLINT(hicpp-signed-bitwise)
    cout << "Starting Script for array of size " << arr_size << "." << endl;

    double arr[arr_size][2][3];
    double mass[arr_size];
    for (auto &i : arr) {
        for (int j = 0; j < 3; ++j) {
            i[0][j] = random() / (double) RAND_MAX; // NOLINT(cppcoreguidelines-narrowing-conversions)
        }
    }
    for (auto &mas : mass)
        mas = random(); // NOLINT(cppcoreguidelines-narrowing-conversions)
    cout << "Computed Position and Mass of Bodies" << endl;
    int nb_threads = omp_get_max_threads();
    cout << "Number of Threads available: " << nb_threads << endl;
    int chunked = arr_size / nb_threads;
    int bonus = arr_size - chunked * nb_threads;

    auto t1 = chrono::high_resolution_clock::now();
#pragma omp parallel for default(none) shared(nb_threads, chunked, bonus, arr, mass, TIME_STEP)
    for (int i = 0; i < nb_threads; i++) {
        int total = chunked;
        if (i + 1 == nb_threads) {
            total += bonus;
        }
        int idx = i * chunked;

        for (int n = idx; n < idx + total; ++n) {
            double *new_acc = get_new_vel(arr, mass, n, arr_size);
            for (int a = 0; a < 3; ++a) {
                arr[n][1][a] = new_acc[a] * TIME_STEP;
            }
        }
    }
    auto t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "Finished calculating step. (in " << duration << " ms)" << endl;
}
