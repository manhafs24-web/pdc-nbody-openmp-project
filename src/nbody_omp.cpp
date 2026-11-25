#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <omp.h>

const double G   = 1.0;
const double EPS = 0.05;
const double DT  = 0.001;

// Parallel over particles i. Each thread owns its own i values and writes
// only to ax[i], ay[i], az[i], so there are no race conditions.
void computeForces(int n,
                   const std::vector<double>& x, const std::vector<double>& y,
                   const std::vector<double>& z, const std::vector<double>& m,
                   std::vector<double>& ax, std::vector<double>& ay,
                   std::vector<double>& az)
{
    const double eps2 = EPS * EPS;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double axi = 0.0, ayi = 0.0, azi = 0.0;   // private to each iteration
        const double xi = x[i], yi = y[i], zi = z[i];

        for (int j = 0; j < n; j++) {
            double dx = x[j] - xi;
            double dy = y[j] - yi;
            double dz = z[j] - zi;
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double inv = 1.0 / std::sqrt(r2);
            double s = G * m[j] * inv * inv * inv;
            axi += dx * s;
            ayi += dy * s;
            azi += dz * s;
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }
}

// Energy is outside the timed loop. It is parallelized only to save waiting time.
double totalEnergy(int n,
                   const std::vector<double>& x, const std::vector<double>& y,
                   const std::vector<double>& z, const std::vector<double>& vx,
                   const std::vector<double>& vy, const std::vector<double>& vz,
                   const std::vector<double>& m)
{
    const double eps2 = EPS * EPS;
    double kinetic = 0.0, potential = 0.0;

    #pragma omp parallel for reduction(+:kinetic)
    for (int i = 0; i < n; i++) {
        kinetic += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    // Triangular loop (j starts at i+1), so work per i is uneven: use dynamic
    #pragma omp parallel for reduction(+:potential) schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            potential -= G * m[i] * m[j] / std::sqrt(r2);
        }
    }

    return kinetic + potential;
}

int main(int argc, char* argv[])
{
    int n             = (argc > 1) ? std::atoi(argv[1]) : 1000;
    int steps         = (argc > 2) ? std::atoi(argv[2]) : 10;
    unsigned int seed = (argc > 3) ? (unsigned int)std::atoi(argv[3]) : 42;
    int threads       = (argc > 4) ? std::atoi(argv[4]) : omp_get_max_threads();

    omp_set_num_threads(threads);

    std::vector<double> x(n), y(n), z(n), vx(n, 0.0), vy(n, 0.0), vz(n, 0.0), m(n);
    std::vector<double> ax(n), ay(n), az(n);

    // Same random generation as the sequential version (keep identical)
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> pos(-1.0, 1.0);
    for (int i = 0; i < n; i++) {
        x[i] = pos(gen);
        y[i] = pos(gen);
        z[i] = pos(gen);
        m[i] = 1.0 / n;
    }

    computeForces(n, x, y, z, m, ax, ay, az);
    double e0 = totalEnergy(n, x, y, z, vx, vy, vz, m);

    double t0 = omp_get_wtime();

    for (int step = 0; step < steps; step++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            vx[i] += ax[i] * 0.5 * DT;
            vy[i] += ay[i] * 0.5 * DT;
            vz[i] += az[i] * 0.5 * DT;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            x[i] += vx[i] * DT;
            y[i] += vy[i] * DT;
            z[i] += vz[i] * DT;
        }

        computeForces(n, x, y, z, m, ax, ay, az);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            vx[i] += ax[i] * 0.5 * DT;
            vy[i] += ay[i] * 0.5 * DT;
            vz[i] += az[i] * 0.5 * DT;
        }
    }

    double t1 = omp_get_wtime();

    double e1 = totalEnergy(n, x, y, z, vx, vy, vz, m);

    double sumx = 0.0;
    for (int i = 0; i < n; i++) sumx += x[i];

    std::cout << "N = " << n << ", steps = " << steps << ", threads = " << threads << "\n";
    std::cout << "Time (s): " << std::fixed << std::setprecision(3) << (t1 - t0) << "\n";
    std::cout << std::setprecision(12);
    std::cout << "Initial energy: " << e0 << "\n";
    std::cout << "Final energy:   " << e1 << "\n";
    std::cout << "x[0], y[0], z[0]: " << x[0] << " " << y[0] << " " << z[0] << "\n";
    std::cout << "Sum of x: " << sumx << "\n";

    return 0;
}