#include <iostream>
#include <string>

#include "utils/Adder.cuh"
#include "construction/rejectionSampling/RejectionSamplingBinarySearch.cuh"
#include "construction/rejectionSampling/RejectionSamplingDirect.cuh"
#include "construction/aliasTable/buildMethod/AliasTableStack.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplitGpu.cuh"
#include "sampling/aliasTable/SamplerExpected.cuh"
#include "plots/PlotConstructionProfile.cuh"

#ifndef ALIAS_GPU_PLOT_CONSTRUCTION
#define ALIAS_GPU_PLOT_CONSTRUCTION

namespace PlotConstruction {

    void plotRejectionPerformance(int N) {
        for (int weight0 = 1e6; weight0 <= 2e7; weight0 += 1e6) {
            RejectionSamplingBinarySearch rejectionBinarySearch(N);
            float time = rejectionBinarySearch.benchmarkBuild(weight0);
            std::cout << "binarySearch;" << N << ";" << weight0 << ";" << time << std::endl;

            RejectionSamplingDirect rejectionDirect(N);
            time = rejectionDirect.benchmarkBuild(weight0);
            std::cout << "direct;" << N << ";" << weight0 << ";" << time << std::endl;
        }
    }

    void plotRejectionPerformance() {
        std::cout << "method;N;weight0;time" << std::endl;
        plotRejectionPerformance(1e6);
        plotRejectionPerformance(1e7);
    }

    template<typename T>
    void plotBuildSpeed(T &table, std::string append, int iterations = BENCHMARK_BUILD_SPEED_ITERATIONS) {
        table.preBuild();
        Timer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            table.build();
        }
        timer.stop();
        table.postBuild();
        std::cout << table.name() << ";" << (timer.elapsedMillis() / iterations) << ";" << append
                  << std::endl;
    }

    void plotBuildSpeedPackVariants() {
        std::cout << "type;duration;N" << std::endl;
        for (int n = 1e6; n <= 2e7; n += 1e6) {
            Variant variants[] = {
                    Variant(splitVariantParySearch, packVariantBasic, false),
                    Variant(splitVariantParySearch, packVariantBasic, true),
                    Variant(splitVariantParySearch, packVariantWithoutWeights, false),
                    Variant(splitVariantParySearch, packVariantWithoutWeights, true),
                    Variant(splitVariantParySearch, packVariantChunkedShared, false),
            };

            for (Variant variant : variants) {
                AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
                plotBuildSpeed(tableSplitGpu, std::to_string(n));
            }
        }
    }

    void plotBuildSpeedSplitVariants() {
        std::cout << "type;duration;splits" << std::endl;
        Variant variants[] = {
                Variant(splitVariantBasic, packVariantBasic, false),
                Variant(splitVariantParySearch, packVariantBasic, false),
                //Variant(splitVariantInverse, packVariantBasic, false),
                //Variant(splitVariantInverseParallel, packVariantBasic, false),
        };

        for (int splits = 1024; splits <= 10 * 1024; splits += 64) {
            for (Variant variant : variants) {
                AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(1e7, variant, splits, weightDistributionUniform);
                int measurements = 10;
                TableSplitTimer::TimingResult time;
                for (int i = 0; i < measurements; i++) {
                    tableSplitGpu.fullBuild();
                    time = time + tableSplitGpu.timer.get();
                }
                std::cout<<tableSplitGpu.name()<<";"<<(time.split/measurements)<<";"<<splits<<std::endl;
            }
        }
    }

    void plotBuildSpeedStructOfArrays() {
        std::cout << "type;duration;N" << std::endl;
        for (int n = 500000; n <= 10000000; n += 500000) {
            Variant variants[] = {
                    Variant(splitVariantBasic, packVariantBasic, false),
            };

            for (Variant variant : variants) {
                AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant);
                plotBuildSpeed(tableSplitGpu, std::to_string(n));

                AliasTableSplitGpu<StructOfArrays> tableSplitGpu2(n, variant);
                plotBuildSpeed(tableSplitGpu2, std::to_string(n));
            }
        }
    }

    void plotBuildSpeedWeightDistribution() {
        std::cout << "type;duration;distribution" << std::endl;
        WeightDistribution distributions[] =
                {weightDistributionUniform, weightDistributionPowerLaw2, weightDistributionRamp};
        int n = 1e7;
        for (WeightDistribution distribution : distributions) {
            Variant variant(splitVariantParySearch, packVariantBasic, true);
            variant.psaPlus = false;
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant, NUMBER_THREADS_AUTO, distribution);
            plotBuildSpeed(tableSplitGpu, std::to_string(distribution));

            variant.psaPlus = true;
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu3(n, variant, NUMBER_THREADS_AUTO, distribution);
            plotBuildSpeed(tableSplitGpu3, std::to_string(distribution));

            Variant variant2(splitVariantParySearch, packVariantChunkedShared, false);
            variant2.psaPlus = false;
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu4(n, variant2, NUMBER_THREADS_AUTO, distribution);
            plotBuildSpeed(tableSplitGpu4, std::to_string(distribution));

            RejectionSamplingBinarySearch rejectionSampling(n, distribution);
            plotBuildSpeed(rejectionSampling, std::to_string(distribution));
        }
    }

    void plotThreadNumber() {
        std::cout << "type;duration;p;N" << std::endl;

        for (int n = 1000000; n <= 10000000; n *= 10) {
            for (int p = 256; p < 4 * 1024 && p < n; p += 256) {
                Variant variant(splitVariantBasic, packVariantBasic, false);
                AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant, p, weightDistributionUniform);
                plotBuildSpeed(tableSplitGpu, std::to_string(p) + ";" + std::to_string(n));
            }
        }
    }

    void plotThreadNumberWeightVsNoWeight() {
        std::cout << "type;duration;p;N" << std::endl;

        for (int n = 1000000; n <= 10000000; n *= 10) {
            for (int p = 256; p < 6 * 1024 && p < n; p += 256) {
                Variant variant(splitVariantBasic, packVariantBasic, false);
                AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n,variant, p, weightDistributionUniform);
                plotBuildSpeed(tableSplitGpu, std::to_string(p) + ";" + std::to_string(n));

                Variant variant2(splitVariantBasic, packVariantWithoutWeights, false);
                AliasTableSplitGpu<ArrayOfStructs> tableSplitGpuNoWeight(n, variant2, p, weightDistributionUniform);
                plotBuildSpeed(tableSplitGpuNoWeight, std::to_string(p) + ";" + std::to_string(n));
            }
        }
    }

    void plotBuildSpeedWithTransfer() {
        std::cout << "type;duration;N" << std::endl;
        for (int n = 1e5; n <= 6e6; n += 1e5) {
            Variant variant(splitVariantParySearch, packVariantChunkedShared, false);
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);

            Timer timerWithoutTransfer;
            timerWithoutTransfer.start();
            tableSplitGpu.preBuild();
            for (int i = 0; i < BENCHMARK_BUILD_SPEED_ITERATIONS; i++) {
                tableSplitGpu.build();
            }
            tableSplitGpu.postBuild();
            timerWithoutTransfer.stop();
            std::cout<<"WithoutTransfer;"<<(timerWithoutTransfer.elapsedMillis()/BENCHMARK_BUILD_SPEED_ITERATIONS)<<";"<<n<<std::endl;

            float totalTimeWithTransfer = 0;
            for (int i = 0; i < BENCHMARK_BUILD_SPEED_ITERATIONS; i++) {
                Timer timerWithTransfer;
                timerWithTransfer.start();
                tableSplitGpu.preBuild();
                tableSplitGpu.build();
                timerWithTransfer.stop();
                tableSplitGpu.postBuild();
                totalTimeWithTransfer += timerWithTransfer.elapsedMillis();
            }
            std::cout<<"WithTransfer;"<<(totalTimeWithTransfer/BENCHMARK_BUILD_SPEED_ITERATIONS)<<";"<<n<<std::endl;

            Timer timerWithFullTransfer;
            timerWithFullTransfer.start();
            for (int i = 0; i < BENCHMARK_BUILD_SPEED_ITERATIONS; i++) {
                tableSplitGpu.preBuild();
                tableSplitGpu.build();
                tableSplitGpu.postBuild();
            }
            timerWithFullTransfer.stop();
            std::cout<<"FullTransfer;"<<(timerWithFullTransfer.elapsedMillis()/BENCHMARK_BUILD_SPEED_ITERATIONS)<<";"<<n<<std::endl;
        }
    }

    void plotBuildSpeedLightVsHeavy(bool randomPositions) {
        std::cout<<"step;duration;fractionHeavy"<<std::endl;
        Variant variant = Variant(splitVariantParySearch, packVariantChunkedShared, true);
        for (int fractionHeavy = 2; fractionHeavy < 100; fractionHeavy += 4) {
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(1e7, variant);

            for (int i = 0; i < tableSplitGpu.N; i++) {
                tableSplitGpu.weights[i] = 1;
            }
            int numHeavyPlaced = 0;
            int numHeavyToPlace = (int) (0.01 * fractionHeavy * tableSplitGpu.N);
            CpuPrng random = CpuPrng();
            while (numHeavyPlaced < numHeavyToPlace) {
                int index;
                if (randomPositions) {
                    index = random.next() * tableSplitGpu.N;
                } else {
                    index = numHeavyPlaced;
                }
                if (tableSplitGpu.weights[index] == 1) {
                    tableSplitGpu.weights[index] = 2;
                    numHeavyPlaced++;
                }
            }

            TableSplitTimer::TimingResult timing = PlotConstructionProfile::profile(tableSplitGpu, 250);
            std::cout<<timing.print(std::to_string(fractionHeavy))<<std::endl;
        }
    }
}
#endif // ALIAS_GPU_PLOT_CONSTRUCTION
