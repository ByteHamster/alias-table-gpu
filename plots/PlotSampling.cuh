#include <iostream>
#include <string>

#include "utils/Adder.cuh"
#include "construction/rejectionSampling/RejectionSamplingDirect.cuh"
#include "construction/aliasTable/buildMethod/AliasTableStack.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSweep.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplit.cuh"
#include "construction/aliasTable/buildMethod/AliasTableNp.cuh"
#include "construction/aliasTable/buildMethod/AliasTableSplitGpu.cuh"
#include "sampling/aliasTable/SamplerCpu.cuh"
#include "sampling/aliasTable/SamplerExpected.cuh"
#include "sampling/aliasTable/SamplerGpuBasic.cuh"
#include "sampling/aliasTable/SamplerGpuSectioned.cuh"
#include "sampling/aliasTable/SamplerGpuSectionedShared.cuh"
#include "sampling/rejectionSampling/SamplerRejection.cuh"

#ifndef ALIAS_GPU_PLOT_SAMPLING
#define ALIAS_GPU_PLOT_SAMPLING
namespace PlotSampling {

    double benchmarkMultiple(Sampler &sampler, int numSamples, int numMeasurements = 10) {
        double average = 0;
        for (int i = 0; i < numMeasurements; i++) {
            average +=sampler.benchmarkSampling(numSamples);
        }
        return average / numMeasurements;
    }

    void plotSamplingDistribution(std::string name, std::vector<int> distribution) {
        for (int i = 0; i < distribution.size(); i++) {
            std::cout << name << ";" << i << ";" << distribution.at(i) << std::endl;
        }
    }

    void plotSamplingDistribution(int numSamples) {
        #ifdef NDEBUG
            std::cerr << "Distribution can not be recorded in release version" << std::endl;
            return;
        #endif

        std::cout << "type;item;sampled" << std::endl;
        AliasTableSplit<ArrayOfStructs> tableSplit(500, 10);
        tableSplit.fullBuild();

        plotSamplingDistribution("Expected", SamplerExpected(tableSplit).getSamplingDistribution(numSamples));
        plotSamplingDistribution("XorWow", SamplerGpuBasic<ArrayOfStructs, XorWowPrng>(tableSplit).getSamplingDistribution(numSamples));
        plotSamplingDistribution("Mt", SamplerGpuBasic<ArrayOfStructs, MtPrng>(tableSplit).getSamplingDistribution(numSamples));
    }

    void plotSamplingSpeed() {
        std::cout << "type;N;gSamplesPerSec" << std::endl;
        for (int numItems = 5e5; numItems <= 1e7; numItems += 5e5) {
            RejectionSamplingDirect rejectionDirect(numItems);
            rejectionDirect.fullBuild();
            std::cout << rejectionDirect.name() << ";" << numItems << ";"
                      << SamplerRejection(rejectionDirect).benchmarkSampling(1e6) << std::endl;

            AliasTableNp<ArrayOfStructs> aliasTableNpBest(numItems, true);
            aliasTableNpBest.fullBuild();
            std::cout << aliasTableNpBest.name() << ";" << numItems << ";"
                      << SamplerGpuBasic<ArrayOfStructs>(aliasTableNpBest).benchmarkSampling(1e6) << std::endl;

            AliasTableNp<ArrayOfStructs> aliasTableNpWorst(numItems, false);
            aliasTableNpWorst.fullBuild();
            std::cout << aliasTableNpWorst.name() << ";" << numItems << ";"
                      << SamplerGpuBasic<ArrayOfStructs>(aliasTableNpWorst).benchmarkSampling(1e6) << std::endl;

            AliasTableSweep aliasTableCpu(numItems);
            aliasTableCpu.fullBuild();
            std::cout << "AliasTableCpu;" << numItems << ";"
                      << SamplerCpu(aliasTableCpu).benchmarkSampling(1e6) << std::endl;
        }
    }

    void plotSamplingSpeedSamplerMethod(int numItems, int maxNumSamples, int deltaNumSamples) {
        std::cout << "type;N;numSamples;gSamplesPerSec" << std::endl;
        Variant variant(splitVariantBasic, packVariantBasic, false);
        AliasTableSplitGpu<ArrayOfStructs> table(numItems, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
        table.fullBuild();

        for (int numSamples = 1e5; numSamples <= maxNumSamples; numSamples += deltaNumSamples) {
            SamplerGpuBasic<ArrayOfStructs> sampler(table);
            std::cout << sampler.name() << ";" << numItems << ";" << numSamples << ";"
                      << sampler.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectioned<ArrayOfStructs> samplerSectioned(table);
            std::cout << samplerSectioned.name() << ";" << numItems << ";" << numSamples << ";"
                      << samplerSectioned.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectioned<ArrayOfStructs> samplerSectionedLim(table, SECTION_SIZE_AUTO, true);
            std::cout << samplerSectionedLim.name() << ";" << numItems << ";" << numSamples << ";"
                      << samplerSectionedLim.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectionedShared samplerSectionedShared(table, 2000);
            std::cout << samplerSectionedShared.name() << ";" << numItems << ";" << numSamples << ";"
                      << samplerSectionedShared.benchmarkSampling(numSamples) << std::endl;
        }
    }

    void plotSamplingSpeedThrashing(int maxNumSamples, int deltaNumSamples) {
        std::cout << "type;N;numSamples;limited;gSamplesPerSec" << std::endl;
        int numItems = 1e6;
        Variant variant(splitVariantBasic, packVariantBasic, false);
        AliasTableSplitGpu<ArrayOfStructs> table(numItems, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
        table.fullBuild();

        for (int numSamples = 1e5; numSamples <= maxNumSamples; numSamples += deltaNumSamples) {
            SamplerGpuSectioned<ArrayOfStructs> samplerSectioned(table, 6000);
            std::cout << samplerSectioned.name() << ";" << numItems << ";" << numSamples << ";false;"
                      << samplerSectioned.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectioned<ArrayOfStructs> samplerSectioned2(table, 6000, true);
            std::cout << samplerSectioned2.name() << ";" << numItems << ";" << numSamples << ";true;"
                      << samplerSectioned2.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectionedShared samplerSectionedShared(table, 2000);
            std::cout << samplerSectionedShared.name() << ";" << numItems << ";" << numSamples << ";false;"
                      << samplerSectionedShared.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectionedShared samplerSectionedShared2(table, 2000, true);
            std::cout << samplerSectionedShared2.name() << ";" << numItems << ";" << numSamples << ";true;"
                      << samplerSectionedShared2.benchmarkSampling(numSamples) << std::endl;
        }
    }

    template <typename TableStorage>
    void plotSamplingSpeedWeightDistribution() {
        std::cout << "type;N;gSamplesPerSec;distribution" << std::endl;
        Variant variant = Variant(splitVariantParySearch, packVariantBasic, true);
        WeightDistribution distributions[] = {weightDistributionUniform, weightDistributionPowerLaw2, weightDistributionSine};
        for (WeightDistribution distribution : distributions) {
            for (int numItems = 5e5; numItems <= 1e7; numItems += 5e5) {
                AliasTableSplitGpu<TableStorage> aliasTableGpu(numItems, variant, NUMBER_THREADS_AUTO, distribution);
                aliasTableGpu.fullBuild();
                SamplerGpuSectioned<TableStorage> sampler(aliasTableGpu, SECTION_SIZE_AUTO, true);
                std::cout << sampler.name() << ";" << numItems << ";"
                          << sampler.benchmarkSampling(1e9) << ";"
                          << distribution << std::endl;
            }
        }
    }

    void plotSamplingSpeedWeightDistributionRejection() {
        std::cout << "type;N;gSamplesPerSec;distribution" << std::endl;
        WeightDistribution distributions[] = {weightDistributionUniform, weightDistributionPowerLaw2, weightDistributionSine};
        for (WeightDistribution distribution : distributions) {
            for (int numItems = 5e5; numItems <= 1e7; numItems += 5e5) {
                RejectionSamplingBinarySearch rejectionSampling(numItems, distribution);
                rejectionSampling.fullBuild();
                SamplerRejection sampler(rejectionSampling);
                std::cout << sampler.name() << ";" << numItems << ";"
                          << sampler.benchmarkSampling(1e9) << ";"
                          << distribution << std::endl;
            }
        }
    }

    template <typename TableStorage>
    double benchmarkSampling(SamplerGpuBasic<TableStorage> sampler, int numSamples, int numMeasurements) {
        double total = 0;
        for (int i = 0; i < numMeasurements; i++) {
            total += sampler.benchmarkSampling(numSamples);
        }
        return total / numMeasurements;
    }

    void plotSamplingSpeedStructOfArrays() {
        std::cout << "type;N;gSamplesPerSec" << std::endl;
        for (int numItems = 5e5; numItems <= 1e7; numItems += 2e5) {
            AliasTableNp<ArrayOfStructs> aliasTableNpBest(numItems, true);
            aliasTableNpBest.fullBuild();
            std::cout << aliasTableNpBest.name() << ";" << numItems << ";"
                      << benchmarkSampling(SamplerGpuBasic<ArrayOfStructs>(aliasTableNpBest), 1e6, 25) << std::endl;

            AliasTableNp<StructOfArrays> tableStructOfArraysBest(numItems, true);
            tableStructOfArraysBest.fullBuild();
            std::cout << tableStructOfArraysBest.name() << ";" << numItems << ";"
                      << benchmarkSampling(SamplerGpuBasic<StructOfArrays>(tableStructOfArraysBest), 1e6, 25) << std::endl;

            AliasTableNp<StructOfArrays> tableStructOfArraysWorst(numItems, false);
            tableStructOfArraysWorst.fullBuild();
            std::cout << tableStructOfArraysWorst.name() << ";" << numItems << ";"
                      << benchmarkSampling(SamplerGpuBasic<StructOfArrays>(tableStructOfArraysWorst), 1e6, 25) << std::endl;
        }
    }

    void plotNumberOfSamples() {
        std::cout << "type;N;numSamples;gSamplesPerSec" << std::endl;
        for (int numItems = 1e5; numItems <= 1e7; numItems *= 10) {
            Variant variant(splitVariantBasic, packVariantBasic, false);
            AliasTableSplitGpu<ArrayOfStructs> table(numItems, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
            table.fullBuild();

            for (int numSamples = 1e5; numSamples <= 5e6; numSamples += 5e4) {
                std::cout << table.name() << ";" << numItems << ";" << numSamples << ";"
                          << SamplerGpuBasic<ArrayOfStructs>(table).benchmarkSampling(numSamples) << std::endl;
            }
        }
    }

    void plotSamplingSpeedMethodHeatmap() {
        std::cout << "N;numSamples;best" << std::endl;

        for (int numItems = 1e4; numItems <= 6e6; numItems += 8e4) {
            Variant variant(splitVariantBasic, packVariantBasic, false);
            AliasTableSplitGpu<ArrayOfStructs> table(numItems, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
            table.fullBuild();

            for (int numSamples = 1e5; numSamples <= 3e7; numSamples += 8e5) {
                double max = 0;
                std::string maxName = "";

                SamplerGpuBasic<ArrayOfStructs> sampler(table);
                double tmp = sampler.benchmarkSampling(numSamples);
                if (tmp > max) {
                    max = tmp;
                    maxName = sampler.name();
                }

                SamplerGpuSectioned<ArrayOfStructs> samplerSectioned(table, 8000);
                tmp = samplerSectioned.benchmarkSampling(numSamples);
                if (tmp > max) {
                    max = tmp;
                    maxName = samplerSectioned.name();
                }

                SamplerGpuSectioned<ArrayOfStructs> samplerSectionedLimited(table, 8000, true);
                tmp = samplerSectionedLimited.benchmarkSampling(numSamples);
                if (tmp > max) {
                    max = tmp;
                    maxName = samplerSectionedLimited.name();
                }

                SamplerGpuSectionedShared samplerSectionedShared(table);
                tmp = samplerSectionedShared.benchmarkSampling(numSamples);
                if (tmp > max) {
                    maxName = samplerSectionedShared.name();
                }

                std::cout << numItems << ";" << numSamples << ";" << maxName << std::endl;
            }
        }
    }

    void plotSamplingSpeedSectionedItemsPerSection(int numSamples, int maxSectionSize) {
        std::cout << "type;numSamples;sectionSize;gSamplesPerSec" << std::endl;

        int numItems = 1e6;
        for (int sectionSize = 1024; sectionSize <= maxSectionSize; sectionSize += 32) {
            Variant variant(splitVariantBasic, packVariantBasic, false);
            AliasTableSplitGpu<ArrayOfStructs> table(numItems, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
            table.fullBuild();

            SamplerGpuBasic<ArrayOfStructs> sampler(table); // Independent from section size
            std::cout << sampler.name() << ";" << numSamples << ";" << sectionSize
                      << ";" << sampler.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectioned<ArrayOfStructs> samplerSectioned(table, sectionSize);
            std::cout << samplerSectioned.name() << ";" << numSamples << ";" << sectionSize
                      << ";" << samplerSectioned.benchmarkSampling(numSamples) << std::endl;

            SamplerGpuSectioned<ArrayOfStructs> samplerSectionedLimited(table, sectionSize, true);
            std::cout << samplerSectionedLimited.name() << ";" << numSamples << ";" << sectionSize
                      << ";" << samplerSectionedLimited.benchmarkSampling(numSamples) << std::endl;

            if (sectionSize <= 3000) {
                SamplerGpuSectionedShared samplerSectionedShared(table, sectionSize);
                std::cout << samplerSectionedShared.name() << ";" << numSamples << ";" << sectionSize
                          << ";" << samplerSectionedShared.benchmarkSampling(numSamples) << std::endl;
            }
        }
    }

    template <typename Prng>
    void plotSamplerGroupSize(int maxThreads) {
        std::cout << "type;gSamplesPerSec;threads;blocks" << std::endl;

        Variant variant(splitVariantParySearch, packVariantBasic, true);
        AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(1e6, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
        tableSplitGpu.fullBuild();

        for (int numBlocks = 10; numBlocks <= 150; numBlocks += 3) {
            for (int numThreads = 32; numThreads <= maxThreads; numThreads += 32) {
                SamplerGpuBasic<ArrayOfStructs, Prng> sampler(tableSplitGpu, numBlocks, numThreads);
                std::cout << sampler.name() << ";" << sampler.benchmarkSampling(1e9)
                        << ";" << numThreads << ";" << numBlocks << std::endl;
            }
        }
    }

    void directMethodComparison(int tableSize) {
        std::cout << "method;gSamplesPerSec" << std::endl;
        Variant variant(splitVariantParySearch, packVariantBasic, true);
        AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(tableSize, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
        tableSplitGpu.fullBuild();

        RejectionSamplingBinarySearch rejectionSampling(tableSize, weightDistributionUniform);
        rejectionSampling.fullBuild();

        std::vector<Sampler *> samplers;
        samplers.push_back(new SamplerCpu(tableSplitGpu));
        samplers.push_back(new SamplerGpuBasic<ArrayOfStructs, XorWowPrng>(tableSplitGpu));
        samplers.push_back(new SamplerGpuSectioned<ArrayOfStructs>(tableSplitGpu));
        samplers.push_back(new SamplerGpuSectioned<ArrayOfStructs>(tableSplitGpu, SECTION_SIZE_AUTO, true));
        samplers.push_back(new SamplerGpuSectionedShared(tableSplitGpu));
        samplers.push_back(new SamplerRejection(rejectionSampling));

        for (Sampler *sampler : samplers) {
            std::cout<<sampler->name()<<";"<<sampler->benchmarkSampling(1e9)<<std::endl;
        }
    }

    void testCacheConfig(enum cudaFuncCache config, std::string configName) {
        cudaDeviceSetCacheConfig(config);

        Variant variant = Variant(splitVariantParySearch, packVariantBasic, true);
        AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(1e6, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
        tableSplitGpu.fullBuild(); // Construction is not influenced by cache size

        SamplerGpuSectioned<ArrayOfStructs> sampler3(tableSplitGpu, 2000);
        std::cout<<configName<<";"<<sampler3.name()<<" 2000;"<<sampler3.benchmarkSampling(1e9)<<std::endl;

        SamplerGpuSectioned<ArrayOfStructs> sampler3a(tableSplitGpu, 6000);
        std::cout<<configName<<";"<<sampler3a.name()<<" 6000;"<<sampler3a.benchmarkSampling(1e9)<<std::endl;

        SamplerGpuSectioned<ArrayOfStructs> sampler4(tableSplitGpu, 2000, true);
        std::cout<<configName<<";"<<sampler4.name()<<" 2000;"<<sampler4.benchmarkSampling(1e9)<<std::endl;

        SamplerGpuSectioned<ArrayOfStructs> sampler4a(tableSplitGpu, 6000, true);
        std::cout<<configName<<";"<<sampler4a.name()<<" 6000;"<<sampler4a.benchmarkSampling(1e9)<<std::endl;

        SamplerGpuSectionedShared sampler5(tableSplitGpu, 2000);
        std::cout<<configName<<";"<<sampler5.name()<<";"<<sampler5.benchmarkSampling(1e9)<<std::endl;
        cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    }

    void cacheConfig() {
        #ifdef SET_CACHE_CONFIG
            std::cerr<<"Trying to test cache config method but sampler sets cache config itself";
            return;
        #endif
        std::cout<<"cache;method;gSamplesPerSec"<<std::endl;
        testCacheConfig(cudaFuncCachePreferNone, "None");
        testCacheConfig(cudaFuncCachePreferShared, "Shared");
        testCacheConfig(cudaFuncCachePreferL1, "L1");
    }
}
#endif // ALIAS_GPU_PLOT_SAMPLING
