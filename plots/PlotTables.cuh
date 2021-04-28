#include <iostream>
#include <string>

#include "utils/Adder.cuh"
#include "construction/rejectionSampling/RejectionSamplingBinarySearch.cuh"
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
#include "plots/PlotSampling.cuh"

#ifndef ALIAS_GPU_PLOT_TABLES
#define ALIAS_GPU_PLOT_TABLES

namespace PlotTables {
    void basicSamplerComparison() {
        AliasTableNp<ArrayOfStructs> best(1e7, true);
        best.fullBuild();
        SamplerGpuBasic<ArrayOfStructs, XorWowPrng> samplerBest(best);

        AliasTableNp<ArrayOfStructs> worst(1e7, false);
        worst.fullBuild();
        SamplerGpuBasic<ArrayOfStructs, XorWowPrng> samplerWorst(worst);

        AliasTableNp<ArrayOfStructs> table1e7(1e7, false);
        table1e7.fullBuild();
        SamplerGpuSectioned<ArrayOfStructs> sampler1e7(table1e7, SECTION_SIZE_AUTO, true);

        AliasTableNp<ArrayOfStructs> table1e6(1e6, false);
        table1e6.fullBuild();
        SamplerGpuSectioned<ArrayOfStructs> sampler1e6(table1e6, SECTION_SIZE_AUTO, true);

        RejectionSamplingBinarySearch rej(1e7);
        rej.fullBuild();
        SamplerRejection samplerRejection(rej);

        std::cout<<SamplerCpu(best).benchmarkSampling(1e9)<<std::endl;
        std::cout<<PlotSampling::benchmarkMultiple(samplerBest, 1e9)<<std::endl;
        std::cout<<PlotSampling::benchmarkMultiple(samplerWorst, 1e9)<<std::endl;
        std::cout<<PlotSampling::benchmarkMultiple(samplerRejection, 1e9)<<std::endl;
        std::cout<<PlotSampling::benchmarkMultiple(sampler1e7, 1e9)<<std::endl;
        std::cout<<PlotSampling::benchmarkMultiple(sampler1e6, 1e9)<<std::endl;
    }

    void basicBadPackMethods() {
        std::cout << "type;duration;N" << std::endl;
        int n = 1e7;
        Variant variants[] = {
                Variant(splitVariantParySearch, packVariantSweep, false),
                Variant(splitVariantParySearch, packVariantBasic, false),
        };
        variants[1].shuffle = true;

        for (Variant variant : variants) {
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
            PlotConstruction::plotBuildSpeed(tableSplitGpu, std::to_string(n));
        }
    }

    void basicBadPackMethodPrecomputedWeight() {
        std::cout << "type;duration;N" << std::endl;
        int n = 1e7;
        Variant variants[] = {
                Variant(splitVariantParySearch, packVariantPrecomputedWeight, false),
        };

        for (Variant variant : variants) {
            AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
            PlotConstruction::plotBuildSpeed(tableSplitGpu, std::to_string(n));
        }
    }

    void comparisonLhs() {
        std::cout << "type;duration;N" << std::endl;
        int n = 1e7;

        AliasTableSplit<ArrayOfStructs> tableSplit(n, NUMBER_THREADS_AUTO, weightDistributionUniform);
        PlotConstruction::plotBuildSpeed(tableSplit, std::to_string(n));

        Variant variant(splitVariantParySearch, packVariantChunkedShared, false);
        AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, variant, NUMBER_THREADS_AUTO, weightDistributionUniform);
        PlotConstruction::plotBuildSpeed(tableSplitGpu, std::to_string(n));
    }

    void plotKsTest(int N, int numSamples) {
        std::cout << "type;item;sampled" << std::endl;
        AliasTableSplit<ArrayOfStructs> tableSplit(N, 10, weightDistributionPowerLaw1Shuffled);
        tableSplit.fullBuild();

        PlotSampling::plotSamplingDistribution("Expected",
                   SamplerExpected(tableSplit).getSamplingDistribution(numSamples));
        PlotSampling::plotSamplingDistribution("XorWow",
                   SamplerGpuBasic<ArrayOfStructs, XorWowPrng>(tableSplit).getSamplingDistribution(numSamples));
    }
}
#endif // ALIAS_GPU_PLOT_TABLES
