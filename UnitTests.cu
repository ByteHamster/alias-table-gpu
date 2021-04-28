#include "UnitTests.cuh"

void UnitTests::execute() {
    PrefixSum::unitTest();
    Adder::unitTest();
    testTableCorrectness();
    testSamplingDistribution();
}

void UnitTests::testTableCorrectness() {

    for (int distributionIdx = 1; distributionIdx <= MAX_WEIGHT_DISTRIBUTION; distributionIdx++) {
        WeightDistribution distribution = (WeightDistribution) distributionIdx;
        AliasTableStack tableStack(10000, distribution);
        tableStack.fullBuild();

        AliasTableSweep tableSweep(10000, distribution);
        tableSweep.fullBuild();

        AliasTableSplit<ArrayOfStructs> tableSplit(10000, 55, distribution);
        tableSplit.fullBuild();

        AliasTableSplit<ArrayOfStructs> tableSplit2(1000000, 1024, distribution);
        tableSplit2.fullBuild();
    }

    Variant variants[] = {
            Variant(splitVariantInverse, packVariantBasic, false),
            Variant(splitVariantInverseParallel, packVariantBasic, false),
            Variant(splitVariantBasic, packVariantBasic, false),
            Variant(splitVariantBasic, packVariantBasic, true),
            Variant(splitVariantBasic, packVariantWithoutWeights, false),
            Variant(splitVariantBasic, packVariantWithoutWeights, true),
            Variant(splitVariantBasic, packVariantSweep, false),
    };

    for (Variant variant : variants) {
        AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(10000, variant);
        tableSplitGpu.fullBuild();

        Variant variantWithoutPsaPlus = variant;
        variantWithoutPsaPlus.psaPlus = false;
        AliasTableSplitGpu<StructOfArrays> tableSplitGpu2(10000, variantWithoutPsaPlus);
        tableSplitGpu2.fullBuild();
    }

    // Starts a group with 0 threads eligible for packing (rounding)
    Variant variant(splitVariantBasic, packVariantBasic, true);
    assert(AliasTableSplitGpu<ArrayOfStructs>(505711, variant, 16576, weightDistributionOneLight).fullBuild());

    // Only reasonably fast variants, ignoring the inverse methods. The test takes ages otherwise.
    Variant interestingVariants[] = {
            Variant(splitVariantBasic, packVariantBasic, false),
            Variant(splitVariantBasic, packVariantBasic, true),
            Variant(splitVariantBasic, packVariantWithoutWeights, false),
            Variant(splitVariantBasic, packVariantWithoutWeights, true),
            Variant(splitVariantBasic, packVariantSweep, false),
            Variant(splitVariantParySearch, packVariantBasic, false),
            Variant(splitVariantParySearch, packVariantBasic, true),
            Variant(splitVariantParySearch, packVariantChunkedShared, false),
    };
    int numTests = 20;
    for (int i = 0; i < numTests; i++) {
        int variant = rand() % (sizeof(interestingVariants) / sizeof(Variant));
        // When drawing random numbers from 0..1e7, most of them are located between 1e6 and 1e7.
        // I also want to test small numbers like 1e5 with the same probability, so I first decide
        // on a maximum exponent and then draw a random number in the range of that exponent.
        int n_max = pow(10, 3 + (rand() % (int) 5)); // Between 1e3 and 1e7
        int n = 100 + rand() % (1 * n_max); // max: 1e7
        int threads = NUMBER_THREADS_AUTO; // rand() % (n / 3); // Can select numbers that are too much for shared weight method
        int distributionIdx = (rand() % MAX_WEIGHT_DISTRIBUTION) + 1;
        WeightDistribution distribution = (WeightDistribution) distributionIdx;

        interestingVariants[variant].setPsaPlusIfSupported((rand() % 2) == 1);
        AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(n, interestingVariants[variant], threads, distribution);
        std::cout<<"\r"<<"Test "<<(i+1)<<" of "<<numTests<<": "<<tableSplitGpu.name()
                 <<" with n="<<n<<", psaPlus="<<(interestingVariants[variant].psaPlus ? "yes" : "no")
                 <<", distribution="<<distribution;
        if (!tableSplitGpu.fullBuild()) {
            std::cout<<"\r"<<"Failed: "<<tableSplitGpu.name()<<" with n="<<n
                    <<", distribution="<<distribution<<std::endl;
        }

        interestingVariants[variant].setPsaPlusIfSupported(false);
        AliasTableSplitGpu<StructOfArrays> tableSplitGpu2(n, interestingVariants[variant], threads, distribution);
        std::cout<<"\r"<<"Test "<<(i+1)<<" of "<<numTests<<": "<<tableSplitGpu2.name()
                 <<" with n="<<n<<", distribution="<<distribution;
        if (!tableSplitGpu2.fullBuild()) {
            std::cout<<"\r"<<"Failed: "<<tableSplitGpu2.name()<<" with n="<<n
                    <<", distribution="<<distribution<<std::endl;
        }
    }
    std::cout<<std::endl;
}

void UnitTests::testSamplingDistribution() {
    #ifndef DEBUG_SUPPORT_SAMPLING_DISTRIBUTION
        std::cout<<"Skipping sampling distribution unit test because DEBUG_SUPPORT_SAMPLING_DISTRIBUTION is not set"<<std::endl;
        return;
    #endif

    int numSamples = 2e6;
    int tableSize = 1e4;

    Variant variant = Variant(splitVariantBasic, packVariantBasic, false);
    AliasTableSplitGpu<ArrayOfStructs> tableSplitGpu(tableSize, variant, weightDistributionSine);
    tableSplitGpu.fullBuild();
    AliasTableSplitGpu<StructOfArrays> tableSplitGpuStructOfArrays(tableSize, variant, weightDistributionSine);
    tableSplitGpuStructOfArrays.fullBuild();
    RejectionSamplingBinarySearch rejectionSampling(tableSize, weightDistributionSine);
    rejectionSampling.fullBuild();
    RejectionSamplingPArySearch rejectionSamplingPary(tableSize, weightDistributionSine);
    rejectionSamplingPary.fullBuild();

    std::vector<int> distributionExpected = SamplerExpected(tableSplitGpu).getSamplingDistribution(numSamples);

    std::vector<Sampler *> samplers;
    samplers.push_back(new SamplerCpu(tableSplitGpu));
    samplers.push_back(new SamplerGpuBasic<ArrayOfStructs, XorWowPrng>(tableSplitGpu));
    samplers.push_back(new SamplerGpuBasic<ArrayOfStructs, MtPrng>(tableSplitGpu));
    samplers.push_back(new SamplerGpuSectioned<ArrayOfStructs>(tableSplitGpu, tableSize / 4));
    samplers.push_back(new SamplerGpuSectionedShared(tableSplitGpu, tableSize / 4));
    samplers.push_back(new SamplerGpuBasic<StructOfArrays, XorWowPrng>(tableSplitGpuStructOfArrays));
    samplers.push_back(new SamplerGpuBasic<StructOfArrays, MtPrng>(tableSplitGpuStructOfArrays));
    samplers.push_back(new SamplerGpuSectioned<StructOfArrays>(tableSplitGpuStructOfArrays, tableSize / 4));
    samplers.push_back(new SamplerRejection(rejectionSampling));
    samplers.push_back(new SamplerRejection(rejectionSamplingPary));
    samplers.push_back(new SamplerRejectionGiveUp(rejectionSampling));

    for (int i = 0; i < samplers.size(); i++) {
        std::cout<<"\rTest "<<(i+1)<<" of "<<samplers.size()<<": "<<samplers.at(i)->name();
        std::vector<int> distributionSampled = samplers.at(i)->getSamplingDistribution(numSamples);

        for (int k = 0; k < tableSize; k++) {
            int difference = std::abs(distributionExpected[k] - distributionSampled[k]);
            if (difference >= numSamples / tableSize) {
                std::cerr<<"\r"<<"Failed: "<<samplers.at(i)->name()<<", expected "
                         <<distributionExpected[k]<<" samples but got "<<distributionSampled[k]<<" samples at position "
                         <<k<<std::endl;
            }
        }
        free(samplers.at(i));
    }
    std::cout<<std::endl;
}
