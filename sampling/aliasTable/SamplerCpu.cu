#include "SamplerCpu.cuh"

SamplerCpu::SamplerCpu(AliasTable<ArrayOfStructs> &samplingAlgorithm)
        : samplingAlgorithm(samplingAlgorithm) {

}

std::string SamplerCpu::name() {
    return "SamplerCpu";
}

int SamplerCpu::sample() {
    float rand = random.next() * samplingAlgorithm.N;
    int tableRow = rand;
    bool takeAlias = (samplingAlgorithm.W/samplingAlgorithm.N) * (rand - tableRow)
            >= samplingAlgorithm.aliasTable.weight(tableRow);
    return (1 - takeAlias) * tableRow + takeAlias * samplingAlgorithm.aliasTable.alias(tableRow);
}

double SamplerCpu::executeBenchmarkSampling(int numSamples, int *distributionOutput) {
    Timer timer;
    double W_N = samplingAlgorithm.W / samplingAlgorithm.N;
    std::vector<int> dummy(1);
    timer.start();
    for (int i = 0; i < numSamples; i++) {
        float rand = random.next() * samplingAlgorithm.N;
        int tableRow = rand;
        ArrayOfStructs::TableRow &item = samplingAlgorithm.aliasTable.rows[tableRow];
        bool takeAlias = W_N * (rand - tableRow) >= item.weight;
        int sample = takeAlias ? item.alias : tableRow;
        dummy[0] = sample;
    }
    timer.stop();
    return ((double) numSamples) / (timer.elapsedMillis() * 1000 * 1000);
}

std::vector<int> SamplerCpu::getSamplingDistribution(int numSamples) {
    assert(samplingAlgorithm.W > 0);
    std::vector<int> output(samplingAlgorithm.N);
    for (int i = 0; i < numSamples; i++) {
        output.at(sample())++;
    }
    return output;
}
