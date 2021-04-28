#include "Timer.cuh"

Timer::Timer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

void Timer::start() {
    cudaDeviceSynchronize();
    cudaEventRecord(startEvent);
}

void Timer::stop() {
    cudaEventRecord(stopEvent);
}

float Timer::elapsedMillis() {
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    return milliseconds;
}

TableSplitTimer::TimingResult
        TableSplitTimer::TimingResult::operator + (TableSplitTimer::TimingResult other) const {
    return {sum + other.sum,
            partition + other.partition,
            prefixsum + other.prefixsum,
            split + other.split,
            pack + other.pack};
}

TableSplitTimer::TimingResult
        TableSplitTimer::TimingResult::operator / (int number) const {
    return {sum / number,
            partition / number,
            prefixsum / number,
            split / number,
            pack / number};
}

std::string TableSplitTimer::TimingResult::print(std::string append) const {
    return "Sum;" + std::to_string(sum) + ";" + append + "\n" +
           "Partition;" + std::to_string(partition) + ";" + append + "\n" +
           "Prefix sum;" + std::to_string(prefixsum) + ";" + append + "\n" +
           "Split;" + std::to_string(split) + ";" + append + "\n" +
           "Pack;" + std::to_string(pack) + ";" + append + "\n";
}

TableSplitTimer::TableSplitTimer () {
    for (cudaEvent_t &event : events) {
        cudaEventCreate(&event);
    }
}

TableSplitTimer::~TableSplitTimer () {
    for (cudaEvent_t &event : events) {
        cudaEventDestroy(event);
    }
}

void TableSplitTimer::notify(int event) {
    cudaEventRecord(events[event]);
}

TableSplitTimer::TimingResult TableSplitTimer::get() {
    cudaEventSynchronize(events[EVENT_NUM - 1]);
    float milliseconds[EVENT_NUM];
    for (int i = 1; i < EVENT_NUM; i++) {
        cudaEventElapsedTime(&milliseconds[i], events[i - 1], events[i]);
    }
    return {milliseconds[EVENT_SUM_FINISHED],
            milliseconds[EVENT_PARTITION_FINISHED],
            milliseconds[EVENT_PREFIXSUM_FINISHED],
            milliseconds[EVENT_SPLIT_FINISHED],
            milliseconds[EVENT_PACK_FINISHED]};
}
