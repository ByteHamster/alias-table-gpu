#ifndef ALIAS_GPU_TIMER_CUH
#define ALIAS_GPU_TIMER_CUH

#include <chrono>
#include <string>

class Timer {
    public:
        Timer();
        void start();
        void stop();
        float elapsedMillis();
    private:
        cudaEvent_t startEvent, stopEvent;
};

#define EVENT_START 0
#define EVENT_SUM_FINISHED 1
#define EVENT_PARTITION_FINISHED 2
#define EVENT_PREFIXSUM_FINISHED 3
#define EVENT_SPLIT_FINISHED 4
#define EVENT_PACK_FINISHED 5
#define EVENT_NUM 6

class TableSplitTimer {
    public:
        struct TimingResult {
            float sum = 0;
            float partition = 0;
            float prefixsum = 0;
            float split = 0;
            float pack = 0;

            TimingResult operator + (TimingResult other) const;
            TimingResult operator / (int number) const;
            std::string print(std::string append) const;
        };

        TableSplitTimer();
        ~TableSplitTimer();
        void notify(int event);
        TimingResult get();
    private:
        cudaEvent_t events[EVENT_NUM] = {};
};


#endif //ALIAS_GPU_TIMER_CUH
