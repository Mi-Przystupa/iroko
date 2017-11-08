#define CNTRL_PORT  20130
#define CNTRL_PCKT_SIZE 9450

typedef struct cntrl_pckt {
    unsigned long epoch;
    size_t buf_size;
} cntrl_pckt;

int cntrl_init();
