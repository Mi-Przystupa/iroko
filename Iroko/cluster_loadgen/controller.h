#define CNTRL_PORT  20130
#define CNTRL_PCKT_SIZE 9450

typedef struct cntrl_pckt {
    char buf_size[20];
} cntrl_pckt;

int cntrl_init();
