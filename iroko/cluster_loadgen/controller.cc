#include<limits.h>
#include<stdlib.h>

#include "common.h"
#include "controller.h"

int cntrl_sock;
struct sockaddr_in cntrl_addr;
size_t tx_rate;
extern const char *net_interface;

int cntrl_init()
{
    int ret = 0;
    tx_rate = INT_MAX;
    
    if ((cntrl_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
        perror("Controller Socket");
        ret = cntrl_sock;
        goto exit;
    }

    bzero((char*) &cntrl_addr, sizeof(cntrl_addr));
    cntrl_addr.sin_family = AF_INET;
    cntrl_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    cntrl_addr.sin_port = htons((unsigned short) CNTRL_PORT);

    if (bind(cntrl_sock, (struct sockaddr*) &cntrl_addr, sizeof(cntrl_addr)) < 0) {
        perror("Bind");
        ret = errno;
        goto exit;
    }

    printf("Interface: %s\n", net_interface);

    char cmd[200];
    sprintf(cmd, "tc qdisc add dev %s root tbf rate 100gbit burst 32kbit latency 400ms", net_interface);
    printf("cmd: %s\n", cmd);
    system(cmd);

exit:
    return ret;

}

void *cntrl_thread_main(void *arg)
{
    int rcvd_bytes = 0;
    cntrl_pckt pckt;

    while(!interrupted) {
        bzero((void*)&pckt, sizeof(pckt));
        rcvd_bytes = recv(cntrl_sock, &pckt, sizeof(cntrl_pckt), 0);
        if (rcvd_bytes != sizeof(cntrl_pckt)) {
            perror("Received control packet");
        }
        tx_rate = atol(pckt.buf_size);
        printf("tx_rate: %lu\n", tx_rate);
        char cmd[200];
        sprintf(cmd, "tc qdisc change dev %s root tbf rate %lu burst 32kbit latency 400ms", net_interface, tx_rate);
        printf("cmd: %s\n", cmd);
        system(cmd);

    }

    return NULL;
}

#if 0
int main()
{
    controller_init();
    controller_main();
    return 0;
}
#endif
