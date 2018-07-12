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

    // char cmd[200];
    // sprintf(cmd, "tc qdisc del dev %s root", net_interface);
    // printf("cmd: %s\n", cmd);
    // system(cmd);

    // char cmd1[200];
    // sprintf(cmd1, "tc class add dev %s parent 5:0 classid 5:1 htb rate 10.0Gbit burst 15k", net_interface);
    // printf("cmd2: %s\n", cmd1);
    // system(cmd1);
exit:
    return ret;

}

void *cntrl_thread_main(void *arg)
{
    cntrl_pckt pckt;

    while(!interrupted) {
        bzero((void*)&pckt, sizeof(pckt));
        int rcvd_bytes = recv(cntrl_sock, &pckt, sizeof(cntrl_pckt), 0);
        if (rcvd_bytes != sizeof(cntrl_pckt)) {
            // fprintf(stderr, "%s Error on receive: %s\n", net_interface, strerror(errno));
        }
        tx_rate = atol(pckt.buf_size);
        printf("tx_rate: %lu\n", tx_rate);
        char cmd[200];
        sprintf(cmd, "tc class change dev %s parent 5:0 classid 5:1 htb rate %lu burst 15k", net_interface, tx_rate);
        printf("cmd: %s\n", cmd);
        int ret = system(cmd);
        if (!ret)
            perror("Problem with tc");
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
