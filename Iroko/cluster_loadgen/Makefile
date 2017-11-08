CC=g++
CFLAGS=-Wall -O3 -g
LFLAGS=-pthread

loadgen: main.o server.o client.o controller.o
	$(CC) $(LFLAGS) main.o server.o client.o controller.o -o $@

main.o: main.cc common.h
	$(CC) -c $(CFLAGS) main.cc -o $@

server.o: server.cc common.h
	$(CC) -c $(CFLAGS) server.cc -o $@

client.o: client.cc common.h
	$(CC) -c $(CFLAGS) client.cc -o $@

controller.o: controller.cc controller.h common.h
	$(CC) -c $(CFLAGS) controller.cc -o $@

clean:
	rm -rf loadgen *.o

