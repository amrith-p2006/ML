#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 12345

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    char *msg = "Hello from server";

    server_fd=socket(AF_INET,SOCK_STREAM,0);

    address.sin_family=AF_INET;
    address.sin_addr.s_addr=INADDR_ANY;
    address.sin_port=htons(PORT);
    
    bind(server_fd,(struct sockaddr*)&address,sizeof(address));

    listen(server_fd,3);
    printf("Server is listening on port %d\n", PORT);

    new_socket=accept(server_fd,(struct sockaddr*)&address,(socklen_t*)&addrlen);


    read(new_socket,buffer,1024);
    printf("Message from client: %s\n",buffer);

    send(new_socket,msg,strlen(msg),0);
    printf("Message sent ...");

    close(new_socket);
    close(server_fd);
    return 0;

}