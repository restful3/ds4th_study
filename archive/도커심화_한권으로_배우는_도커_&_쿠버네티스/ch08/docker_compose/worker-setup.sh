version: '3.8'

services:
  myserver01:
    image: ubuntu:20.04
    container_name: myserver01
    privileged: true
    environment:
      - NODE_ROLE=master
      - DEBIAN_FRONTEND=noninteractive
    networks:
      - k8s_network
    volumes:
      - ./master-setup.sh:/master-setup.sh  # master-setup.sh 파일을 컨테이너로 복사
    command: >
      /bin/bash -c '
      chmod +x /master-setup.sh && /master-setup.sh && sleep infinity'

  myserver02:
    image: ubuntu:20.04
    container_name: myserver02
    privileged: true
    environment:
      - NODE_ROLE=worker
      - DEBIAN_FRONTEND=noninteractive
    networks:
      - k8s_network
    volumes:
      - ./worker-setup.sh:/worker-setup.sh  # worker-setup.sh 파일을 컨테이너로 복사
    command: >
      /bin/bash -c '
      chmod +x /worker-setup.sh && /worker-setup.sh && sleep infinity'

  myserver03:
    image: ubuntu:20.04
    container_name: myserver03
    privileged: true
    environment:
      - NODE_ROLE=worker
      - DEBIAN_FRONTEND=noninteractive
    networks:
      - k8s_network
    volumes:
      - ./worker-setup.sh:/worker-setup.sh
    command: >
      /bin/bash -c '
      chmod +x /worker-setup.sh && /worker-setup.sh && sleep infinity'

networks:
  k8s_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.18.0.0/16
