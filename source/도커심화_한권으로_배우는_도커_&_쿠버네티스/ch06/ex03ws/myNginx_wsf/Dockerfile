FROM nginx:1.25.3
RUN rm /etc/nginx/conf.d/default.conf # 컨테이너 내부에서 기본 Nginx 설정 파일을 삭제합니다.
COPY default.conf /etc/nginx/conf.d/ # 호스트 시스템의 default.conf 파일을 컨테이너의 /etc/nginx/conf.d/ 디렉토리로 복사합니다
CMD ["nginx", "-g", "daemon off;"]