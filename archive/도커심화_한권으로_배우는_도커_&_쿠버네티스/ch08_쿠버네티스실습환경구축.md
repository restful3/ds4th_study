# Kubernetes MySQL Web Application 구조 설명
> 도커데스크톱의 쿠버네티스를 통해서 서비스를 실행하는 방법입니다. VM도, Docker Compose도 가능하지만,\
 설치과정이 복잡하며 설정에러로 많은 시간이 소모되나 도커데스크톱은 자동으로 쿠버네티스 설치 및 설정을 진행합니다.
 
## 1. 애플리케이션 구조
```plaintext
├── dockerfile              # 웹 애플리케이션 컨테이너 빌드 설정
├── package.json           # Node.js 프로젝트 설정
├── server.js             # 웹 서버 애플리케이션 코드
├── init.sql              # MySQL 초기화 스크립트
├── mysql-secrets.yaml    # MySQL 비밀 정보
├── mysql-configmap.yaml  # MySQL 설정
├── mysql-pvc.yaml       # MySQL 영구 저장소
├── mysql-deployment.yaml # MySQL 배포 설정
├── mysql-service.yaml   # MySQL 서비스
├── web-deployment.yaml  # 웹 앱 배포 설정
└── web-service.yaml    # 웹 앱 서비스
```
## 2. 진행방법

1. Docker Desktop에서 쿠버네티스 활성화
- 도커데스크톱을 실행합니다.
- 셋팅 - kubenetes - 'Enable Kubernetes', 'Show system containers(advanced)' , Apply& Restart

```bashCopy
# Kubernetes가 활성화되어 있는지 확인
kubectl cluster-info
```


2. 애플리케이션 빌드

```bashCopy
# Docker 이미지 빌드
docker build -t k8s-demo-web .
```

3. 쿠버네티스 리소스 생성

```bashCopy
# ConfigMap과 Secret 생성
kubectl apply -f mysql-secrets.yaml
kubectl apply -f mysql-configmap.yaml

# MySQL PVC 생성
kubectl apply -f mysql-pvc.yaml

# MySQL 배포
kubectl apply -f mysql-deployment.yaml
kubectl apply -f mysql-service.yaml

# MySQL이 완전히 시작될 때까지 대기
kubectl wait --for=condition=ready pod -l app=mysql

# 웹 애플리케이션 배포
kubectl apply -f web-deployment.yaml
kubectl apply -f web-service.yaml
```

4. 상태 확인
```bashCopy
# 모든 리소스 확인
kubectl get all

# MySQL 파드 로그 확인
kubectl logs -l app=mysql

# 웹 앱 파드 로그 확인
kubectl logs -l app=web-app
```

5. 애플리케이션 테스트

```bashCopy
# 서비스 IP 확인
kubectl get service web-app

# 브라우저에서 접속
# http://localhost
```

6. 스케일링 테스트
```bashCopy
# 웹 앱 스케일 아웃
kubectl scale deployment web-app --replicas=5

# 파드 상태 확인
kubectl get pods
```

7. 데이터베이스 확인

```bashCopy
# MySQL 파드 접속
kubectl exec -it $(kubectl get pod -l app=mysql -o jsonpath="{.items[0].metadata.name}") -- mysql -u root -p

# 비밀번호 입력: password123
# SQL 쿼리 실행
USE k8sDemo;
SELECT COUNT(*) FROM visits;
```

8. 정리
```bashCopy
# 모든 리소스 삭제
kubectl delete -f .
```


### 3. 파일 설명

#### 애플리케이션 파일
- **dockerfile**: Node.js 애플리케이션을 Docker 이미지로 빌드하기 위한 설정 파일. 기본 이미지, 작업 디렉토리, 의존성 설치, 포트 설정 등을 정의합니다.
- **package.json**: Node.js 프로젝트의 메타데이터와 의존성을 정의하는 파일. Express 웹 서버와 MySQL 클라이언트 라이브러리를 포함합니다.
- **server.js**: Express를 사용한 웹 서버 구현 코드. MySQL 데이터베이스와 연결하여 방문자 수를 카운트하는 기능을 제공합니다.
- **init.sql**: MySQL 데이터베이스 초기화 스크립트. 데이터베이스와 테이블 생성 쿼리가 포함되어 있습니다.

#### MySQL 관련 파일
- **mysql-secrets.yaml**: 데이터베이스 비밀번호 등 민감한 정보를 저장하는 Kubernetes Secret 리소스 파일입니다.
- **mysql-configmap.yaml**: 데이터베이스 이름, 사용자 이름 등 일반 설정값을 저장하는 ConfigMap 리소스 파일입니다.
- **mysql-pvc.yaml**: MySQL 데이터를 영구적으로 저장하기 위한 PersistentVolumeClaim 리소스 파일입니다.
- **mysql-deployment.yaml**: MySQL 서버를 배포하기 위한 Deployment 리소스 파일. 컨테이너 설정, 환경 변수, 볼륨 마운트 등을 정의합니다.
- **mysql-service.yaml**: MySQL 서버에 대한 네트워크 접근을 제공하는 Service 리소스 파일입니다.

#### 웹 애플리케이션 관련 파일
- **web-deployment.yaml**: Node.js 웹 애플리케이션을 배포하기 위한 Deployment 리소스 파일. 레플리카 수, 컨테이너 설정, 환경 변수 등을 정의합니다.
- **web-service.yaml**: 웹 애플리케이션에 대한 외부 접근을 제공하는 LoadBalancer 타입의 Service 리소스 파일입니다.

### 4. 상태 확인
```bash
# 전체 리소스 상태 확인
kubectl get all

# MySQL 로그 확인
kubectl logs -l app=mysql

# 웹 앱 로그 확인
kubectl logs -l app=web-app

# 서비스 상태 확인
kubectl get services
```

### 5. 문제 해결
```bash
# 파드 상태 및 이벤트 확인
kubectl describe pod [pod-name]

# 서비스 엔드포인트 확인
kubectl describe service [service-name]

# MySQL 데이터베이스 접속 테스트
kubectl exec -it [mysql-pod-name] -- mysql -u root -p

# 웹 앱 컨테이너 쉘 접속
kubectl exec -it [web-pod-name] -- /bin/sh

# 파드 재시작
kubectl rollout restart deployment [deployment-name]
```

## 이 예제의 특징:

- MySQL 데이터 영속성 (PVC 사용)
- ConfigMap을 통한 설정 관리
- Secret을 통한 비밀번호 관리
- 초기 데이터베이스 자동 생성
- 로드밸런싱된 웹 서비스
- 스케일러블한 웹 애플리케이션