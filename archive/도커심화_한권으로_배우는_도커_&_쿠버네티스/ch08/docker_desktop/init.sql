@"
CREATE DATABASE IF NOT EXISTS k8sDemo;
USE k8sDemo;
CREATE TABLE IF NOT EXISTS visits (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"@ | Out-File -Encoding utf8 init.sql

# mysql-init-configmap.yaml 파일 생성
@"
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-init-sql
data:
  init.sql: |
    CREATE DATABASE IF NOT EXISTS k8sDemo;
    USE k8sDemo;
    CREATE TABLE IF NOT EXISTS visits (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
"@ | Out-File -Encoding utf8 mysql-init-configmap.yaml

# ConfigMap 적용
kubectl apply -f mysql-init-configmap.yaml

# MySQL Deployment 파일 생성
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mysql
spec:
  selector:
    matchLabels:
      app: mysql
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        ports:
        - containerPort: 3306
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secrets
              key: MYSQL_ROOT_PASSWORD
        - name: MYSQL_DATABASE
          valueFrom:
            configMapKeyRef:
              name: mysql-config
              key: MYSQL_DATABASE
        - name: MYSQL_USER
          valueFrom:
            configMapKeyRef:
              name: mysql-config
              key: MYSQL_USER
        - name: MYSQL_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secrets
              key: MYSQL_PASSWORD
        volumeMounts:
        - name: mysql-persistent-storage
          mountPath: /var/lib/mysql
        - name: init-sql
          mountPath: /docker-entrypoint-initdb.d
      volumes:
      - name: mysql-persistent-storage
        persistentVolumeClaim:
          claimName: mysql-pvc
      - name: init-sql
        configMap:
          name: mysql-init-sql
"@ | Out-File -Encoding utf8 mysql-deployment.yaml
