#!/bin/bash
# user_data.sh - provisioning script for EC2 t2.micro to run the Streamlit Docker container
set -e

# Update and install docker
apt-get update
apt-get install -y docker.io awscli
systemctl start docker
systemctl enable docker

# Add ubuntu/ ec2-user to docker group (depending on AMI user)
if id -u ubuntu >/dev/null 2>&1; then
  usermod -aG docker ubuntu || true
fi
if id -u ec2-user >/dev/null 2>&1; then
  usermod -aG docker ec2-user || true
fi

# Login to ECR (expects IAM Role on instance or aws cli configured)
# Replace <REGION> and <ACCOUNT> when running manually or set env vars via EC2 user data
#$(aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com)

# Pull image and run (replace placeholders before use)
# docker pull <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/bh_strategic_navigator:latest
# docker run -d --restart=always -p 8501:8501 --name bh_navigator \ 
#   -e STREAMLIT_SERVER_HEADLESS=true \ 
#   -e STREAMLIT_SERVER_PORT=8501 \ 
#   <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/bh_strategic_navigator:latest

# Simple healthcheck loop (optional)
# while true; do
#   sleep 30
# done
