#!/usr/bin/env bash
set -euo pipefail

# Deployment script for AWS App Runner
# Filled with user-provided account and region.

ACCOUNT=677276094413
REGION=sa-east-1
REPO=bh-strategic-navigator
TAG=latest
IMAGE=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}

echo "Using ACCOUNT=${ACCOUNT}, REGION=${REGION}"

echo "1) Create ECR repository if it does not exist"
aws ecr create-repository --repository-name ${REPO} --region ${REGION} || true

echo "2) Build Docker image"
docker build -t ${REPO}:${TAG} .

echo "3) Authenticate Docker to ECR"
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

echo "4) Tag and push image to ECR"
docker tag ${REPO}:${TAG} ${IMAGE}
docker push ${IMAGE}

echo "5) Create App Runner service from ECR image"
aws apprunner create-service \
  --service-name bh-strategic-navigator-service \
  --source-configuration ImageRepository={ImageIdentifier=\"${IMAGE}\",ImageRepositoryType=\"ECR\",ImageConfiguration={Port=\"8501\"}} \
  --region ${REGION}

echo "Deployment command submitted. Monitor App Runner service in the AWS Console or via 'aws apprunner describe-service'"
