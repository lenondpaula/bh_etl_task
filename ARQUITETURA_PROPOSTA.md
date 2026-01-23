# Evolução do MVP para Data Lakehouse na AWS

Resumo curto da proposta para evoluir o projeto "BH Strategic Navigator" para um Data Lakehouse gerenciável, escalável e consultável.

Fluxo de dados proposto
- Ingestão: fontes da Prefeitura (APIs REST e/ou FTP de arquivos) são consumidas por jobs de ingestão.
- Raw Zone (S3): os arquivos/JSON brutos são colocados em um bucket S3 na zona "raw/" com partição por data de ingestão e retenção configurável.
- Tratamento (AWS Glue / Spark): jobs Glue (Spark) são executados para limpeza, validação e transformação, registrando metadados no Glue Data Catalog.
- Armazenamento Otimizado (Parquet): dados tratados são gravados em S3 em formato Parquet particionado (ex: ano=/mes=/dia=) e compactado (snappy) para leitura eficiente.
- Consulta (Amazon Athena): o Glue Data Catalog permite consultas SQL via Athena para análises ad-hoc e geração de datasets agregados.
- Visualização: o app (container Streamlit) consome resultados via Athena (ou diretamente S3/Glue) para visualizações interativas.

Orquestração e Frequência
- Use Apache Airflow (gerenciado via MWAA ou self-managed) para orquestrar pipelines: ingestão, jobs Glue e atualizações do catálogo.
- Defina jobs de ingestão e transformação com agendamento mensal (ou conforme SLA) e mecanismos de retry/alerta.

Segurança e Governança
- Armazene credenciais em AWS Secrets Manager ou IAM roles atribuídos às instâncias/containers (evitar credenciais hard-coded).
- Controle de acesso via IAM (políticas por princípio do menor privilégio) e bucket policies para S3.
- Catalogação de tabelas e schemas no Glue Data Catalog para governança de dados.

Escalabilidade e Performance
- Particione dados por data e por atributo relevante (ex: cidade/bairro) para reduzir escaneamento no Athena.
- Configure compaction e uso de Parquet/ORC com compressão para reduzir custos e latência.

Deploy do container em EC2 t2.micro (Free Tier) — passos resumidos
1. Build e push da imagem para um registry (ex: Amazon ECR): build localmente e `docker push` para ECR.
2. Provisionar uma instância EC2 t2.micro com Amazon Linux 2 (ou Ubuntu), abrir porta 8501 no Security Group apenas para IPs autorizados.
3. Instalar Docker na instância e efetuar login no ECR.
4. Pull da imagem e executar o container expondo a porta 8501; ligar um volume EBS para persistência de arquivos locais se necessário:

```bash
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker $USER
docker login -u AWS -p "$(aws ecr get-login-password --region <REGION>)" <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
docker pull <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/bh_strategic_navigator:latest
docker run -d -p 8501:8501 --name bh-navigator \ \
  -e AWS_REGION=<REGION> \ 
  -e S3_BUCKET=<BUCKET_NAME> \ 
  <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/bh_strategic_navigator:latest
```

Notas operacionais
- Para produção, prefira ECS/Fargate ou EKS em vez de EC2 para gerenciamento e escalabilidade.
- Use CloudWatch para logs e métricas, e configure alarmes para falhas de ingestão.
- Automatize o provisionamento com Terraform/CloudFormation e integre CI/CD (GitHub Actions) para builds e deployments automáticos.
