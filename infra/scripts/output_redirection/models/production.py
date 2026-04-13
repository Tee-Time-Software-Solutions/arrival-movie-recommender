from pydantic import BaseModel


class ProductionBackendOutputs(BaseModel):
      SECRETS_MANAGER_DB_CREDENTIALS_KEY: str
      S3_MAIN_BUCKET_NAME: str
      DB_HOST: str  # matches AppSettings — injected from terraform output "db_host"


class ProductionFrontendOutputs(BaseModel):
      EC2_APP_SERVER_PUBLIC_IP: str


class ProductionAnsibleOutputs(BaseModel):
      EC2_APP_SERVER_PUBLIC_IP: str
      EC2_APP_SERVER_SSH_USER: str
      SSH_KEY_SECRET_NAME: str
