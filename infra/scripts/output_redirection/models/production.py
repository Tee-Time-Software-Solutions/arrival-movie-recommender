from typing import Optional
from pydantic import BaseModel


class ProductionBackendOutputs(BaseModel):
      # Both clouds
      SECRETS_MANAGER_DB_CREDENTIALS_KEY: str
      DB_HOST: str
      # AWS only
      S3_MAIN_BUCKET_NAME: Optional[str] = None
      # Azure only — Key Vault name needed to call the secrets API
      AZURE_KEY_VAULT_NAME: Optional[str] = None


class ProductionFrontendOutputs(BaseModel):
      # AWS
      EC2_APP_SERVER_PUBLIC_IP: Optional[str] = None
      # Azure
      VM_APP_SERVER_PUBLIC_IP: Optional[str] = None


class ProductionAnsibleOutputs(BaseModel):
      # AWS
      EC2_APP_SERVER_PUBLIC_IP: Optional[str] = None
      EC2_APP_SERVER_SSH_USER: Optional[str] = None
      SSH_KEY_SECRET_NAME: Optional[str] = None
      # Azure
      VM_APP_SERVER_PUBLIC_IP: Optional[str] = None
      VM_APP_SERVER_SSH_USER: Optional[str] = None
      VM_APP_SERVER_SSH_PRIVATE_KEY_FILE_PATH: Optional[str] = None
