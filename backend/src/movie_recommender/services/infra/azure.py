import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def fetch_db_credentials(secret_name: str, key_vault_name: str) -> Dict[str, Any]:
    """
    Fetch DB credentials from Azure Key Vault.

    The VM has a managed identity with Key Vault Secrets User role, so
    DefaultAzureCredential picks it up automatically — no credentials in env.

    Returns dict with keys: "username", "password"
    """
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    vault_url = f"https://{key_vault_name}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)

    secret = client.get_secret(secret_name)
    credentials = json.loads(secret.value)
    logger.info(
        f"Successfully fetched DB credentials from Key Vault secret: {secret_name}"
    )
    return credentials
