import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def fetch_db_credentials(secret_key: str) -> Dict[str, Any]:
    """
    Fetch DB credentials from AWS Secrets Manager.

    The EC2 instance has an IAM role with secretsmanager:GetSecretValue permission,
    so no explicit AWS credentials are needed — boto3 picks them up automatically
    from the instance metadata service.

    Returns dict with keys: "username", "password"
    """
    import boto3
    from botocore.exceptions import ClientError

    region = os.getenv("AWS_MAIN_REGION", "us-east-1")
    client = boto3.client("secretsmanager", region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_key)
        secret_string = response.get("SecretString")
        if not secret_string:
            raise ValueError(
                f"Secret '{secret_key}' has no SecretString (binary secrets not supported)"
            )
        credentials = json.loads(secret_string)
        logger.info(f"Successfully fetched DB credentials from secret: {secret_key}")
        return credentials
    except ClientError as e:
        code = e.response["Error"]["Code"]
        raise ValueError(
            f"AWS Secrets Manager error ({code}) fetching secret '{secret_key}'"
        ) from e
