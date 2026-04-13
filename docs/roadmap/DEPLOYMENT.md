# Deployment — Movie Recommender (Prod AWS)

> Local is working (incl. Discord + Grafana). Goal: get the same stack running on AWS prod with minimum re-runs.
> Dropped: Google Analytics, nginx complexity, MLflow.

---

## Architecture (prod)

```
Internet → EC2 (t3.medium)
  nginx:80
    /       → static dist (frontend/app/dist)
    /api/   → backend:8000  (rate-limited: 30r/m)

  EC2 containers: backend, redis, neo4j, prometheus, grafana
  AWS managed:   RDS PostgreSQL 16 (private subnet)
```

---

## Pre-flight Checklist — Status

All 7 items below have been applied to the code. Review them before running terraform.

### 1. Terraform — RDS module ✓
`infra/terraform/aws/modules/rds/main.tf` — changed to PostgreSQL 16.3, added `db_name = "app_db"`.
Other attributes (`db.t3.small`, `allocated_storage=10`, `skip_final_snapshot=true`) stay the same — only the engine changed.

### 2. Terraform — production tfvars ✓
`infra/terraform/aws/environment/production/terraform.tfvars` — `project_name` changed to `"movie-recommender"`.
`db_username = "postgres"` was already correct ("root" is invalid in PostgreSQL).

### 3. Terraform — production output.tf ✓
`infra/terraform/aws/environment/production/output.tf` — renamed `rds_mysql_host` → `db_host`.
`infra/scripts/output_redirection/models/production.py` — renamed `RDS_MYSQL_HOST` → `DB_HOST`, removed dead Azure fields.
These three must stay in sync: terraform output name → Pydantic model field → AppSettings env var read.

### 4. Terraform — EC2 instance type ✓
`infra/terraform/aws/environment/production/modules/ec2/main.tf` — already `t3.medium`. No change needed.

### 5. Sync chain ✓
`infra/scripts/output_redirection/models/production.py` rewritten:
```python
class ProductionBackendOutputs(BaseModel):
    SECRETS_MANAGER_DB_CREDENTIALS_KEY: str
    S3_MAIN_BUCKET_NAME: str
    DB_HOST: str
```
The sync script reads terraform outputs, filters by these field names, and appends them to the synced `.env.production` file.

### 6. Backend env example ✓
`backend/env_config/base/.env.production.example` — rewritten for PostgreSQL, Discord webhooks, Firebase individual vars.
`DB_USER` and `DB_PASSWORD` are NOT in this file — AppSettings fetches them from Secrets Manager at startup.
`SECRETS_MANAGER_DB_CREDENTIALS_KEY` and `DB_HOST` are left empty here — injected by sync script.

### 7. Secrets Manager integration ✓ (NEW — was missing)
`backend/src/movie_recommender/services/infra/aws.py` — new module that calls boto3 to fetch DB credentials.
`backend/src/movie_recommender/core/settings/main.py` — `_load_database_settings()` now:
- **dev**: reads `DB_USER` + `DB_PASSWORD` from env
- **production**: calls `fetch_db_credentials(SECRETS_MANAGER_DB_CREDENTIALS_KEY)` → gets `{username, password}` from AWS Secrets Manager

The EC2 has an IAM role (via instance profile) with `secretsmanager:GetSecretValue`. boto3 picks up the role automatically — no AWS credentials in the env file.

`boto3>=1.34.0` added to `backend/pyproject.toml`.

### 8. Docker Compose — base ✓
`deployment/docker-compose.yml`:
- `db-migration` now has `image: javidsegura/movie_recommender:${BACKEND_IMAGE_TAG}` (was missing)
- `prometheus` and `grafana` moved here from dev/remote (they are shared — same config in both environments)
- `prometheus_data` and `grafana_data` volumes added

### 9. Docker Compose — remote ✓
`deployment/docker-compose.remote.yml` rewritten:
- `backend` uses correct image `javidsegura/movie_recommender:${BACKEND_IMAGE_TAG}`
- healthcheck URL fixed to `/api/v1/health/dependencies`
- `depends_on` for db-migration uses `service_completed_successfully`
- `db-migration` has image + env_file + ENVIRONMENT
- `nginx` mounts nginx.remote.conf + frontend dist
- prometheus/grafana removed (now in base)

### 10. nginx.remote.conf ✓
`deployment/reverse-proxy-config/nginx.remote.conf` updated with:
- `limit_req_zone` rate limiting (30r/m, burst 10)
- `client_max_body_size 30m`
- `proxy_read_timeout 300s` / `proxy_send_timeout 300s`
- `proxy_buffering off` (for SSE/streaming)
- SPA fallback: `try_files $uri $uri/ /index.html`

### 11. Ansible playbook ✓
`infra/ansible/playbook/multi-container.yml` fixed:
- `project_name: "movie-recommender"`
- health check URL: `/api/v1/health/dependencies`
- rollback grep: `"movie-recommender"`
- rollback compose file: `docker-compose.remote.yml` (was `docker-compose.{{stage_environment}}.yml`)

### 12. Makefile tasks ✓
Root `Makefile` — added:
- `deploy-start-infra` — full first deploy (terraform + sync + build + push + ansible)
- `deploy-start-artifacts` — re-deploy without infra change
- `deploy-start-ansible` — fastest re-deploy (image already pushed)
- `deploy-stop-infra` — destroy all prod infrastructure
- `check-backend-version` — validates `BACKEND_VERSION` env var

`backend/Makefile` — added `push-docker` (multi-arch buildx for Apple Silicon → EC2 x86).
`frontend/Makefile` — added `build` (`npm run build` → outputs to `frontend/app/dist`).

---

## What's Already Done (don't touch)

- `deployment/telemetry/grafana/provisioning/alerting/alerts.yml` — Discord contact point (`$DISCORD_WEBHOOK_GRAFANA`) ✓
- `backend/src/movie_recommender/services/notifiers/discord.py` — pipeline notifier (`$DISCORD_WEBHOOK_PIPELINE`) ✓
- Local Docker Compose (`docker-compose.dev.yml`) — working, don't change
- VPC, networking, IAM in terraform — generic, reuse as-is
- `infra/scripts/output_redirection/` — sync chain works once model field names are correct ✓
- `infra/scripts/resource_connections/` — SSH key fetcher works as-is
- EC2 IAM role — already has `secretsmanager:GetSecretValue` policy → EC2 can fetch DB credentials without any AWS env vars

---

## How the DB Password Flows (production)

```
Terraform apply
  → creates RDS instance
  → stores {username, password} in Secrets Manager as JSON
  → outputs secret key name as "secrets_manager_db_credentials_key"

make -C infra sync_all
  → sync script reads terraform output
  → appends SECRETS_MANAGER_DB_CREDENTIALS_KEY=<secret-name> to synced .env.production
  → also appends DB_HOST=<rds-endpoint>

Ansible deploys → EC2 runs docker compose
  → backend container starts, AppSettings._load_database_settings() runs
  → reads SECRETS_MANAGER_DB_CREDENTIALS_KEY from env
  → calls boto3 → Secrets Manager (no AWS creds needed, IAM role handles it)
  → gets {username, password}
  → database connection established
```

`make extract-db-credentials` is only for connecting to RDS from your **local machine** (dev debugging). It is NOT part of the production startup flow.

---

## Connection / Debug Commands (run from infra/)

```bash
# SSH into EC2
ENVIRONMENT=production CLOUD_PROVIDER=aws make connection-ssh-web-server

# DB tunnel for local access (local 3307 → RDS 5432)
ENVIRONMENT=production CLOUD_PROVIDER=aws make connection-db

# Grafana tunnel (local 3000 → EC2 3000)
ENVIRONMENT=production CLOUD_PROVIDER=aws make connection-grafana

# HTTP proxy (see nginx traffic)
ENVIRONMENT=production CLOUD_PROVIDER=aws make connection-http-traffic-web-server

# Get DB password locally (for pgAdmin / migrations debugging)
ENVIRONMENT=production CLOUD_PROVIDER=aws make extract-db-credentials
```

`fetch-ssh-key` is called automatically during `make -C infra sync_all` as part of ansible inventory generation — it fetches the EC2 SSH private key from Secrets Manager and caches it at `~/.ssh/aws_production_key.pem`.

---

## Execution Order

```
0. All file changes above are already applied.

1. Check remote state S3 bucket
   - If "ai-ticket-platform-remote-state-bucket-zxmluk37" still exists in AWS:
       Leave main.tf backend block as-is (bucket + key already correct).
   - If gone:
       cd infra/ && ENVIRONMENT=production CLOUD_PROVIDER=aws make start-remote-state
       Update main.tf backend.bucket with the new bucket name.

2. Log into Docker Hub (one-time, cached in ~/.docker/config.json)
   docker login

3. Provision infrastructure + full deploy
   export ENVIRONMENT=production CLOUD_PROVIDER=aws BACKEND_VERSION=v1.0.0
   make prod-launch
   (RDS takes ~5 min — this command blocks until ansible finishes)

   OR step-by-step:
   a. make -C infra terraform-apply
   b. make -C infra sync_all        ← injects DB_HOST + SECRETS_MANAGER_DB_CREDENTIALS_KEY + VITE_BASE_URL
   c. Verify backend/env_config/synced/.env.production has all secrets filled
      (everything should be there — TMDB, Firebase, Discord, Grafana from the base file)
   d. make prod-ship                ← skips terraform, builds + pushes + deploys

4. Verify
   curl http://<EC2_IP>/nginx-health                  → "ok"
   curl http://<EC2_IP>/api/v1/health/dependencies    → 200
   ENVIRONMENT=production CLOUD_PROVIDER=aws make -C infra connection-grafana
   # then open http://localhost:3000

5. Subsequent deploys (no infra change):
   ENVIRONMENT=production CLOUD_PROVIDER=aws BACKEND_VERSION=<tag> make prod-ship

6. Code-only re-deploy (image already pushed):
   ENVIRONMENT=production CLOUD_PROVIDER=aws BACKEND_VERSION=<tag> make prod-rollout
```

---

## Open Questions (decide before step 1)

1. **Remote state bucket** — does `ai-ticket-platform-remote-state-bucket-zxmluk37` still exist?
   Check in AWS console before running `start-remote-state`. If it does, leave the backend config as-is.

2. **Docker Hub image name** — using `javidsegura/movie_recommender` throughout.
   If you change it, update `docker-compose.yml`, `docker-compose.remote.yml`, and `backend/Makefile` together.

3. **NEO4J_PASSWORD** — hardcoded to `dev-password` in `.env.production.example` (intentional, per design decision).
   The neo4j healthcheck and `NEO4J_AUTH` both default to `dev-password` if the variable isn't overridden.
