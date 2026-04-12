# Deployment Strategy — Movie Recommender (Prod AWS)

> Scope: production-only. This doc covers how to adapt the old `ai-ticket-platform` infra to this repo, what's broken, what needs to change, and what to add (MLflow, Discord alerts, rate limiter, Google Analytics).

---

`
Executive summary:
Critical blockers (will fail on first deploy)
RDS module uses MySQL 8.0 — app is PostgreSQL. Must change engine = "postgres" + version.
docker-compose.remote.yml is entirely commented out with old ai_ticket_platform image names. Full rewrite needed.
docker-compose.yml base has a broken db-migration — no image: or build: defined.
Ansible health check hits /api/health/dependencies — missing v1 prefix, rollback fires every deploy.
Both .env.production.example and .env.staging.example use MySQL vars (MYSQL_*) — app reads DB_* PostgreSQL vars.
High priority (wrong behavior)
Terraform tfvars still says project_name = "ai-ticket-platform" — tags everything wrong, wrong remote state key.
Ansible project_name wrong — deploys to /opt/ai-ticket-platform on the remote.
Grafana alerts point to Slack ($SLACK_WEBHOOK_URL) — just change type: slack → type: discord, rename the env var.
EC2 is t3.small (2 GB RAM) — Neo4j alone wants ~2 GB. Upgrade to t3.medium.
New additions (simple)
MLflow: one service definition, SQLite backend, served via nginx /mlflow/. No app code changes yet.
Nginx remote config: new nginx.remote.conf — serves static dist, proxies /api/, proxies /mlflow/, adds limit_req_zone rate limiter (30 req/min with burst 10).
Google Analytics: two <script> tags in index.html + one VITE_GA_MEASUREMENT_ID env var. Vite replaces at build time.
`


## 1. Architecture Overview (Production)

```
Internet
  └── EC2 (t3.medium) ── nginx:80
        ├── /            → static files from frontend/app/dist
        ├── /api/        → backend:8000
        └── /mlflow/     → mlflow:5001

  EC2 internal services:
    - backend        (FastAPI, pulls image from Docker Hub)
    - redis          (redis:7.2-alpine)
    - neo4j          (neo4j:5-community)
    - mlflow         (ghcr.io/mlflow/mlflow)
    - prometheus     (prom/prometheus)
    - grafana        (grafana/grafana)

  AWS managed:
    - RDS PostgreSQL 16 (private subnet)
    - S3 (optional, for future artifact store)
```

The base `docker-compose.yml` defines all services that exist in every environment. The `docker-compose.remote.yml` overrides those definitions for production: swapping `build:` for `image:` references, pointing nginx to the static dist, and wiring in production env vars.

---

## 2. Issues Found (Ordered by Impact)

### CRITICAL — will break prod

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `infra/terraform/aws/modules/rds/main.tf` | Engine is `mysql 8.0`. App uses PostgreSQL. RDS will provision the wrong database. | Change engine to `postgres`, version `16`, and driver outputs accordingly. |
| 2 | `deployment/docker-compose.remote.yml` | Entirely commented out. Old image `javidsegura/ai_ticket_platform` would fail to pull. | Rewrite it from scratch with correct image names. |
| 3 | `deployment/docker-compose.yml` | `db-migration` has no `build:` or `image:`. Compose would fail to resolve it. | Add `image:` referencing the backend Docker Hub tag. |
| 4 | `infra/ansible/playbook/multi-container.yml` | Health check hits `/api/health/dependencies` — missing `v1` prefix. Rollback triggers on every deploy. | Fix URL to `/api/v1/health/dependencies`. |
| 5 | `backend/env_config/base/.env.staging.example` + `.env.production.example` | Both use MySQL connection vars (`MYSQL_*`). App uses `DB_*` PostgreSQL vars. Running ansible with these will fail at DB connect. | Rewrite both to match `.env.dev.example` structure. |

### HIGH — wrong behavior

| # | File | Problem | Fix |
|---|------|---------|-----|
| 6 | `infra/terraform/aws/environment/production/terraform.tfvars` | `project_name = "ai-ticket-platform"`. Tags all AWS resources incorrectly. S3 remote state bucket hardcoded to old project. | Update tfvars + S3 backend key. Provision new remote-state S3 bucket. |
| 7 | `infra/ansible/playbook/multi-container.yml` | `project_name: "ai-ticket-platform"` used in remote path `/opt/ai-ticket-platform`. Wrong directory. | Change to `"movie-recommender"`. |
| 8 | `deployment/telemetry/grafana/provisioning/alerting/alerts.yml` | Contact point is Slack (`type: slack`, reads `$SLACK_WEBHOOK_URL`). No Discord support. Alerts silently go nowhere if that env var is absent. | Switch to Discord contact point (`type: discord`, `$DISCORD_WEBHOOK_URL`). |
| 9 | `infra/terraform/aws/environment/production/modules/ec2/main.tf` | Instance type `t3.small` (2 vCPU, 2 GB RAM). Neo4j alone recommends ≥2 GB. With backend + redis + neo4j + nginx + prometheus + grafana + mlflow, this will OOM. | Upgrade to `t3.medium` (4 GB). |

### MEDIUM — missing features

| # | What | Note |
|----|------|------|
| 10 | MLflow service | Not defined anywhere in docker-compose. Pipeline runs daily but metrics/artifacts have nowhere to go. |
| 11 | Nginx remote config | Only `nginx.dev.conf` exists. Prod nginx needs to serve static dist + proxy backend + proxy mlflow + rate limit. |
| 12 | Rate limiter | No rate limiting anywhere. Simple nginx `limit_req_zone` on `/api/`. |
| 13 | Google Analytics | Missing `gtag.js` in `frontend/app/index.html`. One script tag + one env var. |
| 14 | Frontend env examples | `frontend/app/env_config/synced/.env.dev.example` has dev vars only. Need a `.env.production.example` with prod base URL + GA ID. |

### LOW — cosmetic / cleanup

| # | File | Note |
|----|------|------|
| 15 | `infra/terraform/aws/modules/s3/main.tf` | Bucket name contains `shorten-url`. Functional but misleading. Can rename. |
| 16 | `deployment/docker-compose.dev.yml` | Image tags `movie_recommender_frontend:dev`, `movie_recommender_backend:dev` — these are correct. No change needed. |
| 17 | `infra/ansible/playbook/multi-container.yml` | Rollback block contains `url-shortener-backend` in `grep` — won't find the right container. |

---

## 3. Change Plan

### 3.1 Terraform (`infra/terraform/aws/`)

**`environment/production/terraform.tfvars`**
```hcl
# Change:
project_name = "movie-recommender"

# The S3 backend bucket in main.tf must also be updated:
# bucket = "movie-recommender-remote-state-<suffix>"
# key    = "remote-state/production/terraform.tfstate"
```
> You'll need to run `make start-remote-state` in `infra/terraform/aws/set-up/` first to provision the new state bucket, then update `main.tf` and `terraform init -reconfigure`.

**`modules/rds/main.tf`** — change the `aws_db_instance` block:
```hcl
engine         = "postgres"
engine_version = "16.3"
# remove: MYSQL-specific options
# username / password flow stays the same
```
The Secrets Manager output shape stays the same (username + password), so `infra/scripts/output_redirection` scripts should need no changes.

**`environment/production/modules/ec2/main.tf`**
```hcl
instance_type = "t3.medium"   # was t3.small
```

**`infra/terraform/aws/modules/s3/main.tf`** *(optional)*
Rename bucket prefix from `shorten-url` to `movie-recommender`. Only do this if you're tearing down and reprovisioning — don't rename in place.

No other Terraform files need changes. VPC, networking, IAM roles, and Secrets Manager are all generic enough to reuse.

---

### 3.2 Docker Compose

#### `deployment/docker-compose.yml` (base — shared across all envs)

Current state: has redis, neo4j, nginx (shell), db-migration (broken — no image/build). Missing: postgres placeholder for dev, prometheus, grafana, mlflow, volumes for them.

Changes:
- **db-migration**: add `image: javidsegura/movie_recommender:${BACKEND_IMAGE_TAG}` so remote deployments resolve it.
- **Uncomment prometheus + grafana** sections (already written, just commented out).
- **Add mlflow** service (new — see section 3.5).
- **Add volumes**: `prometheus_data`, `grafana_data`, `mlflow_data`.
- Postgres stays in `docker-compose.dev.yml` only (prod uses RDS).

#### `deployment/docker-compose.dev.yml` (dev overrides)

Nearly perfect as-is. Only additions:
- Uncomment or add the prometheus + grafana block (optional for dev, but useful for local testing of alerts).
- Add mlflow service with a local sqlite backend.

No breaking changes needed here.

#### `deployment/docker-compose.remote.yml` (production overrides)

Rewrite from scratch. The full structure:

```yaml
services:
  backend:
    image: javidsegura/movie_recommender:${BACKEND_IMAGE_TAG}
    ports:
      - "8000:8000"
    env_file:
      - ${BACKEND_ENV_FILE}
    environment:
      ENVIRONMENT: production
    depends_on:
      - db-migration
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/dependencies"]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 15s

  db-migration:
    image: javidsegura/movie_recommender:${BACKEND_IMAGE_TAG}
    env_file:
      - ${BACKEND_ENV_FILE}
    environment:
      ENVIRONMENT: production

  nginx:
    volumes:
      - ./reverse-proxy-config/nginx.remote.conf:/etc/nginx/conf.d/default.conf:ro
      - ../frontend/app/dist:/usr/share/nginx/html:ro
    depends_on:
      - backend

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./telemetry/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_UNIFIED_ALERTING_ENABLED=true
    volumes:
      - grafana_data:/var/lib/grafana
      - ./telemetry/grafana/provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./telemetry/grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards
      - ./telemetry/grafana/provisioning/alerting:/etc/grafana/provisioning/alerting
    env_file:
      - ${BACKEND_ENV_FILE}
    depends_on:
      - prometheus

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: >
      mlflow server
        --backend-store-uri sqlite:///mlflow/mlflow.db
        --default-artifact-root /mlflow/artifacts
        --host 0.0.0.0
        --port 5001
        --serve-artifacts
    volumes:
      - mlflow_data:/mlflow

volumes:
  prometheus_data:
  grafana_data:
  mlflow_data:
```

> Note: No `database` service in remote — that's RDS. No `frontend` service in remote — nginx serves the built `dist/` directly.

---

### 3.3 Nginx — add `nginx.remote.conf`

Create `deployment/reverse-proxy-config/nginx.remote.conf`. Key additions vs dev config:

```nginx
# Rate limiting zone (define outside server block)
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;

server {
    listen 80;
    server_name _;

    client_max_body_size 30m;

    # Serve React SPA
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Backend API with rate limiting
    location /api/ {
        limit_req zone=api_limit burst=10 nodelay;

        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # MLflow UI (internal access via SSH tunnel; no auth on this endpoint)
    location /mlflow/ {
        proxy_pass http://mlflow:5001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /nginx-health {
        add_header Content-Type text/plain;
        return 200 "ok";
    }
}
```

Rate limit choice: `30r/m` (0.5 req/s) with burst of 10 is conservative and won't affect real users (swipe actions fire ~1/sec max) but blocks bots. Adjust up if needed.

---

### 3.4 Ansible (`infra/ansible/playbook/multi-container.yml`)

Changes needed:

```yaml
# Line 9 — fix project name
project_name: "movie-recommender"

# Line 10 — env path is already correct pattern:
src_backend_env_synced_path: "../../../backend/env_config/synced/.env.{{ stage_environment }}"

# Line 186 — fix health check URL (add v1)
url: "http://{{ ansible_default_ipv4.address | default(ansible_host) }}:8000/api/v1/health/dependencies"

# Line 204 — fix rollback grep (old project name)
shell: docker ps -a | grep "movie-recommender-backend" | head -n 1 | awk '{print $1}'

# Line 169 — fix files list (docker-compose.remote.yml not docker-compose.{{stage_environment}}.yml)
files:
  - docker-compose.yml
  - docker-compose.remote.yml

# Line 227 — same fix in rollback block
files:
  - docker-compose.yml
  - docker-compose.remote.yml
```

Also add a task to create the mlflow data directory on the remote (before docker-compose up):
```yaml
- name: Create mlflow data directory
  ansible.builtin.file:
    path: "{{ remote_project_path }}/mlflow"
    state: directory
    owner: "{{ target_user }}"
    group: docker
    mode: "0755"
```

**No Firebase credentials JSON copy task needed** — the old project uploaded a `firebase-adminsdk.json` file. This project uses individual env vars (`FIREBASE_PROJECT_ID`, `FIREBASE_PRIVATE_KEY`, etc.) in `.env.production`. Nothing to add or remove here — it's already gone.

---

### 3.5 MLflow (simple deployment)

The goal is just tracking runs and metrics from the nightly ALS pipeline. No model registry, no artifact versioning needed for now.

**Service definition** (goes in `docker-compose.yml` base, enabled in remote via remote.yml override or just always-on in base):
- Image: `ghcr.io/mlflow/mlflow:latest`
- Backend store: SQLite at `/mlflow/mlflow.db` (volume-mounted)
- Artifact root: `/mlflow/artifacts` (same volume)
- Port: `5001` internal, not exposed to host (access via nginx `/mlflow/` or SSH tunnel)

**Backend integration** — add to `backend/env_config/base/.env.production.example`:
```
MLFLOW_TRACKING_URI=http://mlflow:5001
MLFLOW_EXPERIMENT_NAME=als_pipeline
```

The ALS pipeline (`services/recommender/pipeline/offline/models/als/main.py`) currently prints metrics to stdout. Adding MLflow logging is a separate task (not part of this deployment doc) — but the server should be ready.

**Access in production**: Use `make connection-port LOCAL_PORT=5001 REMOTE_PORT=5001` to tunnel to Grafana. Or access via nginx `/mlflow/` path if you want it browser-accessible.

---

### 3.6 Grafana → Discord Alerts

Edit `deployment/telemetry/grafana/provisioning/alerting/alerts.yml`:

```yaml
# Change contact point from:
contactPoints:
  - orgId: 1
    name: slack-alerts
    receivers:
      - uid: slack-notifier
        type: slack
        settings:
          url: $SLACK_WEBHOOK_URL

# To:
contactPoints:
  - orgId: 1
    name: discord-alerts
    receivers:
      - uid: discord-notifier
        type: discord
        settings:
          url: $DISCORD_WEBHOOK_URL
          message: |
            **Alert:** {{ .Status }}
            **Summary:** {{ .CommonAnnotations.summary }}
            **Description:** {{ .CommonAnnotations.description }}

policies:
  - orgId: 1
    receiver: discord-alerts   # was: slack-alerts
    ...
```

Grafana's Discord contact point sends to a Discord webhook URL. Format: `https://discord.com/api/webhooks/<id>/<token>`. Get this from your Discord server settings → Integrations → Webhooks.

Add `DISCORD_WEBHOOK_URL` to `.env.production` (synced file, not committed).
Remove `SLACK_WEBHOOK_URL` from all env example files.

---

### 3.7 Env File Updates

#### `backend/env_config/base/.env.production.example`

Complete rewrite to match actual `AppSettings` structure:

```bash
# SET-UP
ENVIRONMENT="production"

# APP LOGIC
BATCH_SIZE=15
QUEUE_MIN_CAPACITY=5
LEARNING_RATE=0.05
NORM_CAP=10.0
OVER_FETCH_FACTOR=2

# REDIS
REDIS_URL="redis://redis:6379"

# NEO4J (running in Docker on EC2)
NEO4J_URI="bolt://neo4j:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD=""
NEO4J_DATABASE="neo4j"

# DB (RDS endpoint — injected by infra/scripts/output_redirection)
DB_HOST=""
DB_PORT="5432"
DB_NAME=""
DB_USER=""
DB_PASSWORD=""
DB_SYNC_DRIVER="+psycopg2"
DB_ASYNC_DRIVER="postgresql+asyncpg"

# TMDB
TMDB_API_KEY=""
TMDB_API_READ_ACCESS_TOKEN=""
TMDB_IMG_URL="https://image.tmdb.org/t/p/w500"
TMDB_BASE_URL="https://api.themoviedb.org/3"

# FIREBASE (individual vars, no JSON file)
FIREBASE_PROJECT_ID=""
FIREBASE_PRIVATE_KEY_ID=""
FIREBASE_PRIVATE_KEY=""
FIREBASE_CLIENT_EMAIL=""
FIREBASE_CLIENT_ID=""
FIREBASE_WEB_API_KEY=""

# GRAFANA
GF_SECURITY_ADMIN_PASSWORD=""

# DISCORD
DISCORD_WEBHOOK_URL=""

# MLFLOW
MLFLOW_TRACKING_URI="http://mlflow:5001"
MLFLOW_EXPERIMENT_NAME="als_pipeline"
```

Delete or overwrite `.env.staging.example` similarly. Staging is out of scope per the brief, so you can leave it as a copy of production.example for now.

#### `frontend/app/env_config/`

Add `base/.env.production.example`:
```bash
VITE_BASE_URL=https://<your-domain>/api/v1
VITE_FIREBASE_API_KEY=
VITE_FIREBASE_AUTH_DOMAIN=
VITE_FIREBASE_PROJECT_ID=
VITE_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

---

### 3.8 Google Analytics

**`frontend/app/index.html`** — add two script tags in `<head>`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=%VITE_GA_MEASUREMENT_ID%"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', '%VITE_GA_MEASUREMENT_ID%');
</script>
```

Vite replaces `%VITE_GA_MEASUREMENT_ID%` at build time from the env file. If the var is empty (dev), the script loads but does nothing — no side effects.

That's it. No React code changes, no store changes.

---

## 4. Execution Order

Work sequentially — each step unblocks the next.

```
Step 1 — Terraform infra
  a. Provision new remote-state S3 bucket (make start-remote-state in infra/)
  b. Update main.tf backend key + tfvars (project_name, RDS engine, instance_type)
  c. terraform init -reconfigure && terraform apply
  d. make sync_all  → writes .env.production synced and updates ansible inventory

Step 2 — Docker Compose
  a. Fix docker-compose.yml (db-migration image, add prometheus/grafana/mlflow, volumes)
  b. Write docker-compose.remote.yml from scratch
  c. Write nginx.remote.conf

Step 3 — Env files
  a. Rewrite .env.production.example (backend)
  b. Write .env.production.example (frontend)
  c. Fill in actual .env.production (synced, not committed) using terraform outputs

Step 4 — Alerting + Observability
  a. Switch Grafana alerts.yml Slack → Discord
  b. Add DISCORD_WEBHOOK_URL to .env.production

Step 5 — Frontend additions
  a. Add Google Analytics script tags to index.html
  b. Build frontend: make -C frontend build (or npm run build in frontend/app/)

Step 6 — Ansible
  a. Fix multi-container.yml (project_name, health check URL, file list, rollback grep)
  b. Verify inventory/production.ini has new EC2 IP (from terraform output)
  c. Run: make -C infra ansible-start ENVIRONMENT=production

Step 7 — Verify
  a. SSH into EC2: all containers running (docker ps)
  b. curl http://<ip>/nginx-health → 200
  c. curl http://<ip>/api/v1/health/dependencies → 200
  d. Open frontend in browser — confirm load + GA network request fires
  e. make -C infra connection-grafana ENVIRONMENT=production → check dashboards
```

---

## 5. What NOT to Change

- `infra/terraform/aws/environment/production/modules/vpc/` — VPC, subnets, IGW, routing are all generic. No changes.
- `infra/terraform/aws/modules/s3/` — used for optional app assets. Not needed right now; module is referenced but output isn't used by the app yet.
- `infra/scripts/` — the `output_redirection` scripts that write terraform outputs into `.env.production` are generic enough. They just need the new RDS output variable names to match (verify after changing engine).
- `deployment/docker-compose.dev.yml` — working locally, leave it alone except optionally adding mlflow.
- `.github/workflows/ci-backend.yml` — CI only runs lint + tests, no deploy. Fine as-is. You may want to add a CD workflow later to build and push the Docker image to Docker Hub.
- `infra/ansible/inventory/production.ini` — updated automatically by `make sync_all`.

---

## 6. Open Questions

1. **Docker Hub image name** — current docker-compose.remote.yml commented section uses `javidsegura/ai_ticket_platform`. What's the correct new image name? Suggest `javidsegura/movie_recommender`. The CI workflow doesn't push images yet — you'll need a CD step or push manually before first Ansible deploy.

2. **Domain / TLS** — current nginx configs only handle HTTP:80. If you have a domain, you'll want Let's Encrypt + certbot or AWS ACM with a load balancer. Out of scope for now but worth noting.

3. **MLflow backend wire-up** — the pipeline cron job runs daily but doesn't log to MLflow yet. That's a separate task: wrap `ALSPipeline.run_pipeline()` in an `mlflow.start_run()` context. The server just needs to be running and reachable.

4. **Neo4j password in prod** — currently `dev-password` in dev. Add `NEO4J_PASSWORD` to `.env.production` as a proper secret.
