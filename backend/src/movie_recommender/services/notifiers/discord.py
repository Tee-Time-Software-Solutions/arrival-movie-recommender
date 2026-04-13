import os
import mimetypes
import tempfile
import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parents[4] / "env_config" / "synced" / ".env.dev"
load_dotenv(_ENV_PATH)


class DiscordNotifier:
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK")

    def send_discord_notification(self, msg: str, file_path: str = None):
        if not self.webhook_url:
            return None

        payload = {"content": msg}

        if file_path and os.path.exists(file_path):
            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, mime_type)}
                response = requests.post(self.webhook_url, data=payload, files=files)
        else:
            response = requests.post(self.webhook_url, json=payload)

        return response

    def send_training_report(
        self, model_name: str, metrics: dict, config_meta: dict
    ) -> None:
        """
        Send a training-run summary to Discord.

        Builds a text summary from `metrics` and `config_meta`, generates a bar-chart
        plot of the metric values, and uploads both to the webhook.

        Args:
            model_name:  Human-readable model name, e.g. "ALS" or "LightFM (BPR)".
            metrics:     Dict of metric_name → float, e.g. {"recall@10": 0.23, ...}.
            config_meta: Dict of hyper-parameter name → value to include in the report.
        """
        if not self.webhook_url:
            raise Exception(
                "You need to provide the environmental var: 'DISCORD_WEBHOOK'"
            )

        now = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        k = next((v for k, v in {"k": None}.items()), 10)  # default

        metric_lines = "\n".join(
            f"  **{name}**: `{value:.4f}`" for name, value in metrics.items()
        )
        config_lines = "\n".join(
            f"  `{name}`: {value}" for name, value in config_meta.items()
        )

        msg = (
            f"## Training run complete — {model_name}\n"
            f"**Finished at:** {now}\n\n"
            f"### Metrics\n{metric_lines}\n\n"
            f"### Hyperparameters\n{config_lines}"
        )

        plot_path = self._generate_metrics_plot(model_name, metrics)
        try:
            self.send_discord_notification(msg, plot_path)
        finally:
            if plot_path and os.path.exists(plot_path):
                os.remove(plot_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_metrics_plot(self, model_name: str, metrics: dict) -> str | None:
        """Render a bar chart of `metrics` and return the temp file path."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        names = list(metrics.keys())
        values = [metrics[n] for n in names]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, values, color="#5865F2", width=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylim(0, max(values) * 1.25 if values else 1)
        ax.set_title(
            f"{model_name} — evaluation metrics", fontsize=13, fontweight="bold"
        )
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_{model_name.lower().replace(' ', '_')}_metrics.png",
            delete=False,
        )
        fig.savefig(tmp.name, dpi=150)
        plt.close(fig)
        return tmp.name
