"""Stock mini-agent model with per-call elapsed times; no extra HTTP calls."""
import json
import os
import time
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel


class TimedModel(LitellmModel):
    def _query(self, messages, **kwargs):
        start = time.monotonic()
        record: dict = {"started_epoch": time.time()}
        try:
            response = super()._query(messages, **kwargs)
            record["usage"] = response.model_dump().get("usage")
            record["finish_reasons"] = [c.finish_reason for c in response.choices]
            return response
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            record["latency_seconds"] = time.monotonic() - start
            path = os.getenv("PAIRED_SWE_REQUEST_LOG")
            if path:
                with open(path, "a") as output:
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    os.fsync(output.fileno())

    def query(self, messages, **kwargs):
        start = time.monotonic()
        try:
            result = super().query(messages, **kwargs)
        except FormatError as exc:
            for message in exc.messages:
                message.setdefault("extra", {})["request_latency_seconds"] = time.monotonic() - start
            raise
        result.setdefault("extra", {})["request_latency_seconds"] = time.monotonic() - start
        return result
