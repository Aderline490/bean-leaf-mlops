"""Simulate many users sending leaf photos to the model.

Example: locust -f load_tests/locustfile.py --host http://127.0.0.1:8080
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
IMAGES = [(path.name, path.read_bytes()) for path in sorted((ROOT / "demo" / "predict").glob("*.jpg"))]
LABELS = {
    "demo_angular_leaf_spot.jpg": "angular_leaf_spot",
    "demo_bean_rust.jpg": "bean_rust",
    "demo_healthy.jpg": "healthy",
}


class LeafUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, content = random.choice(IMAGES)
        with self.client.post(
            "/predict",
            files={"file": (name, content, "image/jpeg")},
            name="POST /predict",
            timeout=60,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif response.json().get("label") != LABELS[name]:
                response.failure("Wrong label")
            else:
                response.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health", timeout=60)
