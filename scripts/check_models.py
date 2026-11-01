"""
Check that the model IDs in config.py exist and answer, before you run the app.

Run from the project folder:
    python scripts/check_models.py

The script checks NVIDIA chat, vision, and embedding models against the
/v1/models catalog, then makes small live requests for each model type.
Rerankers are checked through NVIDIA's separate retrieval/reranking endpoint.
Groq is checked only when its backup key is configured.

Nothing is written to disk. Exit code 0 means every enabled check passed.
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from llm_client import get_embedding_model, _nvidia_chat  # noqa: E402

TIMEOUT = 45


def list_ids(base_url: str, api_key: str) -> set[str]:
    """Return model IDs exposed by a provider's /models endpoint."""
    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return {item["id"] for item in response.json().get("data", []) if "id" in item}


def report(label: str, ok: bool, detail: str) -> bool:
    """Print one result and return its pass/fail state."""
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return ok


def check_nvidia_reranker(model_id: str, api_key: str) -> tuple[bool, str]:
    """
    Call NVIDIA's dedicated reranking endpoint.

    NVIDIA rerankers are not reliably represented in the standard /v1/models
    catalog. The documented endpoint format is:
      https://ai.api.nvidia.com/v1/retrieval/{model_id}/reranking
    """
    endpoint = (
        "https://ai.api.nvidia.com/v1/retrieval/"
        f"{model_id.strip('/')}/reranking"
    )

    payload = {
        "model": model_id,
        "query": {"text": "Which passage explains how photosynthesis works?"},
        "passages": [
            {
                "text": (
                    "Photosynthesis is the process by which green plants use "
                    "sunlight, water, and carbon dioxide to produce sugars "
                    "and release oxygen."
                )
            },
            {
                "text": (
                    "The Roman Empire expanded across parts of Europe, Asia, "
                    "and North Africa."
                )
            },
        ],
        "truncate": "END",
    }

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    body = response.json()

    # NVIDIA's response is a ranked list. Accept common response wrappers,
    # but fail clearly if the service returns an unexpected structure.
    rankings = None
    if isinstance(body, list):
        rankings = body
    elif isinstance(body, dict):
        for key in ("rankings", "results", "data"):
            value = body.get(key)
            if isinstance(value, list):
                rankings = value
                break

    if rankings is None:
        return False, f"Unexpected response format: {str(body)[:250]}"

    if len(rankings) < 2:
        return False, f"Expected 2 ranked passages, received {len(rankings)}"

    return True, f"received {len(rankings)} rankings from {model_id}"


def main() -> int:
    """Run configured model checks and return an appropriate exit code."""
    results = []

    if not config.NVIDIA_API_KEY:
        print("NVIDIA_API_KEY is empty. Add it to .env first.")
        return 1

    # NVIDIA catalog: chat, vision, and embedding only.
    # Rerankers use a separate retrieval endpoint, not this catalog check.
    try:
        nvidia_ids = list_ids(config.NVIDIA_BASE_URL, config.NVIDIA_API_KEY)
    except Exception as error:
        print(f"Could not reach NVIDIA model catalog: {error}")
        return 1

    for label, model_id in [
        ("NVIDIA chat", config.NVIDIA_CHAT_MODEL),
        ("NVIDIA vision", config.NVIDIA_VISION_MODEL),
        ("NVIDIA embedding", config.NVIDIA_EMBEDDING_MODEL),
    ]:
        results.append(
            report(
                f"{label} ID",
                model_id in nvidia_ids,
                model_id
                if model_id in nvidia_ids
                else f"'{model_id}' not in the NVIDIA catalog",
            )
        )

    # NVIDIA chat request.
    try:
        reply = _nvidia_chat(config.NVIDIA_CHAT_MODEL).invoke(
            "Reply with the word OK."
        )
        results.append(
            report("NVIDIA chat call", bool(reply.content), "answered")
        )
    except Exception as error:
        results.append(report("NVIDIA chat call", False, str(error)[:250]))

    # NVIDIA embedding request.
    try:
        vectors = get_embedding_model().embed_query("hello")
        results.append(
            report(
                "NVIDIA embedding call",
                len(vectors) > 0,
                f"{len(vectors)} dimensions",
            )
        )
    except Exception as error:
        results.append(
            report("NVIDIA embedding call", False, str(error)[:250])
        )

    # NVIDIA vision request using a tiny 1x1 PNG.
    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    try:
        vision_reply = _nvidia_chat(config.NVIDIA_VISION_MODEL).invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with one word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": tiny_png},
                        },
                    ],
                }
            ]
        )
        results.append(
            report(
                "NVIDIA vision call",
                bool(vision_reply.content),
                "answered",
            )
        )
    except Exception as error:
        results.append(
            report("NVIDIA vision call", False, str(error)[:250])
        )

    # NVIDIA reranker: test the dedicated retrieval endpoint directly.
    try:
        ok, detail = check_nvidia_reranker(
            config.NVIDIA_RERANK_MODEL,
            config.NVIDIA_API_KEY,
        )
        results.append(report("NVIDIA reranker call", ok, detail))
    except requests.HTTPError as error:
        response = error.response
        status = response.status_code if response is not None else "unknown"
        body = response.text[:300] if response is not None else str(error)
        results.append(
            report(
                "NVIDIA reranker call",
                False,
                f"HTTP {status}: {body}",
            )
        )
    except Exception as error:
        results.append(
            report("NVIDIA reranker call", False, str(error)[:300])
        )

    # Groq backup: check only when a key is configured.
    if config.groq_backup_enabled():
        try:
            groq_ids = list_ids(config.GROQ_BASE_URL, config.GROQ_API_KEY)
            model_id = config.GROQ_CHAT_MODEL
            results.append(
                report(
                    "Groq backup ID",
                    model_id in groq_ids,
                    model_id
                    if model_id in groq_ids
                    else f"'{model_id}' not in the Groq catalog",
                )
            )
        except Exception as error:
            results.append(
                report("Groq backup ID", False, str(error)[:250])
            )
    else:
        print("[SKIP] Groq backup: GROQ_API_KEY not set, backup is off.")

    passed = all(results)
    print(
        "\nAll checks passed."
        if passed
        else "\nSome checks failed. Review the errors above."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
