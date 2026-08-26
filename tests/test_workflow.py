from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/main.yml"
BASE_CONFIG = Path(__file__).resolve().parents[1] / "config/base.yaml"


def test_daily_workflow_contract():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "checkout_ref:" in text
    assert "send_email:" in text
    assert "default: false" in text
    assert "inputs.checkout_ref || vars.REF" in text
    assert text.count("cron:") == 1
    assert "cron: '0 22 * * *'" in text
    assert "cancel-in-progress: false" in text
    assert "executor.max_paper_num=20" in text
    assert "email.enabled=${{ github.event_name == 'schedule' || inputs.send_email }}" in text
    assert "source.arxiv.extract_full_text=" not in text
    assert "manifest.enabled=true" in text
    assert "manifest.limit=20" in text
    assert "executor.reranker=" not in text
    assert "actions/upload-artifact@v4" in text
    assert "name: daily-recommendations-${{ github.run_id }}" in text
    assert "retention-days: 30" in text

    config_text = BASE_CONFIG.read_text(encoding="utf-8")
    assert "reranker: local" in config_text
    assert "model: jinaai/jina-embeddings-v5-text-nano-retrieval" in config_text
