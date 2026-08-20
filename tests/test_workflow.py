from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/main.yml"


def test_daily_workflow_contract():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "checkout_ref:" in text
    assert "inputs.checkout_ref || vars.REF" in text
    assert "cron: '0 22 * * *'" in text
    assert "cron: '0 4 * * *'" in text
    assert "cancel-in-progress: false" in text
    assert "executor.max_paper_num=10" in text
    assert "email.enabled=false" in text
    assert "source.arxiv.extract_full_text=false" in text
    assert "manifest.enabled=true" in text
    assert "actions/upload-artifact@v4" in text
    assert "retention-days: 30" in text
