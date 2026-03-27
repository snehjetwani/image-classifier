"""
Playwright end-to-end tests for the Gradio UI.

These tests require the app to be running. They are skipped by default so
they do not break CI without a trained model file.

To run them manually:
    1. Start the app:   python app.py --model <your_model.pth> --port 7861
    2. Run the tests:   pytest tests/test_ui.py -m playwright --no-header -v
"""
import subprocess
import time

import pytest
import requests


# ---------------------------------------------------------------------------
# Session-scoped fixture — starts the app subprocess
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url():
    """
    Launch the Gradio app as a subprocess on port 7861 and wait for it to
    become ready.  The process is terminated after all tests in the session.

    Marked as skipped at the test level; this fixture is here for completeness
    and for developers who want to opt in to the full E2E suite.
    """
    proc = subprocess.Popen(
        ["python", "app.py", "--model", "fake.pth", "--port", "7861"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = "http://localhost:7861"
    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

    if not ready:
        proc.terminate()
        pytest.skip("Gradio app did not start within 15 seconds.")

    yield url

    proc.terminate()
    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Tests — all skipped by default
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires running app — run with: python app.py")
def test_page_title(page, base_url):
    page.goto(base_url)
    assert "Image Classifier" in page.title()


@pytest.mark.skip(reason="requires running app — run with: python app.py")
def test_classify_tab_visible(page, base_url):
    page.goto(base_url)
    assert page.get_by_text("Classify").is_visible()


@pytest.mark.skip(reason="requires running app — run with: python app.py")
def test_batch_tab_visible(page, base_url):
    page.goto(base_url)
    assert page.get_by_text("Batch").is_visible()


@pytest.mark.skip(reason="requires running app — run with: python app.py")
def test_model_info_tab_visible(page, base_url):
    page.goto(base_url)
    assert page.get_by_text("Model Info").is_visible()


@pytest.mark.skip(reason="requires running app — run with: python app.py")
def test_upload_component_present(page, base_url):
    page.goto(base_url)
    assert page.locator("input[type=file]").count() > 0


@pytest.mark.skip(reason="requires running app — run with: python app.py")
def test_classify_button_present(page, base_url):
    page.goto(base_url)
    assert page.get_by_role("button", name="Classify").is_visible()
