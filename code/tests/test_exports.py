from pathlib import Path


def test_export_csv_html_pdf_minimal():
    from tools.core.engine import get_engine

    engine = get_engine()
    csv_basic = engine.export.csv()
    assert isinstance(csv_basic, str) and len(csv_basic) > 0

    csv_detailed = engine.export.csv_detailed()
    assert isinstance(csv_detailed, str) and len(csv_detailed) > 0

    html = engine.export.html()
    assert isinstance(html, str) and "<html" in html.lower()

    # PDF may be bytes or empty fallback; accept both but ensure type is bytes/str
    pdf_bytes = engine.export.pdf()
    assert isinstance(pdf_bytes, (bytes, str))

