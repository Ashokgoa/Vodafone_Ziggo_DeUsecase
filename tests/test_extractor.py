"""Tests for app.ingestion.extractor.

These use small, hand-written fixture HTML rather than the real Ziggo page,
so the tests stay fast and don't break if the live page changes.
"""

from app.ingestion.extractor import extract_text


def test_extract_text_keeps_main_content() -> None:
    html = """
    <html>
      <head><style>body { color: red; }</style></head>
      <body>
        <nav>Home | Internet | TV</nav>
        <header>Ziggo</header>
        <main>
          <h1>Internet</h1>
          <p>Snel en betrouwbaar internet voor thuis.</p>
        </main>
        <footer>Copyright 2026</footer>
        <script>console.log('tracking');</script>
      </body>
    </html>
    """

    text = extract_text(html)

    assert "Internet" in text
    assert "Snel en betrouwbaar internet voor thuis." in text


def test_extract_text_drops_boilerplate_tags() -> None:
    html = """
    <html><body>
      <nav>Home | Internet | TV</nav>
      <header>Ziggo</header>
      <footer>Copyright 2026</footer>
      <script>console.log('tracking');</script>
      <style>body { color: red; }</style>
      <main><p>Real content</p></main>
    </body></html>
    """

    text = extract_text(html)

    assert "Home | Internet | TV" not in text
    assert "Copyright 2026" not in text
    assert "console.log" not in text
    assert "color: red" not in text
    assert "Real content" in text


def test_extract_text_drops_non_script_json_ld() -> None:
    # Regression test: some frameworks (e.g. Vue) render structured data via
    # a non-<script> tag, such as <component :is="'script'" type=
    # "application/ld+json">, which a tag-name-only check would miss.
    html = """
    <html><body>
      <main><p>Real content</p></main>
      <component :is="'script'" type="application/ld+json">
        {"@context": "https://schema.org", "@type": "FAQPage"}
      </component>
    </body></html>
    """

    text = extract_text(html)

    assert "Real content" in text
    assert "schema.org" not in text
    assert "FAQPage" not in text
