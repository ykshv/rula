"""Validate crawlable HTML, canonical URLs, author identity and sitemap before publication."""

import json
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import yaml

AUTHOR = "https://igor-ya.ru/"


class Document(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.meta: dict[str, str] = {}
        self.links: list[dict[str, str]] = []
        self.anchors: list[str] = []
        self.title = ""
        self.structured: list[str] = []
        self.in_title = False
        self.in_json = False
        self.content_languages: set[str] = set()
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta":
            self.meta[values.get("name", values.get("property", ""))] = values.get("content", "")
        elif tag == "link":
            self.links.append(values)
        elif tag == "a":
            self.anchors.append(values.get("href", ""))
        elif tag == "title":
            self.in_title = True
        elif tag == "script" and values.get("type") == "application/ld+json":
            self.in_json = True
            self.structured.append("")
        if tag == "div" and "lang" in values:
            self.content_languages.add(values["lang"])

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "script":
            self.in_json = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.in_json:
            self.structured[-1] += data


def navigation_files(items):
    for item in items:
        if isinstance(item, str):
            yield item
        else:
            for value in item.values():
                if isinstance(value, list):
                    yield from navigation_files(value)
                else:
                    yield value


def main() -> None:
    config = yaml.safe_load(Path("mkdocs.yml").read_text(encoding="utf-8"))
    base, name = config["site_url"], config["site_name"]
    site = Path(config.get("site_dir", "site"))
    docs = Path(config.get("docs_dir", "docs"))
    expected_urls, titles, descriptions = set(), set(), set()
    for filename in navigation_files(config["nav"]):
        source = (docs / filename).read_text(encoding="utf-8")
        metadata = yaml.safe_load(source.split("---", 2)[1]) if source.startswith("---") else {}
        relative = Path(filename).with_suffix("").as_posix()
        page = "" if relative == "index" else relative + "/"
        expected = base + page
        document = Document((site / page / "index.html").read_text(encoding="utf-8"))
        canonical = [link.get("href") for link in document.links if link.get("rel") == "canonical"]
        assert canonical == [expected], f"Wrong canonical URL: {filename}"
        assert document.title and name in document.title, f"Missing project title: {filename}"
        assert document.title not in titles, f"Duplicate title: {filename}"
        titles.add(document.title)
        description = document.meta.get("description", "")
        assert description and description == metadata["description"], (
            f"Wrong description: {filename}"
        )
        assert description not in descriptions, f"Duplicate description: {filename}"
        descriptions.add(description)
        robots = document.meta.get("robots", "").lower().replace(",", " ").split()
        assert not {"noindex", "none"}.intersection(robots), f"Blocked indexing: {filename}"
        assert document.meta.get("author") == config["site_author"], f"Missing author: {filename}"
        assert document.meta.get("og:url") == expected, f"Wrong Open Graph URL: {filename}"
        assert document.meta.get("og:description") == description, (
            f"Wrong social description: {filename}"
        )
        assert document.meta.get("og:title"), f"Missing social title: {filename}"
        assert document.meta.get("twitter:card") == "summary", f"Missing social card: {filename}"
        assert AUTHOR in document.anchors, f"Missing visible author link: {filename}"
        assert ("ru" if page == "ru/" else "en") in document.content_languages
        if not page:
            data = [json.loads(item) for item in document.structured]
            software = next(item for item in data if item.get("@type") == "SoftwareSourceCode")
            assert software["name"] == name and software["url"] == base
            assert software["codeRepository"] == config["repo_url"]
            assert software["license"] == config["repo_url"] + "/blob/main/LICENSE"
            assert software["author"]["@id"] == AUTHOR + "#person"
            assert software["author"]["url"] == AUTHOR
        expected_urls.add(expected)
    tree = ElementTree.parse(site / "sitemap.xml")
    urls = [node.text for node in tree.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    assert len(urls) == len(set(urls)) and set(urls) == expected_urls, "Sitemap/canonical mismatch"
    print(
        f"Publication contract verified: {len(urls)} canonical URLs, metadata and author identity"
    )


if __name__ == "__main__":
    main()
