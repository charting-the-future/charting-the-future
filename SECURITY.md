# Security Policy

## Supported Versions

This repository is provided as a companion to the book  
**Charting the Future: Harnessing LLMs for Quantitative Finance**.

It is intended for **educational and research use**.  
The code and notebooks are reference implementations only, not production systems.

As such:

- Only the **main branch** is maintained by the author.
- There are **no long-term support (LTS) versions**.
- Security updates are not guaranteed.

---

## Reporting a Vulnerability

If you discover a security vulnerability in this repository:

1. **Do NOT open a public Issue or Pull Request.**

2. Use the **“Report a vulnerability”** feature in GitHub’s [Security Advisories](https://github.com/charting-the-future/charting-the-future/security/advisories) section.

   - Reports will remain private between you and the repository maintainer.

3. When submitting a report, please provide as much detail as possible:

   - File or notebook where the issue appears
   - Steps to reproduce
   - Potential impact

4. The author will review and, if relevant, update the repository.  
   Please note that response times may be limited, as this is a book companion project.

---

## Scope of Security Concerns

Because this repository is educational in nature:

- No sensitive or private data is included.
- Dependencies are pinned in `pyproject.toml` and managed with [`uv`](https://docs.astral.sh/uv/).
- Security concerns should focus on:
  - Insecure code examples that could mislead readers if used in practice
  - Unsafe default configurations (e.g., use of hardcoded secrets in demo code)
  - Outdated dependencies with known vulnerabilities

---

## Best Practices for Users

Readers and students are encouraged to:

- Run all examples in **isolated environments** (via `uv run ...`).
- Use **test data only** — do not connect examples to production trading systems or sensitive information.
- Review dependencies regularly with `uv pip audit` or `pip-audit`.

---

## Dependency Security

This repository uses **Dependabot** to help keep dependencies up to date.

- Vulnerabilities in dependencies will be surfaced through Dependabot alerts.
- Updates are applied automatically where appropriate.

---

## Disclaimer

The code in this repository is provided **“as is”**, without warranty of any kind.  
Users are responsible for applying appropriate security practices if adapting the examples for research or production environments.
