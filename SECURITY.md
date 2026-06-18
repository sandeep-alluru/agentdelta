# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report vulnerabilities privately by emailing **onepuncchh@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact

You will receive a response within 48 hours. Once confirmed, a patch will be
issued within 7 days and credited to you in the changelog unless you prefer
to remain anonymous.

## Scope

redline processes local trace files and runs sentence-transformer inference.
It does not make network requests except to download the `all-MiniLM-L6-v2`
model on first use. It does not store or transmit trace data externally.
