# Security, Privacy & Air-Gap Compliance Specification

**Project:** Micro-Quantized CNN for Onboard Crater Detection — Chandrayaan-Inspired Lunar Research  
**Document:** Security, Privacy, and Aerospace Compliance Policy  

---

## 1. Aerospace Air-Gap & Offline Execution Compliance

In aerospace defense and planetary exploration programs (such as ISRO, NASA, and ESA), mission-critical software must operate inside strict **air-gapped environments** without dependence on external internet connections or commercial cloud services.

### Implemented Controls:
1. **100% Local Inference Execution:** Model inference via `CraterDetector` runs entirely within local host process memory using ONNX Runtime CPU. Zero inference data, images, or predictions are transmitted to external servers.
2. **Deactivated Commercial Telemetry:** Streamlit usage statistics and telemetry tracking are explicitly disabled in `.streamlit/config.toml` (`gatherUsageStats = false`).
3. **Local Self-Contained Storage:** The experiment registry operates out of a local SQLite database (`artifacts/experiments.db`), requiring no external database daemons or cloud endpoints.

---

## 2. Credential Protection & Secret Hygiene

1. **Zero Hardcoded Secrets:** Static codebase scanning verifies that zero API keys, passwords, cloud access tokens, or private certificates exist in the repository.
2. **Environment Variable Isolation:** If optional PostgreSQL persistence is utilized, connection credentials are passed exclusively via the `DATABASE_URL` environment variable.
3. **Log Sanitization:** In `src/storage/database.py`, connection error handlers mask connection strings to prevent database credentials from leaking to application logs or console streams.
4. **Git Exposure Prevention:** `.gitignore` enforces exclusion of `.env`, `.venv/`, checkpoints (`*.pth`), database files (`*.db`), temporary files, and debug logs.

---

## 3. Data Integrity & Supply Chain Security

1. **Cryptographic Signatures:** Every downloaded lunar mosaic image and annotation file is indexed with a SHA-256 hash recorded in `data/splits/split_manifest.json`. During data loading, checksums can be recomputed to verify that dataset tiles have not been altered or corrupted.
2. **Atomic File Serialization:** Checkpoints are written to temporary files before being renamed atomically, preventing partial or corrupted model states from persisting if the process is terminated abruptly.
3. **Dependency Pinning:** Dependencies are recorded in `requirements.txt` with tested package versions, preventing supply-chain poisoning from unpinned upstream packages.
