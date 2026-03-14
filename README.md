# ROCm 7.2 auf AMD Instinct MI50 (gfx906)

> Ollama + vLLM mit voller GPU-Beschleunigung auf einer GPU, die AMD offiziell nicht mehr unterstützt.

## Überblick

AMD hat die MI50 (gfx906) seit ROCm 6.0 aus der offiziellen Support-Matrix entfernt. Mit ROCm 7.2 fehlen zudem die rocBLAS-Tensile-Kernel für gfx906. Dieses Repository enthält alle Patches und Konfigurationen, um **Ollama** und **vLLM** trotzdem mit voller HIP/ROCm-GPU-Beschleunigung auf der MI50 zu betreiben.

### Getestete Konfiguration

| Komponente    | Version / Info                              |
|---------------|---------------------------------------------|
| GPU           | AMD Instinct MI50 32 GB (gfx906:sramecc+:xnack-) |
| OS            | Ubuntu 24.04, Kernel 6.17.0-14-generic      |
| ROCm          | 7.2.0                                       |
| Ollama        | 0.17.7                                      |
| vLLM          | 0.9.1 (ROCm)                                |
| Docker        | 29.2.1, Compose v2.40.3                     |

### Getestete Modelle

| Modell                | Typ          | Prompt (tok/s) | Generierung (tok/s) | Status |
|-----------------------|-------------|---------------:|--------------------:|--------|
| llama3.2 (3B)         | Dense        |          403+  |              264+   | ✅     |
| qwen3.5:35b (MoE)    | MoE (Q4_K_M) |           20   |               44    | ✅     |
| qwen3:30b             | Dense        |            –   |                –    | ✅     |
| gpt-oss:20b           | Dense        |            –   |                –    | ✅     |
| llama3.1:8b           | Dense        |            –   |                –    | ✅     |

---

## Schnellstart

### Option A: Docker Compose (empfohlen)

```bash
git clone https://github.com/noshitcoding/rocm7-mi50.git
cd rocm7-mi50

# Ollama starten (Port 11434)
docker compose up -d ollama

# Modell laden (wird beim ersten Aufruf heruntergeladen)
docker exec ollama-mi50 ollama pull llama3.2
docker exec ollama-mi50 ollama run llama3.2 "Hallo!"

# vLLM starten (Port 8000) — GGUF-Modelle in ./models/ ablegen
docker compose --profile vllm up -d
```

### Option B: Native Installation (ohne Docker)

```bash
git clone https://github.com/noshitcoding/rocm7-mi50.git
cd rocm7-mi50

# Alles in einem Schritt: Tensile-Symlinks, Build, Installation, Neustart
sudo ./install-native.sh

# Test
ollama run llama3.2 "Hallo!"
```

### Option C: Nur den Patch manuell anwenden

```bash
# Ollama-Quellcode klonen
git clone --depth 1 --branch v0.17.7 https://github.com/ollama/ollama.git
cd ollama

# Patch anwenden
git apply /path/to/patches/gfx906-rocm72-all.diff

# Bauen
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS="gfx906" \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_HIP_NO_MMQ_MFMA=ON \
    -DCMAKE_PREFIX_PATH="/opt/rocm"
cmake --build build --target ggml-hip -j$(nproc)

# Installieren
sudo cp build/lib/ollama/libggml-hip.so /usr/local/lib/ollama/rocm/
sudo systemctl restart ollama
```

---

## Was die Patches machen

### Problem

ROCm 7.2 enthält keine rocBLAS-Tensile-Kernel für gfx906. Jeder Aufruf von `rocblas_initialize()` oder cuBLAS/hipBLAS-Funktionen (die intern Tensile verwenden) führt zu:
- **SIGABRT** bei `rocblas_initialize()`
- **"invalid device function"** bei `cublasStrsmBatched`, `cublasSgemm`

### Lösung: 4 Patches in 4 Dateien

Der Patch besteht aus Änderungen an 4 CUDA/HIP-Quelldateien. Alle Änderungen werden durch das Compile-Flag `GGML_CUDA_FORCE_MMQ` aktiviert und beeinflussen den normalen Code-Pfad nicht.

#### 1. `ggml-cuda.cu` — cuBLAS-Umgehung (5 Änderungen)

| Stelle | Was | Warum |
|--------|-----|-------|
| `ggml_cuda_init()` | `rocblas_initialize()` übersprungen | SIGABRT auf gfx906 |
| `ggml_cuda_mul_mat()` | Batched-cuBLAS-Pfad deaktiviert (`#ifndef GGML_CUDA_FORCE_MMQ`) | Verhindert Tensile-Aufrufe |
| `ggml_cuda_mul_mat()` | Else-Fallback → `ggml_cuda_op_mul_mat_vec_f` statt `cublas` | Alternativer Rechenweg |
| `ggml_cuda_mul_mat_id()` | `type_src1_sorted` erzwungen auf `GGML_TYPE_F32` | MoE-Expert-Dispatch sendet F16 → Assertion |
| `ggml_backend_cuda_graph_reserve()` | `cublas_handle()` Vorab-Erzeugung deaktiviert | Verhindert Tensile-init bei Graph-Capture |

#### 2. `mmvf.cu` — Vektor-Matmul-Kernel (2 Änderungen)

| Stelle | Was | Warum |
|--------|-----|-------|
| `ggml_cuda_op_mul_mat_vec_f()` | Dynamische F16/BF16→F32 Konvertierung von `src1` | Attention Q×K^T sendet F16, Kernel erwartet F32 |
| `ggml_cuda_op_mul_mat_vec_f()` | Chunk-Schleife für `ncols > 8` | Der Vec-Kernel unterstützt max. 8 Spalten |

#### 3. `solve_tri.cu` — Dreiecksgleichungslöser (1 Änderung)

| Stelle | Was | Warum |
|--------|-----|-------|
| `ggml_cuda_op_solve_tri()` | Naiver GPU-Kernel statt `cublasStrsmBatched` | Tensile-Kernel für gfx906 nicht vorhanden |

Der naive Kernel nutzt Forward-Substitution mit einem Thread pro Spalte. Langsamer als Tensile, aber funktional korrekt auf gfx906.

#### 4. `out-prod.cu` — Äußeres Produkt (1 Änderung)

| Stelle | Was | Warum |
|--------|-----|-------|
| `ggml_cuda_out_prod()` | Naiver sgemm-Kernel statt `cublasSgemm` | Tensile-Kernel für gfx906 nicht vorhanden |

16×16 Thread-Block naiver Matmul. Wird nur für Flash-Attention-Gradienten verwendet (selten aufgerufen).

---

## Tensile-Symlinks

Zusätzlich zu den Patches werden Symlinks von `gfx908` → `gfx906` Tensile-Bibliotheken erstellt. Dies ist nötig, damit die wenigen rocBLAS-Aufrufe, die noch durchkommen (z.B. bei `hipblasCreate`), die passenden Kernel finden.

```bash
# Manuell erstellen:
sudo ./scripts/create-tensile-symlinks.sh

# Oder automatisch im Docker-Build/install-native.sh
```

Dies erstellt ~172 Symlinks in `/opt/rocm/lib/rocblas/library/`.

---

## Projektstruktur

```
rocm7-mi50/
├── README.md                          ← Diese Datei
├── docker-compose.yml                 ← Ollama + vLLM Container-Setup
├── build.sh                           ← Native Build-Skript
├── install-native.sh                  ← Alles-in-einem Installations-Skript
├── .gitignore
├── patches/
│   └── gfx906-rocm72-all.diff        ← Alle Source-Patches (4 Dateien)
├── ollama/
│   ├── Dockerfile                     ← Multi-Stage Build: ROCm 7.2 + Ollama
│   └── override.conf                  ← Systemd-Override für native Installation
├── vllm/
│   └── Dockerfile                     ← vLLM mit ROCm, gfx906 Target
├── models/
│   └── README.md                      ← GGUF-Modelle für vLLM hier ablegen
└── scripts/
    └── create-tensile-symlinks.sh     ← gfx908→gfx906 Tensile-Symlinks
```

---

## Docker Compose Dienste

### Ollama (Port 11434)

```bash
# Starten
docker compose up -d ollama

# Modelle ziehen/nutzen
docker exec ollama-mi50 ollama pull qwen3.5:35b
docker exec ollama-mi50 ollama run qwen3.5:35b "Erkläre Quantenmechanik."

# API-Zugriff
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.2","prompt":"Hallo!","stream":false}'

# Modelle auflisten
curl http://localhost:11434/api/tags | python3 -m json.tool
```

**Volumes:** Das Ollama-Modellverzeichnis wird vom Host gemountet (`/usr/share/ollama/.ollama/models`). Alle bereits heruntergeladenen Modelle sind sofort verfügbar.

### vLLM (Port 8000)

```bash
# GGUF-Modell in ./models/ ablegen
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf \
  --local-dir ./models

# vLLM starten
docker compose --profile vllm up -d

# OpenAI-kompatible API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/llama-2-7b.Q4_K_M.gguf","prompt":"Hello","max_tokens":50}'
```

**Hinweis:** vLLM auf MI50/gfx906 ist experimentell. Die offiziellen vLLM-ROCm-Images sind für MI200/MI300 gebaut. Der hier enthaltene Dockerfile baut vLLM from source mit `PYTORCH_ROCM_ARCH=gfx906`.

---

## Systemd-Override (native Installation)

Für die native Installation (ohne Docker) wird ein systemd-Override verwendet:

```bash
sudo cp ollama/override.conf /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Wichtige Environment-Variablen:

| Variable | Wert | Zweck |
|----------|------|-------|
| `OLLAMA_LLM_LIBRARY` | `rocm` | ROCm-Backend erzwingen |
| `GGML_CUDA_INIT` | `1` | Aktiviert den gfx906-Patch-Pfad |
| `OLLAMA_FLASH_ATTENTION` | `1` | Flash-Attention aktivieren |
| `GPU_MAX_HW_QUEUES` | `2` | Stabilität auf GCN-GPUs |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | MI50 hat 32GB VRAM |

---

## CMake-Build-Flags

| Flag | Wert | Zweck |
|------|------|-------|
| `AMDGPU_TARGETS` | `gfx906` | Nur für MI50 kompilieren |
| `GGML_CUDA_FORCE_MMQ` | `ON` | Alle cuBLAS/rocBLAS-Umgehungen aktivieren |
| `GGML_HIP_NO_MMQ_MFMA` | `ON` | MFMA-Instruktionen deaktivieren (gfx906 hat keine) |
| `CMAKE_PREFIX_PATH` | `/opt/rocm` | ROCm-Installation finden |

---

## Fehlerbehebung

### SIGABRT bei Modell-Laden

```
signal: aborted (core dumped)
```

→ `GGML_CUDA_INIT=1` ist nicht gesetzt, oder die ungepatchte `libggml-hip.so` wird geladen. Prüfen:

```bash
# Welche Library wird geladen?
ldd /usr/local/lib/ollama/rocm/libggml-hip.so

# Override prüfen
cat /etc/systemd/system/ollama.service.d/override.conf
```

### "invalid device function"

→ Tensile-Symlinks fehlen:

```bash
sudo ./scripts/create-tensile-symlinks.sh
```

### Modell lädt nicht (OOM)

→ MI50 hat 32 GB VRAM. Große Modelle (>30B Dense) passen möglicherweise nicht:

```bash
# VRAM-Nutzung prüfen
rocm-smi --showmeminfo vram

# Kontext reduzieren
OLLAMA_CONTEXT_LENGTH=2048 ollama run qwen3:32b "test"
```

### GPU wird nicht erkannt

```bash
# ROCm-Installation prüfen
rocm-smi
/opt/rocm/bin/rocminfo | grep gfx

# Benutzer in video/render-Gruppe?
groups $(whoami)
sudo usermod -aG video,render $(whoami)
```

---

## Technische Details

### gfx906 Architektur-Eigenschaften

| Eigenschaft | Wert |
|-------------|------|
| Architektur | GCN 5.1 (Vega 20) |
| GGML CC     | `GGML_CUDA_CC_VEGA20` |
| ISA-Klasse  | `IS_GCN` (nicht CDNA) |
| MFMA        | ❌ Nicht verfügbar |
| WMMA        | ❌ Nicht verfügbar |
| FP16 MMA    | ❌ (Hardware) |
| Fast FP16   | ✅ Verfügbar |
| VRAM        | 32 GB HBM2 |
| Bandbreite  | 1024 GB/s |

### Warum diese Patches nötig sind

1. **ROCm 7.2** hat gfx906 aus den rocBLAS-Tensile-Kerneln entfernt
2. **Ollama/ggml** verwendet cuBLAS/hipBLAS für bestimmte Operationen (batched matmul, TRSM, outer product)
3. Diese Operationen rufen intern **Tensile** auf → **"invalid device function"** auf gfx906
4. Die Patches ersetzen alle Tensile-abhängigen Codepfade durch native HIP-Kernel
5. Der Build-Flag `GGML_CUDA_FORCE_MMQ` steuert die Aktivierung — normaler Code bleibt unverändert

---

## Lizenz

Die Patches basieren auf [Ollama](https://github.com/ollama/ollama) (MIT License).
Dieses Repository: MIT License.

---

## Verwandte Projekte

- [noshitcoding/ollama-rocm-mi50](https://github.com/noshitcoding/ollama-rocm-mi50) — ROCm 6.2.4 Lösung (Vorgänger)
- [Ollama](https://github.com/ollama/ollama)
- [vLLM](https://github.com/vllm-project/vllm)
