# ROCm 7.x auf AMD MI50 (gfx906)

## Status: IN PROGRESS

## Ziel
Upgrade von ROCm 6.2.4 auf ROCm 7.x auf einem Host mit AMD Instinct MI50 32GB (gfx906).

## Problem
- AMD hat MI50/gfx906 seit ROCm 6.0 aus der offiziellen Support-Matrix entfernt
- Das Toolchain kann gfx906 weiterhin targeten, wird aber nicht getestet
- Ollama's libggml-hip.so muss nach dem Upgrade neu gebaut werden

## System
- GPU: AMD Instinct MI50 32GB (gfx906:sramecc+:xnack-)
- OS: Ubuntu 24.04
- Vorher: ROCm 6.2.4
- Ziel: ROCm 7.2.0 (neueste stabile Version)

## Risiken
- GPU-Funktionalität könnte nach Upgrade anders sein (nicht getestet von AMD)
- Kernel-Treiber-Kompatibilität
- Ollama muss neu gebaut werden (libggml-hip.so)

## Rollback
Vor dem Upgrade wird der Zustand dokumentiert. Bei Fehlern kann auf ROCm 6.2.4 zurückgesetzt werden.
