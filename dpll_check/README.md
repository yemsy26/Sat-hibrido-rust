# 🧠 Neuro-Symbolic R(5,5) HPC Solver

![Rust](https://img.shields.io/badge/rust-Edition_2021-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)
![TensorRT](https://img.shields.io/badge/TensorRT-11.3-76B900.svg)
![Status](https://img.shields.io/badge/Status-Active_Siege-brightgreen.svg)

## 🌌 La Frontera Matemática: El Problema de Ramsey $R(5,5)$
El número de Ramsey $R(5,5)$ es uno de los mayores enigmas de la combinatoria extrema. Encontrar su límite requiere explorar un espacio topológico tan vasto que supera la capacidad computacional de la fuerza bruta clásica. 
Este proyecto no es un solucionador tradicional; es un **motor híbrido neuro-simbólico de alto rendimiento (HPC)** diseñado para masacrar el árbol de búsqueda de la "Estrella 18" de $R(5,5)$, reduciendo milenios de cálculo a horas gracias a la aceleración por hardware.

## ⚙️ Arquitectura del Motor
El sistema fusiona la rigidez del cálculo determinista con la intuición espacial de la Inteligencia Artificial, operando íntegramente en la VRAM de la GPU para eliminar los cuellos de botella de memoria.

*   **Coordinador en Rust (Orquestador HPC):** Escrito en Rust (Edition 2021), administra el estado global, inyecta la topología base (archivos `.sat`) y ejecuta el asedio en ráfagas adaptativas (`--steps 1000`) para evadir los límites térmicos y los bloqueos del sistema operativo (TDR) en Windows.
*   **Kernel Matemático (C++ / CUDA 12.6):** Ejecuta la validación estricta de grafos y el *backtracking* (retroceso) a nivel de bit. La ingeniería de este puente garantiza **cero derrames de registros (`0 spills`)**, manteniendo cada variable dentro de la caché ultra-rápida de la GPU.
*   **Motor de Inferencia de IA (TensorRT 11.3):** El cerebro del sistema. Una red neuronal de política (`neuro_ramsey_policy.onnx`) reside permanentemente en la VRAM. Utiliza los Tensor Cores de la arquitectura Ada Lovelace (RTX 4060) para evaluar la topología del grafo en tiempo real y dictar qué aristas y colores explorar a continuación, podando ramas infructuosas antes de procesarlas.

## 🚀 Rendimiento y Sincronización Asíncrona
El núcleo de este repositorio es su puente de memoria unificada. El motor CUDA consulta al modelo neuronal de TensorRT de manera asíncrona, logrando tasas de millones de consultas por segundo con **cero fallbacks**. La IA descarta hasta el 20% del árbol combinatorio de una sola pasada, sin interrumpir los ciclos de reloj del procesador (CPU).

## 🛠️ Despliegue y Ejecución
**Requisitos de Hardware:** NVIDIA GPU (sm_89+) con +8GB VRAM.

1. **Configuración del Entorno SDK:**
   ```powershell
   $env:TENSORRT_ROOT = 'C:\SDK\TensorRT-11.3.0.99'
   $env:PATH = "$env:TENSORRT_ROOT\bin;$env:TENSORRT_ROOT\lib;$env:PATH"
   ```

2. **Lanzamiento del Asedio:**
   El script compila automáticamente el puente C++/Rust con el feature `neuro-trt` y despliega el modelo en VRAM.
   ```cmd
   .\iniciar_neuro_ramsey_k43.bat
   ```
