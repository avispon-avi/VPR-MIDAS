# XIAO ESP32S3 Feasibility Report for VPRS

This document evaluates the practical hardware feasibility of deploying the optimized INT8 MobileNetV3-Small pose regression model directly onto the Seeed Studio XIAO ESP32S3 Sense.

## 1. Hardware Specifications (XIAO ESP32S3 Sense)
- **SoC:** ESP32-S3 (Dual-core Xtensa LX7 @ 240MHz with AI vector instructions)
- **Internal SRAM:** 512 KB
- **External PSRAM:** 8 MB (Octal SPI)
- **Flash Storage:** 8 MB (SPI)

## 2. Model Memory Requirements

### Model Parameter Memory (Flash)
- **Requirement:** ~2.5 MB (Fully static INT8 quantized MobileNetV3-Small weights).
- **Feasibility:** The XIAO has 8 MB of Flash. Factoring in the FreeRTOS firmware, Wi-Fi stack, camera drivers, and OTA partitions (~2.5 MB), there is around 5.5 MB available. **Fits comfortably.**

### Activation Memory & Tensor Arena (SRAM/PSRAM)
- **Input Tensor:** 224 x 224 x 3 (INT8) = 150 KB.
- **Max Intermediate Tensor:** ~800 KB (early convolution layers).
- **Total Tensor Arena Requirement:** ~1.5 MB to 2.0 MB.
- **Feasibility:** The internal SRAM (512 KB) is entirely insufficient to host the Tensor Arena, especially since ~200 KB is already reserved for the Wi-Fi stack and camera DMA buffers. The Tensor Arena **MUST** be allocated in the external 8 MB PSRAM.

## 3. Compute Requirements & Inference Latency

### Approximate MACs
- The PyTorch `mobilenet_v3_small` architecture requires approximately **28 to 56 Million Multiply-Accumulate operations (MMACs)** per inference, depending on exact operator fusion.

### Estimated Inference Latency
- The ESP32-S3 includes vector instructions (ESP32-S3 PIE), which significantly accelerates INT8 operations (supported natively by TFLite Micro + ESP-NN).
- However, because the Tensor Arena is forced into the slower external PSRAM, cache misses will introduce severe bottlenecks.
- **Estimated Latency:** ~400 ms to 800 ms per frame.
- **Estimated FPS:** 1.2 to 2.5 FPS onboard.

## 4. Conclusion

**Verdict:** **PARTIALLY FEASIBLE**

The INT8 quantized VPRS model physically fits on the XIAO ESP32S3 Sense and can technically execute entirely onboard without cloud or PC assistance. 

However, it is strictly classified as *Partially Feasible* because:
1. **Memory bottleneck:** Execution is completely reliant on slower PSRAM.
2. **Latency limitation:** At an estimated 1.2 - 2.5 FPS, the navigation loop will be very slow. While acceptable for hovering or slow, methodical waypoint navigation, it is entirely insufficient for high-speed, agile drone acrobatics. 

*Note: Do not assume this model can run seamlessly onboard until it has been empirically benchmarked on the physical silicon via TensorFlow Lite for Microcontrollers (TFLM).*
