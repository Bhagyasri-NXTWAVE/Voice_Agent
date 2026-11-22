FROM python:3.11-slim

WORKDIR /app

ENV MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

RUN pip install --no-cache-dir onnx && \
    python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('/app/models/kokoro-v1.0.onnx', '/app/models/kokoro-q8.onnx', weight_type=QuantType.QUInt8)" && \
    mv /app/models/kokoro-q8.onnx /app/models/kokoro-v1.0.onnx && \
    pip uninstall -y onnx

RUN python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('/app/models/kokoro-v1.0.onnx', '/app/models/kokoro-q8.onnx', weight_type=QuantType.QUInt8)" && \
    mv /app/models/kokoro-q8.onnx /app/models/kokoro-v1.0.onnx

COPY . .

CMD ["python", "main.py", "start"]