PYTHON := uv run python
UV := uv
NPM := npm --prefix apps/web

.PHONY: install run-api run-web test lint schema dataset-check split eval calibrate chatcut-dataset augment audio-onnx checks provenance-check up down logs clean \
	service-install service-uninstall service-start service-stop service-restart service-status service-logs \
	service-url service-public-url

install:
	$(UV) sync --extra dev --extra onnx
	$(NPM) install

run-api:
	$(UV) run uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload

run-web:
	$(NPM) run dev

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(NPM) run lint

schema:
	$(PYTHON) scripts/export_schemas.py

dataset-check:
	$(PYTHON) -m dhfp.evaluation.cli dataset-check --index data/indexes/dataset.jsonl

split:
	$(PYTHON) -m dhfp.evaluation.cli split --index data/indexes/dataset.jsonl --output data/indexes/splits.json

eval:
	$(PYTHON) -m dhfp.evaluation.cli eval --index data/indexes/dataset.jsonl --splits data/indexes/splits.json --output data/reports

calibrate:
	$(PYTHON) -m dhfp.evaluation.cli calibrate --report data/reports/stage0_report.json --detector fusion-pipeline --output configs/calibration.json

chatcut-dataset:
	$(PYTHON) scripts/build_chatcut_dataset.py --input data/raw/chatcut --output data/indexes/chatcut.jsonl --generator-version chatcut-v1

augment:
	$(PYTHON) -m dhfp.evaluation.cli augment-reencode --index data/indexes/dataset.jsonl --output-dir data/augmented --output-index data/indexes/dataset_augmented.jsonl --strengths crf23,crf28,crf32

audio-onnx:
	$(PYTHON) scripts/export_audio_spoof_onnx.py --repo third_party/AASIST --output data/models/audio_aasist.onnx

# 用法：make provenance-check FILE=sample.mp4
provenance-check:
	@test -n "$(FILE)" || (echo "用法：make provenance-check FILE=<媒体文件>"; exit 2)
	$(PYTHON) scripts/verify_provenance.py "$(FILE)"

checks: test lint

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api worker web

service-install:
	./scripts/platform-service.sh install

service-uninstall:
	./scripts/platform-service.sh uninstall

service-start:
	./scripts/platform-service.sh start

service-stop:
	./scripts/platform-service.sh stop

service-restart:
	./scripts/platform-service.sh restart

service-status:
	./scripts/platform-service.sh status

service-logs:
	./scripts/platform-service.sh logs api

service-url:
	./scripts/platform-service.sh url

service-public-url:
	./scripts/platform-service.sh public-url

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov apps/web/.next
