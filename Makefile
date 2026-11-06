# Every target delegates to scripts/run_app.py, which also works where make is not installed.
PYTHON ?= python
INPUT ?=

.PHONY: app demo test

app:
	$(PYTHON) scripts/run_app.py app

demo:
	$(PYTHON) scripts/run_app.py demo $(if $(INPUT),--input "$(INPUT)")

test:
	$(PYTHON) scripts/run_app.py test
