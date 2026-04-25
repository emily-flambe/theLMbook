PYTHON ?= python3
VENV = .venv
PIP = $(VENV)/bin/pip
MARIMO = $(VENV)/bin/marimo

.PHONY: install marimo verify clean

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/python
	$(PIP) install -r requirements.txt

marimo:
	$(MARIMO) edit

verify:
	@for f in chapter-*/*.py extras/*.py; do \
		$(MARIMO) export script $$f >/dev/null 2>&1 \
			&& echo "PASS  $$f" \
			|| echo "FAIL  $$f"; \
	done

clean:
	rm -rf $(VENV)
