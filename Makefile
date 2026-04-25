VENV = .venv
UV = uv
MARIMO = $(VENV)/bin/marimo

.PHONY: install marimo verify clean

$(VENV)/bin/python:
	$(UV) venv $(VENV)

install: $(VENV)/bin/python
	$(UV) pip install --python $(VENV)/bin/python -r requirements.txt

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
