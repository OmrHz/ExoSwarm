PYTHON ?= python

.PHONY: install test verify-cache reproduce ui

install:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest -q

verify-cache:
	$(PYTHON) -m exoswarm.cli verify-cache

reproduce: verify-cache test
	$(PYTHON) -m exoswarm.cli reproduce

ui:
	$(PYTHON) -m exoswarm.cli ui
