default:
	pip install .
	-rm -rf dist build gunpowder.egg-info

.PHONY: install-full
install-full:
	pip install .[full]

.PHONY: install-dev
install-dev:
	pip install -e .[full]

.PHONY: test
test:
	pytest tests -v -m "not slow"

.PHONY: test-all
test-all:
	pytest tests -v
