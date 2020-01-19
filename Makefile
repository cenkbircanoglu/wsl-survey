dpl ?= deploy.env
include $(dpl)
export $(shell sed 's/=.*//' $(dpl))

VERSION=$(shell ./version.sh)

.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	pip install yapf==0.29.0
	yapf --exclude "*egg*" --recursive --in-place wsl_survey tests detection
	yapf --exclude "*egg*" --recursive --diff wsl_survey tests detection

test: ## run tests quickly with the default Python
	nosetests --with-doctest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source wsl_survey -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/wsl_survey.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ wsl_survey
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

# DOCKER TASKS
# Build the container
docker-build: dist ## Build the container
	docker build -t $(APP_NAME) . -f docker/cpu.Dockerfile
	docker build -t $(APP_NAME)-gpu . -f docker/gpu.Dockerfile

docker-build-nc: ## Build the container without caching
	docker build --no-cache -t $(APP_NAME) . -f docker/cpu.Dockerfile
	docker build --no-cache -t $(APP_NAME)-gpu . -f docker/gpu.Dockerfile

docker-run: ## Run container
	docker run -i -t --rm --name="$(APP_NAME)" $(APP_NAME)

docker-up: docker-build docker-run ## Run container on port configured in `config.env` (Alias to run)

docker-stop: ## Stop and remove a running container
	docker stop $(APP_NAME); docker rm $(APP_NAME)

docker-release: docker-build-nc docker-publish ## Make a release by building and publishing the `{version}` ans `latest` tagged containers to ECR

# Docker publish
docker-publish: docker-publish-latest docker-publish-version ## Publish the `{version}` ans `latest` tagged containers to ECR

docker-publish-latest: docker-tag-latest ## Publish the `latest` taged container to ECR
	@echo 'publish latest to $(DOCKER_REPO)'
	docker push $(DOCKER_REPO)/$(APP_NAME):latest
	@echo 'publish latest to $(DOCKER_REPO)-gpu'
	docker push $(DOCKER_REPO)/$(APP_NAME)-gpu:latest

docker-publish-version: docker-tag-version ## Publish the `{version}` taged container to ECR
	@echo 'publish $(VERSION) to $(DOCKER_REPO)'
	docker push $(DOCKER_REPO)/$(APP_NAME):$(VERSION)
	@echo 'publish $(VERSION) to $(DOCKER_REPO)-gpu'
	docker push $(DOCKER_REPO)/$(APP_NAME)-gpu:$(VERSION)

# Docker tagging
docker-tag: docker-tag-latest docker-tag-version ## Generate container tags for the `{version}` ans `latest` tags

docker-tag-latest: ## Generate container `{version}` tag
	@echo 'create tag latest'
	docker tag $(APP_NAME) $(DOCKER_REPO)/$(APP_NAME):latest
	docker tag $(APP_NAME)-gpu $(DOCKER_REPO)/$(APP_NAME)-gpu:latest

docker-tag-version: ## Generate container `latest` tag
	@echo 'create tag $(VERSION)'
	docker tag $(APP_NAME) $(DOCKER_REPO)/$(APP_NAME):$(VERSION)
	docker tag $(APP_NAME)-gpu $(DOCKER_REPO)/$(APP_NAME)-gpu:$(VERSION)

# HELPERS


version: ## Output the current version
	@echo $(VERSION)
