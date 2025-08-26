# MakeFile for E2E Dev
#
# The following ENV variables are used:
# E2E_PYTHON_REPO, E2E_PYTHON_REPO_USER, E2E_PYTHON_REPO_PASSWORD for publishing to pypi repo ( mandatory for publish to repo )
#
# make check - For linting, formatting and flake8 checks
# make format - For formatting
# make test - For running all tests
# make publish - For publishing to pypi repo ( E2E_PYTHON_REPO, E2E_PYTHON_REPO_USER, E2E_PYTHON_REPO_PASSWORD is required )


CWD := $(realpath .)

PYTHON3 := $(shell which python3)

# Get the Python version
PYTHON_VERSION := $(shell $(PYTHON3) --version | awk '{print $$2}')

GIT_URL := https://gitlab-master.nvidia.com/hwinf-dcm/nv-one-logger.git
CODE_DIR := $(CWD)/src
TEST_DIR := $(CWD)/tests

PYTHON3_DIR := $(abspath $(PYTHON3)/..)
POETRY_BIN := $(shell which poetry)
GIT_URL_TOKEN := $(subst https://,https://${CI_REGISTRY_USER}:${CI_JOB_TOKEN}@,$(GIT_URL))
CURRENT_USER :=  $(subst $\",,$(shell id -un))
PYPROJECT_VER := $(shell $(POETRY_BIN) version -s)

.PHONY: check
check: venv
	@echo "---------------------checking--------------------------";
	$(POETRY_BIN) run black --check $(CODE_DIR) $(TEST_DIR)
	$(POETRY_BIN) run isort --check-only $(CODE_DIR) $(TEST_DIR)
	$(POETRY_BIN) run flake8 $(CODE_DIR) $(TEST_DIR)

.PHONY: format
format: venv
	@echo "---------------------formatting--------------------------";
	$(POETRY_BIN) run black  $(CODE_DIR) $(TEST_DIR)
	$(POETRY_BIN) run isort  $(CODE_DIR) $(TEST_DIR)

# Setting up venv using poetry install
.PHONY: venv
venv:
	@echo "---------------------creating venv--------------------------";
	@echo ---------------------poetry install------------------------
	$(POETRY_BIN) install  
	@echo ---------------------poetry install end-----------------------

# Clean up cache and venv
.PHONY: clean
clean:
	@echo "---------------------Cleaning--------------------------";
	rm -rf $(CWD)/.cache $(CWD)/.venv $(CWD)/.coverage $(CWD)/.mypy_cache $(CWD)/.pytest_cache $(CWD)/htmlcov

.PHONY: test
test: unit

.PHONY: unit
unit: clean venv
	@echo ---------------------Running unit tests------------------------
	$(POETRY_BIN) run coverage run --branch  -m pytest
	$(POETRY_BIN) run coverage report -m

.PHONY: publish
publish:
	@echo ---------------------publishing package------------------------
	$(POETRY_BIN) config repositories.pypi ${E2E_PYTHON_REPO}
	$(POETRY_BIN) config http-basic.pypi ${E2E_PYTHON_REPO_USER} ${E2E_PYTHON_REPO_PASSWORD}
	$(POETRY_BIN) publish --build --repository pypi

# ------------------------- CI/CD Required commands -------------------------
.PHONY: ci_tag
ci_tag:
	@echo "---------------------setting git user and email--------------------------";
	@git config user.name "MLWFO GitLab CI"
	@git config user.email "hwinf-mlwfo-e2e-dev@nvidia.com"
	@echo -n ---------------------Tagging v$(PYPROJECT_VER)--------------------------
	git tag -a v$(PYPROJECT_VER) -m "Release v$(PYPROJECT_VER)"
	git push $(GIT_URL_TOKEN) v$(PYPROJECT_VER)
	
