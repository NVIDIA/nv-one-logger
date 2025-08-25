# MakeFile for E2E Dev
#
# The following ENV variables are used:
# E2E_CI=1 when running as part of a CI pipeline
# GH_TOKEN <user>:<token> for git tagging and creating git release ( mandatory for release )
# E2E_PYTHON_REPO, E2E_PYTHON_REPO_USER, E2E_PYTHON_REPO_PASSWORD for publishing to pypi repo ( mandatory for publish to repo )
#
# make check - For linting, formatting and flake8 checks
# make format - For formatting
# make test - For running all tests
# make build - For bulding cli using nv-release
# make release - For releasing ( GH_TOKEN is required )
# make publish - For publishing to pypi repo ( E2E_PYTHON_REPO, E2E_PYTHON_REPO_USER, E2E_PYTHON_REPO_PASSWORD is required )


CWD := $(realpath .)

PYTHON3 := $(shell which python3)

# Get the Python version
PYTHON_VERSION := $(shell $(PYTHON3) --version | awk '{print $$2}')

TOOL_NAME := one_logger_pytorch_lightning_integration
RELEASE_API_URL := https://gitlab-master.nvidia.com/api/v4/projects/108141/releases
GIT_URL := https://gitlab-master.nvidia.com/hwinf-dcm/nv-one-logger.git
CODE_DIR := $(CWD)/src
TEST_DIR := $(CWD)/tests

TOOL_NAME_UC := $(shell echo $(TOOL_NAME) | tr  '[:lower:]' '[:upper:]')
TOOL_VERSION_ENV := $(TOOL_NAME_UC)_VERSION
PYTHON3_DIR := $(abspath $(PYTHON3)/..)
POETRY_BIN := $(shell which poetry)
GIT_URL_TOKEN := $(subst https://,https://${GH_TOKEN}@,$(GIT_URL))
CURRENT_USER :=  $(subst $\",,$(shell id -un))
GH_TOKEN_ONLY := $(word 2, $(subst :, ,${GH_TOKEN}))
PYPROJECT_VER := $(shell $(POETRY_BIN) version -s)
RELEASE_DATA = $(shell $(shell which jq) -R -s '{"name": "Release $(NEXT_GIT_TAG_VER)", "description":., "tag_name": "v$(NEXT_GIT_TAG_VER)"}' <CHANGELOG.md)
RELEASE_URL := $(subst .git,/-/releases,$(GIT_URL))
NVRELEASE_OUT := $(CWD)/release
CACHE_DIR := $(CWD)/.cache

ifeq (${E2E_CI},1)
	SET_LOCAL_POETRY_CONFIG := $(POETRY_BIN) config installer.parallel false --local
else
    SET_LOCAL_POETRY_CONFIG := :
endif

ifdef ${TOOL_VERSION_ENV}
	NEXT_GIT_TAG_VER := $(subst v,,${$(TOOL_VERSION_ENV)})
else
	NEXT_GIT_TAG_VER := $(shell if [ -x "$(CWD)/.venv/bin/python-semantic-release" ]; then "$(CWD)/.venv/bin/python-semantic-release" version --print; else echo "0.0.0"; fi)
endif


.PHONY: check
check: venv-test
	@echo "---------------------checking (poetry venv with local wheels)--------------------------";
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

.PHONY: venv-test
venv-test:
	@echo "---------------------preparing poetry venv with local core+tt wheels--------------------------";
	$(POETRY_BIN) env use $(PYTHON3)
	@echo "---------------------installing poetry-plugin-export--------------------------";
	$(POETRY_BIN) run pip install poetry-plugin-export
	@echo "---------------------building local wheels (core, training_telemetry)--------------------------";
	$(POETRY_BIN) -C $(CWD)/../one_logger_core build
	$(POETRY_BIN) -C $(CWD)/../one_logger_training_telemetry build
	@echo "---------------------exporting requirements (with dev), filtering out internal deps--------------------------";
	$(POETRY_BIN) export -f requirements.txt --with dev --without-hashes -o /tmp/req.txt
	grep -Ev '^(nv-one-logger-core|nv-one-logger-training-telemetry)(\b|[=<>])' /tmp/req.txt > /tmp/req.filtered.txt || true
	@echo "---------------------installing deps into poetry venv via pip--------------------------";
	$(POETRY_BIN) run python -m pip install --upgrade pip setuptools wheel
	$(POETRY_BIN) run pip install -r /tmp/req.filtered.txt
	$(POETRY_BIN) run pip install --no-deps $(CWD)/../one_logger_core/dist/nv_one_logger_core-*.whl $(CWD)/../one_logger_training_telemetry/dist/nv_one_logger_training_telemetry-*.whl
	$(POETRY_BIN) run pip install .


# Clean up cache and venv
.PHONY: clean
clean:
	@echo "---------------------Cleaning--------------------------";
	rm -rf $(CWD)/.cache $(CWD)/.venv $(CWD)/.coverage $(CWD)/.mypy_cache $(CWD)/.pytest_cache $(CWD)/htmlcov

.PHONY: test
test: unit

.PHONY: unit
unit: clean venv-test
	@echo "---------------------Running unit tests (poetry venv with local wheels)------------------------"
	$(POETRY_BIN) run coverage run --branch  -m pytest
	$(POETRY_BIN) run coverage report -m

.PHONY: publish
publish:
	@echo ---------------------publishing package------------------------
	$(POETRY_BIN) config repositories.pypi ${E2E_PYTHON_REPO}
	$(POETRY_BIN) config http-basic.pypi ${E2E_PYTHON_REPO_USER} ${E2E_PYTHON_REPO_PASSWORD}
	$(POETRY_BIN) publish --build --repository pypi

.PHONY: release_version
release_version: venv 
	@echo "------Next version is : $(NEXT_GIT_TAG_VER)-----------";

.PHONY: tag
# Run via args, example make version=v2.1.0 tag
tag: venv
	@echo "---------------------Tagging in VCS--------------------------";
	@echo -n "Next release version is $(NEXT_GIT_TAG_VER), If you do not agree, select 'n' to exit, and then set environment variable $(TOOL_VERSION_ENV) and run again, do you want to proceed  with release ? [y/N] " && read ans && if [ $${ans:-'N'} = 'y' ]; then \
			echo ---------------------pushing tag $(NEXT_GIT_TAG_VER)------------------------; \
			git tag -f v$(NEXT_GIT_TAG_VER); \
			git push $(GIT_URL_TOKEN) --tags; \
    else \
        exit 1 ; \
    fi \

.PHONY: changelog
changelog:  venv
	@echo ---------------------Generating Changelog-------------------------
	$(CWD)/.venv/bin/semantic-release changelog

.PHONY: poetry_version
poetry_version: 
	@echo ---------------------Updating version in pyptoject.toml-------------------------
	$(POETRY_BIN) version $(NEXT_GIT_TAG_VER)

.PHONY: release_git
release_git:
	@echo ---------------------Creating release and publishing in git-------------------------
	curl --header 'Content-Type: application/json' --request POST --data '$(RELEASE_DATA)' --header "PRIVATE-TOKEN:$(GH_TOKEN_ONLY)" "$(RELEASE_API_URL)"
	@echo ------------------------------------------------------------------
	@echo -n ---Release is created in : $(RELEASE_URL)-----

.PHONY: release
# Run via args, example make version=v2.1.0 release
release: tag changelog poetry_version release_git
	@echo -n ---------------------Release complete-------------------------
	@echo -n ---------------------Optionally submit pyproject.toml and CHANGELOG.md to VCS If needed-------------------------


# ------------------------- CI/CD Required commands -------------------------
.PHONY: ci_tag
# Run via args, example make version=v2.1.0 tag
ci_tag: venv
	@echo -n ---------------------Tagging v$(PYPROJECT_VER)--------------------------
	git tag -f v$(PYPROJECT_VER)
	git push $(GIT_URL_TOKEN) --tags

.PHONY: ci_release_git
ci_release_git:
	@echo -n ---------------------Creating release and publishing in git-------------------------
	curl --header 'Content-Type: application/json' --request POST --data '$(RELEASE_DATA)' --header "JOB-TOKEN:$(GH_TOKEN_ONLY)" "$(RELEASE_API_URL)"
	@echo -n ------------------------------------------------------------------ 
	@echo -n ---Release is created in : $(RELEASE_URL)----- 

.PHONY: ci_release
# Run via args, example make version=v2.1.0 release
ci_release: ci_tag changelog ci_release_git
	@echo -n ---------------------Release complete-------------------------
	
