.PHONY: info active install clean test lint 

#################################################################################
# GLOBALS                                                                       #
#################################################################################

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

ifeq ($(ENVIRONMENT), "cloud")
    include cfg_cloud.env
    export
else
    include cfg_local.env
    export
endif

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := $(notdir $(ROOT_DIR))
BUCKET := $(S3_BUCKET)
PROFILE := $(ENVIRONMENT)
PYTHON_INTERPRETER := python3.9

#################################################################################
# MAIN COMMANDS                                                                 #
#################################################################################

## Show project info
info:
	@echo "Globals"
	@echo "ROOT DIR \t= $(ROOT_DIR)"
	@echo "PROJECT DIR \t= $(PROJECT_DIR)"
	@echo "PROJECT NAME \t= $(PROJECT_NAME)"
	@echo "S3 BUCKET \t= $(BUCKET)"
	@echo "AWS PROFILE \t= $(PROFILE)"
	@echo "PYTHON VERSION\t= $(PYTHON_INTERPRETER)"
	@echo "\nConfiguration"
	poetry config --list

## Clear screen
cls:
	clear

## Activate virtual environment
active:
	@echo "\nActivate virtual environment"
	poetry shell

## Install Python dependencies
install: active
	@echo "\nInstall Python dependencies"
	poetry install

## Update Python dependencies
update: active install
	@echo "\nUpdate Python dependencies"
	poetry update

## Show Python dependencies
show: cls
	@echo "\nShow Python dependencies"
	poetry show

## Update project to the new version
version: update
	@echo "\nUpdate project to the new version"
	poetry version patch

## Delete cache, temp files, and all compiled Python files
clean:
	@echo "\nDelete cache and all compiled Python files"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name ".pytest_cache" -exec rm -rv {} +
	find "data/temp" -maxdepth 1 -type f ! -name ".gitkeep" -delete

## Run all tests
test: cls
	@echo "\nRun all tests"
	pytest src

## Lint using Flake8
lint: cls
	@echo "\nLint using Flake8"
	flake8 src

## Find security issues with Bandit
security: cls
	@echo "\nFind security issues with Bandit"
	bandit --ini tox.ini

########
# DATA #
########

## Upload new and modified local data to S3
push_data:
	@echo "\nUpload new and modified local data to S3"
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)

## Download new and modified data from S3
pull_data:
	@echo "\nDownload new and modified data from S3"
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)

##########
# MODELS #
##########

## Upload new and modified local models to S3
push_models:
	@echo "\nUpload new and modified local models to S3"
	aws s3 sync models/ s3://$(BUCKET)/models/ --profile $(PROFILE)

## Download new and modified models from S3
pull_models:
	@echo "\nDownload new and modified models from S3"
	aws s3 sync s3://$(BUCKET)/models/ models/ --profile $(PROFILE)

##########
# IMAGES #
##########

## Build SAM docker image
build: cls clean active
	@echo "\nBuild SAM docker image"
	sam validate
	sam build --use-container

## Deploy SAM docker image to AWS
deploy: build
	@echo "\nDeploy docker image to AWS"
	sam deploy --guided --capabilities CAPABILITY_NAMED_IAM --profile $(PROFILE) --parameter-overrides "BucketName=\"$(BUCKET)\"" --stack-name "car-pricing" --config-file "samconfig.toml" --config-env $(PROFILE) --confirm-changeset

## Clean all docker images, containers, and volumes
prune: cls stop
	@echo "\nClean all docker images, containers, and volumes"
	docker system prune --all

## Stop running containers
stop: cls
	@echo "\nStop running containers"
	docker container stop $$(docker container list -qa)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

########
# MAIN #
########

## Start main app
app: active
	@echo "\nStart main app"
	$(PYTHON_INTERPRETER) src/app/api.py

## Start API in development mode
api_dev: active
	@echo "\nStart api dev"
	uvicorn src.app.api:app --reload --host 0.0.0.0 --port 3000

## Clean port with clean_port PORT=xxxx
clean_port:
	@echo "\nCleaning up port $(PORT)"
	@lsof -ti:$(PORT) | xargs kill -9 || true

##############
# PRICING #
##############

## Generate data
data: active
	@echo "\nGenerate data"
	$(PYTHON_INTERPRETER) src/data/dataset.py --download

## Create an initial dataset
dataset: active
	@echo "\nCreate an initial dataset"
	$(PYTHON_INTERPRETER) src/data/dataset.py --process

## Train model
train: active
	@echo "\nTrain model"
	$(PYTHON_INTERPRETER) src/pricing/model.py --train

## Evaluate model
eval: active
	@echo "\nEvaluate model"
	$(PYTHON_INTERPRETER) src/pricing/model.py --eval

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
