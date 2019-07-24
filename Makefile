include make_env

VERSION ?= latest
CONTAINER_NAME ?= docker_ds_test
CONTAINER_INSTANCE ?= default

.PHONY: run
build:
	docker build -f Dockerfile -t $(NS)/$(IMAGE_NAME):$(VERSION) context
run:
	docker run --rm -d --name $(CONTAINER_NAME)-$(CONTAINER_INSTANCE) \
		$(PORTS) $(VOLUMES) $(NS)/$(IMAGE_NAME):$(VERSION)

shell:
	docker exec -it $(CONTAINER_NAME)-$(CONTAINER_INSTANCE) bash

default: run
