# -------------------------------------------------------------------
# Makefile (Compose en raíz) con derribo “fuerte” si hay desalineación
# -------------------------------------------------------------------

ROOT_DIR       := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE_FILE   ?= $(ROOT_DIR)docker-compose.yaml

# 1) Si hay COMPOSE_PROJECT_NAME en el entorno, úsalo; si no, intenta detectar
# el proyecto de los contenedores actuales; si no, usa el nombre del directorio
DETECTED_PROJECT := $(shell docker inspect mlflow-tracking-server --format '{{ index .Config.Labels "com.docker.compose.project"}}' 2>/dev/null)
PROJECT_NAME     ?= $(if $(COMPOSE_PROJECT_NAME),$(COMPOSE_PROJECT_NAME),$(if $(DETECTED_PROJECT),$(DETECTED_PROJECT),digital-orders))

# 2) Detectar binario compose
COMPOSE_BIN := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || (docker-compose version >/dev/null 2>&1 && echo "docker-compose" || echo ""))
ifeq ($(COMPOSE_BIN),)
  $(error "No se encontró 'docker compose' ni 'docker-compose'")
endif
COMPOSE = $(COMPOSE_BIN) -f $(COMPOSE_FILE) -p $(PROJECT_NAME)

API_SERVICE      = api
MLFLOW_SERVICE   = mlflow
POSTGRES_SERVICE = postgres

.PHONY: up up-ml up-api down down-v down-any restart restart-api logs logs-api ps ps-all \
        build-api rebuild-api health-api clean clean-all clean-unused rebuild disk-usage doctor help

up: clean-unused
	$(COMPOSE) up -d --build
	@echo "🎉 Infra iniciada:"
	@echo "📊 MLflow:  http://localhost:5555"
	@echo "🛰️  API:     http://localhost:8000/health"

up-ml: clean-unused
	$(COMPOSE) up -d --build $(POSTGRES_SERVICE) $(MLFLOW_SERVICE)
	@echo "📊 MLflow:  http://localhost:5555"

up-api:
	$(COMPOSE) up -d --build $(API_SERVICE)
	@echo "🛰️  API:     http://localhost:8000/health"

# Baja el stack *de este proyecto*
down:
	$(COMPOSE) down --remove-orphans

# Baja el stack y borra volúmenes *de este proyecto*
down-v:
	$(COMPOSE) down -v --remove-orphans

# 🔥 Derribo “fuerte”: elimina cualquier contenedor etiquetado con el project actual,
# aunque haya sido levantado con otro compose/ubicación
down-any:
	@echo "🛑 Derribo fuerte para project: $(PROJECT_NAME)"
	@IDS=$$(docker ps -aq -f "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$IDS" ]; then docker rm -f $$IDS; else echo "No hay contenedores con ese proyecto."; fi
	@VOLS=$$(docker volume ls -q -f "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$VOLS" ]; then docker volume rm $$VOLS; else echo "No hay volúmenes con ese proyecto."; fi
	@NW=$$(docker network ls -q -f "label=com.docker.compose.project=$(PROJECT_NAME)"); \
	if [ -n "$$NW" ]; then docker network rm $$NW; else echo "No hay redes con ese proyecto."; fi

restart: down up
restart-api:
	$(COMPOSE) restart $(API_SERVICE)

logs:
	$(COMPOSE) logs -f
logs-api:
	$(COMPOSE) logs -f $(API_SERVICE)

ps:
	$(COMPOSE) ps
ps-all:
	docker ps -a

build-api:
	$(COMPOSE) build $(API_SERVICE)
rebuild-api:
	$(COMPOSE) build --no-cache $(API_SERVICE) && $(COMPOSE) up -d $(API_SERVICE)

health-api:
	@curl -fsS http://localhost:8000/health >/dev/null && echo "✅ API OK" || (echo "❌ API no responde" && exit 1)

clean:
	docker image prune -f
	docker container prune -f
clean-all:
	docker system prune -a -f --volumes
clean-unused:
	docker image prune -f >/dev/null 2>&1 || true

rebuild: clean-all
	$(COMPOSE) build --no-cache && $(COMPOSE) up -d

disk-usage:
	docker system df

doctor:
	@echo "🐳 Docker:        $$(docker --version)"
	@echo "🧩 Compose bin:   $(COMPOSE_BIN)"
	@echo "📄 Compose file:  $(COMPOSE_FILE)"
	@echo "📦 Project:       $(PROJECT_NAME)"
	@echo "🏷  Detectado:     $(DETECTED_PROJECT)"
	@echo "🧾 Activos con etiqueta del proyecto:"
	@docker ps --format '{{.ID}} {{.Names}}' -f "label=com.docker.compose.project=$(PROJECT_NAME)" || true
	@echo "🔎 Servicios en compose:"
	@grep -E '^[[:space:]]{2}[a-zA-Z0-9_-]+:' $(COMPOSE_FILE) | sed 's/^/  - /'

help:
	@echo "Comandos:"
	@echo "  make up / down / down-v / down-any / restart"
	@echo "  make up-ml / up-api / restart-api / logs / logs-api"
	@echo "  make ps / ps-all / build-api / rebuild-api / health-api"
	@echo "  make clean / clean-all / disk-usage / doctor / rebuild"
