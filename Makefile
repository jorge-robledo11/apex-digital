# -----------------------------------------------------------------------------
# Makefile para gestionar infraestructura ML con control de espacio
# -----------------------------------------------------------------------------

COMPOSE = docker compose -f deployment/mlflow/docker-compose.yaml -f deployment/redis/docker-compose.yaml

.PHONY: up down restart logs ps clean rebuild help

## Levanta toda la infraestructura ML (limpia antes)
up: clean-unused
	$(COMPOSE) up -d --build
	@echo "🎉 Infraestructura completa iniciada:"
	@echo "📊 MLflow: http://localhost:5555"
	@echo "🔄 Redis UI: http://localhost:8081"

## Detiene toda la infraestructura
down:
	$(COMPOSE) down

## Reinicia toda la infraestructura
restart: down up

## Muestra logs de todos los servicios
logs:
	$(COMPOSE) logs -f

## Lista el estado de todos los contenedores
ps:
	$(COMPOSE) ps

## Limpieza ligera (solo imágenes colgantes)
clean:
	@echo "🧹 Limpiando imágenes no utilizadas..."
	docker image prune -f
	docker container prune -f
	@echo "✅ Limpieza básica completada"

## Limpieza completa (TODAS las imágenes no utilizadas)
clean-all:
	@echo "⚠️ Limpieza completa de Docker..."
	docker system prune -a -f --volumes
	@echo "✅ Limpieza completa terminada"

## Limpieza inteligente (solo antes de levantar)
clean-unused:
	@echo "🧹 Limpiando imágenes no utilizadas antes del deploy..."
	docker image prune -f
	@echo "✅ Listo para deploy"

## Reconstruye sin cache y reinicia
rebuild: clean-all
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

## Muestra uso de espacio Docker
disk-usage:
	@echo "💾 Uso de espacio Docker:"
	docker system df

## Muestra esta ayuda
help:
	@echo "Comandos disponibles:"
	@echo "  make up          -> Levanta MLflow + Redis (limpia antes)"
	@echo "  make down        -> Detiene toda la infraestructura"
	@echo "  make restart     -> Reinicia toda la infraestructura"
	@echo "  make logs        -> Ver logs en tiempo real"
	@echo "  make ps          -> Estado de contenedores"
	@echo "  make clean       -> Limpieza ligera de Docker"
	@echo "  make clean-all   -> Limpieza completa de Docker"
	@echo "  make disk-usage  -> Ver uso de espacio"
	@echo "  make rebuild     -> Reconstruir sin cache"

.DEFAULT_GOAL := help
