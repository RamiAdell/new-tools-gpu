build:
	docker compose -f compose.yml up --build -d --remove-orphans

up:
	docker compose -f compose.yml up -d

down:
	docker compose -f compose.yml down

down-v:
	docker compose -f compose.yml down -v

config:
	docker compose -f compose.yml config