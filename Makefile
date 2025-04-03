spark-up:
	docker compose -f ./spark/docker-compose.yaml up --scale spark-worker=2 -d

spark-down:
	docker compose -f ./spark/docker-compose.yaml down --volumes
