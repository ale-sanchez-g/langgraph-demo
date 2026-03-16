#!/bin/bash

# Stop all containers and remove them
docker compose down

# Start all services in detached mode
docker compose up -d