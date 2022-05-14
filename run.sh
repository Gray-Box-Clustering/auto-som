#!/usr/bin/env bash

if docker-compose --version; then
  docker-compose up --remove-orphans
else
  echo "'docker-compose' service wasn't found. Make sure you have docker-compose installed"
fi