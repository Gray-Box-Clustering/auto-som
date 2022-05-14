#!/usr/bin/env bash

cd ..
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mongo-container