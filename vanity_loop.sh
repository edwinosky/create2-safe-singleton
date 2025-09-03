#!/bin/bash

FACTORY="0xC0DEb853af168215879d284cc8B4d0A645fA9b0E"
INITCODEHASH="0xdf4612f952fb0a508367652224771fd55e8dad6c4d624b4fdd29d31775e290c2"
PREFIX="7702"
SUFFIX="7702"

START=0
BATCH=10000000000   # 1e10
BLOCKS=1024
THREADS=1024
INNER=10000

while true; do
    echo "⏳ Buscando desde salt $START (batch $BATCH)..."

    ./vanity \
      --factory $FACTORY \
      --initcodehash $INITCODEHASH \
      --prefix $PREFIX \
      --suffix $SUFFIX \
      --start $START \
      --batch $BATCH \
      --blocks $BLOCKS \
      --threads $THREADS \
      --inner $INNER | grep -E "Encontrado|match"

    # incrementar START para la siguiente corrida
    START=$((START + BATCH))
done
