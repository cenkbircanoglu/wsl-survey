

make docker-release

docker-compose -f scripts/experiments/wildcat/cpu.yml  up

docker-compose -f scripts/experiments/wildcat/gpu.yml  up

