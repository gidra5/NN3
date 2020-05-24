SHELL = /bin/bash

all: clean build test

clean:
	@rm -r ./build || true

build:
	@echo 'Building...'
	@tsc -t 'ESNEXT' --strictFunctionTypes --noImplicitReturns --outDir ./build ./src/src.ts

test:
	node ./tests/*