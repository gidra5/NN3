SHELL = /bin/bash

all: clean
	@echo 'Building...'
	@tsc -t 'ESNEXT' --strictFunctionTypes --noImplicitReturns --outDir ./build ./src.ts
	@echo 'Executing...'
	@node ./build/src.js

clean:
	@rm -r ./build || true