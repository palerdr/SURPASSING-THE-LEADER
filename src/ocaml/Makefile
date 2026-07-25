.PHONY: build build-lib test cli fmt clean

build:
	dune build

build-lib:
	dune build lib/engine/dth_engine.cmxa lib/solver/dth_solver.cmxa

test:
	dune runtest

cli:
	dune exec dth-play

fmt:
	dune fmt

clean:
	dune clean
