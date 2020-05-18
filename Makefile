.PHONY: all lint test test-cov install dev uninstall-dev clean distclean

PYTHON ?= python

all: ;

lint:
	flake8

test: all
	py.test

test-cov: all
	py.test --cov-report=term-missing --cov=q2_diversity_lib

install:
	$(PYTHON) setup.py install

dev: all
	pip install -e .

uninstall-dev:
	pip uninstall q2-diversity-lib

clean: distclean

distclean: ;
