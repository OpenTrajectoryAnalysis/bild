MODULE           = bild
DISTDIR          = dist
CYTHONVENV       = .venv_cython
CYTHONPYTHON     = $(CYTHONVENV)/bin/python3
CYTHONSRCDIR     = $(MODULE)/src
CYTHONBINDIR     = $(MODULE)/bin
CYTHONYELLOWDIR  = $(DOCDIR)/cython_yellow
TESTDIR          = tests
TESTVENV         = $(TESTDIR)/env
TESTPYTHON       = ./$(TESTVENV)/bin/python3 -I
TESTFILE         = $(TESTDIR)/test_bild.py
COVERAGEREPFLAGS = --omit=*/noctiluca/*,*/rouse/*,*/bayesmsd/*
COVERAGEREPDIR   = $(TESTDIR)/coverage
BUILDVENV        = build.env# protected by setuptools
BUILDPYTHON      = ./$(BUILDVENV)/bin/python3 -I
LINUXPLAT        = linux_x86_64
BUILDPLAT        = manylinux2014_x86_64
DOCDIR           = doc
SPHINXDIR        = $(DOCDIR)/sphinx
SPHINXSOURCE     = $(SPHINXDIR)/source
SPHINXBUILD      = $(SPHINXDIR)/source/_build

PYTHON           = python3

.PHONY : setup cythonize recompile yellow build pre-docs docs docs-latex tests clean

FORCE: # dummy target for forcing rebuilds

setup :
	nbstripout --install
	vim "+g/-m nbstripout/norm A --drop-empty-cells" +wq .git/config

$(TESTVENV) : # to rebuild: make -B <TESTVENV>
	$(PYTHON) -m venv --clear $(TESTVENV)
	env PYTHONPATH= $(TESTPYTHON) -m ensurepip # ensurepip during venv ignores -I option
	$(TESTPYTHON) -m pip install --upgrade pip
	$(TESTPYTHON) -m pip install coverage
	-$(TESTPYTHON) -m pip install $$($(CYTHONPYTHON) -m pip freeze | grep ^numpy)
	-$(TESTPYTHON) -m pip install $$($(CYTHONPYTHON) -m pip freeze | grep ^scipy)
	$(TESTPYTHON) -m pip install -e .

$(CYTHONVENV) : # to rebuild: make -B <CYTHONVENV>
	$(PYTHON) -m venv --clear $(CYTHONVENV)
	$(CYTHONPYTHON) -m pip install --upgrade pip setuptools
	$(CYTHONPYTHON) -m pip install --upgrade-strategy only-if-needed -r cython_requirements.txt

$(CYTHONSRCDIR)/*.c : $(CYTHONSRCDIR)/*.pyx $(CYTHONVENV)
	$(CYTHONPYTHON) -c "from setup import extensions, cythonize; cythonize(extensions)"

cythonize: $(CYTHONSRCDIR)/*.c

$(CYTHONBINDIR)/*.so : $(CYTHONSRCDIR)/*.c
	mkdir -p $(CYTHONBINDIR)
	-rm -r $(CYTHONBINDIR)/*
	$(CYTHONPYTHON) setup.py build_ext --inplace

recompile : $(CYTHONBINDIR)/*.so

yellow : $(CYTHONSRCDIR)/*.pyx
	mkdir -p $(CYTHONYELLOWDIR)
	-rm -r $(CYTHONYELLOWDIR)/*
	$(PYTHON) -m cython -3 -a $(CYTHONSRCDIR)/*.pyx
	@mv $(CYTHONSRCDIR)/*.html $(CYTHONYELLOWDIR)

build : $(CYTHONSRCDIR)/*.c $(MODULE)/*.py
	@# clean
	mkdir -p $(CYTHONBINDIR) $(DISTDIR)
	-rm -r $(CYTHONBINDIR)/*
	-rm -r $(DISTDIR)/*

	@# set up venv
	-rm -r $(BUILDVENV)
	$(PYTHON) -m venv --clear $(BUILDVENV)
	$(BUILDPYTHON) -m pip install --upgrade pip setuptools
	$(BUILDPYTHON) -m pip install --upgrade build patchelf auditwheel

	# make python-only fall-back
	PYTHON_ONLY=1 $(BUILDPYTHON) -m build --wheel
	
	# build source dist & linux wheel
	$(BUILDPYTHON) -m build -o $(DISTDIR)

	# fix linux wheel
	@# auditwheel looks for patchelf in shell PATH, so provide it there (PATH=...)
	PATH="$$PATH:$(BUILDVENV)/bin/" $(BUILDPYTHON) -m auditwheel repair --plat $(BUILDPLAT) \
					    -w $(DISTDIR) \
					    $(DISTDIR)/$(MODULE)-*-$(LINUXPLAT).whl
	rm $(DISTDIR)/$(MODULE)-*-$(LINUXPLAT).whl

	# test everything
	@echo "Testing all wheels found in $(DISTDIR)"
	@for whl in $(DISTDIR)/*.whl; do \
		$(BUILDPYTHON) -m pip install -q $${whl}; \
		echo; \
		echo "with $${whl}"; \
		echo "$$ python -c \"import $(MODULE)\""; \
		$(BUILDPYTHON) -c "import $(MODULE)"; \
		$(BUILDPYTHON) -m pip uninstall -yq $(MODULE); \
	done
	@echo
	@echo "Build successful."
	@echo "To upload built packages, run 'python3 -m twine upload $(DISTDIR)/*'"

pre-docs :
	sphinx-apidoc -f -o $(SPHINXSOURCE) $(MODULE)
	@rm $(SPHINXSOURCE)/modules.rst
	@cd $(SPHINXSOURCE) && vim -nS post-apidoc.vim
	cd $(SPHINXDIR) && $(MAKE) clean

docs : pre-docs recompile
	cd $(SPHINXDIR) && $(MAKE) html

docs-latex : pre-docs recompile
	cd $(SPHINXDIR) && $(MAKE) latex

tests : recompile | $(TESTVENV)
	mkdir -p $(COVERAGEREPDIR)
	$(TESTPYTHON) -m coverage run $(TESTFILE)
	$(TESTPYTHON) -m coverage html -d $(COVERAGEREPDIR) $(COVERAGEREPFLAGS)
	$(TESTPYTHON) -m coverage report --skip-covered $(COVERAGEREPFLAGS)

clean :
	-rm -r $(CYTHONBINDIR)
	-rm -r $(CYTHONYELLOWDIR)
	-rm -r $(CYTHONVENV)
	-rm -r $(BUILDVENV)
	-rm -r $(DISTDIR)
	-rm -r $(SPHINXBUILD)
	-rm -r $(COVERAGEREPDIR)
	-rm -r $(TESTVENV)
	-rm .coverage
	-rm -r build # created by python -m build


# # Personal convenience targets
# # Edit DUMPPATH to point to a directory that you can easily access
# # For example, when working remotely, sync output via Dropbox to inspect locally
# DUMPPATH = "/home/simongh/Dropbox (MIT)/htmldump"
# mydocs : docs
# 	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx
# 
# mydocs-latex : docs-latex
# 	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx
# 
# mytests : tests
# 	cp -r $(COVERAGEREPDIR)/* $(DUMPPATH)/coverage
# 
# myall : mydocs mytests
# 
# myclean : clean
# 	-rm -r $(DUMPPATH)/sphinx/*
# 	-rm -r $(DUMPPATH)/coverage/*
