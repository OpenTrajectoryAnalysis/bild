DOCDIR = doc
SPHINXDIR = $(DOCDIR)/sphinx
SPHINXSOURCE = $(SPHINXDIR)/source
SPHINXBUILD = $(SPHINXDIR)/source/_build
TESTDIR = tests
TESTFILE = test_bild.py
COVERAGEREPFLAGS = --omit=*/noctiluca/*,*/rouse/*,*/bayesmsd/*
COVERAGEREPDIR = $(TESTDIR)/coverage
DISTDIR = dist
MODULE = bild
CYTHONSRCDIR = bild/src
CYTHONBINDIR = bild/bin
CYTHONYELLOWDIR = doc/cython_yellow

.PHONY : recompile yellow build pre-docs docs tests all clean mydocs mytests myall myclean

setup :
	mkdir -p $(COVERAGEREPDIR) $(DISTDIR) $(CYTHONBINDIR) $(CYTHONYELLOWDIR)
	nbstripout --install
	vim "+g/-m nbstripout/norm A --drop-empty-cells" +wq .git/config

all : docs tests build

bild/src/*.c : bild/src/*.pyx
	-@cd $(CYTHONSRCDIR) && rm *.c
	CYTHONIZE=1 python3 setup.py build_ext --inplace

recompile : bild/src/*.c

yellow : bild/src/*.pyx
	cython -3 -a $(CYTHONSRCDIR)/*.pyx
	@mv $(CYTHONSRCDIR)/*.html $(CYTHONYELLOWDIR)

build : recompile
	-@cd $(DISTDIR) && rm *
	python3 -m build # source dist & linux wheel
	@cd $(DISTDIR) && auditwheel repair *-linux_*.whl
	@cd $(DISTDIR) && mv wheelhouse/* . && rmdir wheelhouse
	@cd $(DISTDIR) && rm *-linux_*.whl
	PYTHON_ONLY=1 python3 -m build --wheel # py3-none-any wheel

pre-docs :
	sphinx-apidoc -f -o $(SPHINXSOURCE) $(MODULE)
	@rm $(SPHINXSOURCE)/modules.rst
	@cd $(SPHINXSOURCE) && vim -nS post-apidoc.vim
	cd $(SPHINXDIR) && $(MAKE) clean

docs : pre-docs
	cd $(SPHINXDIR) && $(MAKE) html

docs-latex : pre-docs
	cd $(SPHINXDIR) && $(MAKE) latex

tests :
	cd $(TESTDIR) && coverage run $(TESTFILE)
	@mv $(TESTDIR)/.coverage .
	coverage html -d $(COVERAGEREPDIR) $(COVERAGEREPFLAGS)
	coverage report --skip-covered $(COVERAGEREPFLAGS)

clean :
	-rm -r $(SPHINXBUILD)/*
	-rm -r $(COVERAGEREPDIR)/*
	-rm .coverage

# Personal convenience targets
# Edit DUMPPATH to point to a directory that you can easily access
# For example, when working remotely, sync output via Dropbox to inspect locally
DUMPPATH = "/home/simongh/Dropbox (MIT)/htmldump"
mydocs : docs
	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx

mydocs-latex : docs-latex
	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx

mytests : tests
	cp -r $(COVERAGEREPDIR)/* $(DUMPPATH)/coverage

myall : mydocs mytests

myclean : clean
	-rm -r $(DUMPPATH)/sphinx/*
	-rm -r $(DUMPPATH)/coverage/*
