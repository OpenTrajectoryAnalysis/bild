DOCDIR = doc
SPHINXDIR = $(DOCDIR)/sphinx
SPHINXBUILD = $(SPHINXDIR)/build
SPHINXSOURCE = $(SPHINXDIR)/source
TESTDIR = tests
TESTFILE = test_bild.py
COVERAGEREPFLAGS = --omit=*/noctiluca/*,*/rouse/*
COVERAGEREPDIR = $(TESTDIR)/coverage
MODULE = bild

.PHONY : docs tests all clean mydocs mytests myall myclean

all : docs tests

docs :
	sphinx-apidoc -f -o $(SPHINXSOURCE) $(MODULE)
	@rm $(SPHINXSOURCE)/modules.rst
	@cd $(SPHINXSOURCE) && vim -n -S post-apidoc.vim
	cd $(SPHINXDIR) && $(MAKE) html

tests :
	cd $(TESTDIR) && coverage run $(TESTFILE)
	@mv $(TESTDIR)/.coverage .
	coverage html -d $(COVERAGEREPDIR) $(COVERAGEREPFLAGS)

clean :
	-rm -r $(SPHINXBUILD)/*
	-rm -r $(COVERAGEREPDIR)/*
	-rm .coverage

# Personal convenience targets
DUMPPATH = "/home/simongh/Dropbox (MIT)/htmldump"
mydocs : docs
	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx

mytests : tests
	cp -r $(COVERAGEREPDIR)/* $(DUMPPATH)/coverage

myall : mydocs mytests

myclean : clean
	-rm -r $(DUMPPATH)/sphinx/*
	-rm -r $(DUMPPATH)/coverage/*
