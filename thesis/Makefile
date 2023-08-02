LANG = en-US

TMP_DIR ?= /tmp/$(notdir $(CURDIR))

TEXS ?= $(wildcard *.tex)
PDFS ?= $(TEXS:.tex=.pdf)
TXTS ?= $(TEXS:.tex=.txt)

LATEXMKFLAGS+=-pdf -pdflatex="pdflatex --shell-escape %O %S"

########################################################################
# top-level targets for compilation in /tmp
#

all: pdf

# You can suffix all targets with "_in_tmp", e.g. ``make debug_in_tmp``.
# If you have a tmpfs mounted to ``/tmp`` you can use this to speed up
# compilation.
%_in_tmp:
	rsync -qa --inplace "$(CURDIR)/" "$(TMP_DIR)/"
	$(MAKE) -C "$(TMP_DIR)" $*

########################################################################
# top-level targets
#

debug: $(PDFS)

pdf: LATEXMKFLAGS += -quiet
pdf: $(PDFS)

########################################################################
# targets for specific files
#

%.pdf: %.tex
	latexmk $(LATEXMKFLAGS) $^

%.txt: %.pdf
	pdftotext $^ > $@

########################################################################
# QA, stats, etc.
#

count: $(TEXS)
	texcount -inc $^

languagetool: $(TXTS)
	languagetool-commandline -l $(LANG) $^

lint: $(TEXS)
	for f in $^; do \
		lacheck $$f; \
		chktex $$f; \
	done

clean:
	latexmk -c
	find \( \
		-type f \
		-name '*.bbl' \
		-or -name '*-blx.bib' \
		-or -name '*.run.xml' \
		-or -name '*.aux' \
	\) -exec rm {} \;
	rm -rf "$(TMP_DIR)"

distclean:
	rm -f \
		$(PDFS) \
		$(TXTS)
