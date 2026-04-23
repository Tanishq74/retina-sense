TEX=retinasense_ieee
PDF=$(TEX).pdf

all: $(PDF)

$(PDF): $(TEX).tex
	pdflatex $(TEX).tex
	pdflatex $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).bbl $(TEX).blg $(TEX).pdf

.PHONY: all clean
