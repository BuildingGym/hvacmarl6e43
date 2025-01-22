latexmk = latexmk

default: *.pdf clean

%.pdf: %.tex
	$(latexmk) -pdf -pdflatex=lualatex -shell-escape -use-make -g -interaction=nonstopmode -halt-on-error $<

clean:
	$(latexmk) -c