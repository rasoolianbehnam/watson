pdflatex $1
#while [ 1 ];do find test.md -mmin -.0333 -exec pandoc --bibliography=./lit.bib -o test.pdf test.md \;; sleep 2; done
#bibtex basics.aux
#pdflatex basics.tex
#pdfllatex basics.tex
