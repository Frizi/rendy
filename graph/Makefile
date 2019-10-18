graphviz:
	dot -Gsize="20,30" -Gpackmode=graph -Gquantum=0.5 -Nfontname=consolas *.dot | gvpack -g -u -array_r2 -m30 | neato  -n2 -Tsvg -o graph.svg

watch:
	while true; do ls *.dot | entr -d -s 'make graphviz'; done