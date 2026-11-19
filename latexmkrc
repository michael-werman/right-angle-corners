# Lets supplement.tex read the article's numbering (package xr): when the
# supplement is compiled, latexmk first compiles main.tex to produce main.aux.
add_cus_dep('tex', 'aux', 0, 'makeexternaldocument');
sub makeexternaldocument {
    if (!($root_filename eq $_[0])) {
        system("latexmk -cd -pdf \"$_[0]\"");
    }
}
