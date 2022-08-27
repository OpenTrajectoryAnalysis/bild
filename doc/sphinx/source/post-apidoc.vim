" Usage: vim -S post-apidoc.vim

function! s:underline(underline_char)
	let text = getline(".")
	let underline = substitute(text, '.', a:underline_char, "g")
	norm o
	call setline(".", underline)
endfunction

function! s:fix_underline()
	norm j
	let underline_char = strpart(getline("."), 0, 1)
	norm ddk
	call s:underline(underline_char)
endfunction

for fname in glob("bild*.rst", 0, 1)
	exe "edit! " . fname

	" Remove "Submodules" and "Subpackages" headings
	norm gg
	while search('Sub\(packages\|modules\)') > 0
		norm 2dd
	endwhile

" 	" Move submodule headings one level down
" 	" (Unnecessary since we deleted the 'Submodules' heading)
" 	norm gg
" 	while search('\w\.\w.*module\n-', "W") > 0
" 		let myline = line(".")
" 		let line = getline(myline)
" 		let repl = substitute(line, '.', '^', 'g')
" 		call setline(myline+1, repl)
" 	endwhile

	" Remove the words "package" and "module" from headings
	norm gg
	while search(' \(package\|module\)\n[-^=]') > 0
		norm / \(package\|module\)D
		call s:fix_underline()
	endwhile

" 	" List only toplevel names instead of whole name
" 	norm gg
" 	while search('^bild\..*\n[-^=]') > 0
" 		norm $T.d0
" 		call s:fix_underline()
" 	endwhile

	if match(fname, 'bild\..*\.rst') >= 0 " we are in a subpackage
		norm 3ggo.. contents::   :local:
	endif

	" The 'Module contents' section is usually empty (unless we define
	" functions in __init__.py or crap like that)
	call search("^Module contents$")
	norm dG

	write!
endfor

" Manually copying __init__.py docstring
" How do we do this automatically?
edit! ../../../bild/__init__.py
norm G/"""j"syn

edit! bild.rst
call search("^==")
norm o.. contents::   :local:k
norm O"sP
write!

quit
