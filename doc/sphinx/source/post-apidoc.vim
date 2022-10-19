" Usage: vim -S post-apidoc.vim

function! s:underline(underline_char)
	let text = getline(".")
	let underline = substitute(text, '.', a:underline_char, "g")
	call append(line("."), underline)
endfunction

function! s:fix_underline()
	let underline_char = strpart(getline(line(".")+1), 0, 1)
	norm jddk
	call s:underline(underline_char)
endfunction

for fname in glob("bild*.rst", 0, 1)
	exe "edit! " . fname

	" Remove "Submodules" and "Subpackages" headings
	0
	while search('Sub\(packages\|modules\)') > 0
		delete _ 2
	endwhile

	" Remove the words "package" and "module" from headings
	0
	while search(' \(package\|module\)\n[-^=]') > 0
		substitute/ \(package\|module\)//
		call s:fix_underline()
	endwhile

	" add local contents to toc
	if match(fname, 'bild\..*\.rst') >= 0 " we are in a subpackage
		call append(2, ["",
			       \".. contents::",
			       \"   :local:",
			       \])
	endif

	" The 'Module contents' section is usually empty (unless we define
	" functions in __init__.py or crap like that)
	call search("^Module contents$")
	.,$delete _

	write!
endfor

edit! bild.rst
call setline(1, "API reference") " replace the title; we know what the module is called
call setline(2, "=============")
call append(3, [".. contents::",
	       \"   :local:",
	       \"",
	       \".. automodule:: bild",
	       \"   :members:",
	       \"",
	       \])
write!

quit
