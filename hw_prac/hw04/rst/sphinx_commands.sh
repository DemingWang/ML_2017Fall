Author: Bhishan Poudel
Date  :  Oct 16, 2017

Topic: Sphinx documentation for hw04
====================================
function spb(){
    cd docs; make clean; cd -
    cd docs; sphinx-build -b html source build/html; cd -
    open docs/build/html/index.html
}

## sprstq  copies updated rst files and builds again
function spb_rstq(){
    # copy rst files  (not index.rest file)
    cp rst/*.rst docs/source
    cd docs; make clean; cd -
    cd docs; sphinx-build -b html source build/html; cd -
    open docs/build/html/index.html
}



# spallf_no_open code/scipy/    
function spallf_no_open () {

    #1. Quickstart
    # Outputs: docs
    # docs has three things: Makefile source build
    # NOTE: --ext-napoleon gives error, but runs fine if added in conf.py
    sphinx-quickstart -q -p "Bhishan's" -a "Bhishan Poudel" -v 1 -r 1 \
     --ext-autodoc --ext-doctest --ext-viewcode --ext-imgmath \
     --no-batchfile --sep docs

    #2. Copy edit_conf file
    cp ~/Applications/edit_sphinx_conf.py edit_sphinx_conf.py

    #3. Edit conf.py file.
    python edit_sphinx_conf.py; rm -rf edit_sphinx_conf.py

    #4. Create html folder (also creates doctrees).
    cd docs; make html; cd -


    #5. Auto create rst files.
    # sphinx-apidoc -o docs/source src/
    sphinx-apidoc -o docs/source "${1%?}"

    #6. Remove the string 'module' from all rst files
    for f in docs/source/*.rst; do sed -ie '1s/module//' $f; done
    for f in docs/source/*.rste; do rm $f; done


    #7. Rename modules.rst to index.rst
    mv docs/source/modules.rst docs/source/index.rst

    #8. Add path to conf.py
    # path.append is relative to Makefile not conf.py
    # vi docs/source/conf.py  then, sys.path.append('../src/')
    awk -v n=23 -v s="sys.path.append('../${1%?}')" 'NR == n {print s} {print}' \
    docs/source/conf.py > docs/source/conf_new.py;
    rm docs/source/conf.py; mv docs/source/conf_new.py docs/source/conf.py

    #9 b. Add napoleon extension to conf.py (it did not worked adding above)
    # 'sphinx.ext.napoleon',
    # cd docs; make clean; make html; open build/html/index.html
    awk -v n=38 -v s="    'sphinx.ext.napoleon'," 'NR == n {print s} {print}' \
    docs/source/conf.py > docs/source/conf_new.py;
    rm docs/source/conf.py; mv docs/source/conf_new.py docs/source/conf.py

    #10. Add Table of Contents to index.rst
    awk -v n=1 -v s=".. contents:: Table of Contents\n   :depth: 3\n\n" \
                    'NR == n {print s} {print}' \
                  docs/source/index.rst > docs/source/tmp; mv docs/source/tmp docs/source/index.rst

    #11. Get index.html (pdf is very very bad.)
    cd docs; sphinx-build -b html source build/html; cd -


    #12. Delete pycache
    rm -rf "${1%?}"/__pycache__

    #13. Open html
    # open docs/build/html/index.html
    }


    # sp_dir2 scikit
    function sp_dir2(){
      rm -rf docs
      spallf_no_open code/scipy/
      folder2=$1
      mkdir docs/source/$folder2
      sphinx-apidoc -o docs/source/$folder2 code/$folder2
      
      awk -v n=24 -v s="sys.path.append('../code')" 'NR == n {print s} {print}' docs/source/conf.py > docs/source/conf_new.py
      awk -v n=25 -v s="sys.path.append('../code/$folder2')" 'NR == n {print s} {print}' docs/source/conf_new.py > docs/source/conf_new2.py
      rm -rf docs/source/conf.py docs/source/conf_new.py; mv docs/source/conf_new2.py docs/source/conf.py  
      
      # Remove the string 'module' from all rst files
      for f in docs/source/$folder2/*.rst; do sed -ie '1s/module//' $f; done
      for f in docs/source/$folder2/*.rste; do rm $f; done
      
      echo "" >> docs/source/index.rst
      echo "" >> docs/source/index.rst
      cat docs/source/$folder2/modules.rst >> docs/source/index.rst
      rm docs/source/$folder2/modules.rst
      
      rm -rf "$1"/__pycache__
      
      cp rst/*.rst docs/source    # copy rst files except index.rest
      cp rst/*.rest docs/source   # copy index.rest
      
      mv docs/source/index.rst docs/source/index2.rst
      mv docs/source/index.rest docs/source/index.rst
      cat docs/source/index2.rst >> docs/source/index.rst
      
      
      open docs/source/index.rst  # add $folder2/  prefix to added folder
      
      open docs/source/$folder2/*.rst # .. automodule:: scikit.softmaxExercise
      
      
      # Then build again:  spb_rstq

    }
