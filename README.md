1. activate environtment (running on python version 3.5)
pyvenv-3.5 ~/Envs/thesis_python3/
source ~/Envs/thesis_python3/bin/activate

2. Construction ontology (on readOntology prj)
python constructOntology.py

3. run clustering on specific algorithms
python clustering/hierarchicalclustering.py
python clustering/bikmeansclustering.py
...

4. replace PHI instance after run clustering
python replace/replace.py

5. deactivate environtments
deactivate

