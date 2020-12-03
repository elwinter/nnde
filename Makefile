version:
	true

sdist:
	python setup.py sdist

bdist_wheel:
	python setup.py bdist_wheel

pypi:
	python -m twine upload dist/*
