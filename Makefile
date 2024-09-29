.PHONY: format
format: 
	isort -m VERTICAL_HANGING_INDENT --skip-gitignore --ac --tc --profile black .
	black .
