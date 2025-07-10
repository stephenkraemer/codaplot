# Style

- use black for python modules and notebooks

To ensure formatting we use a pre-commit hook (githooks/pre-commit)

To enable this hook
- make sure the hook script is executable
- make sure you have black installed
- run `git config core.hooksPath githooks`