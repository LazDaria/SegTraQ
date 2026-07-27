# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/LazDaria/segtraq/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/LazDaria/segtraq/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `segtraq` for local development.

1. Fork the `segtraq` repo on GitHub.
2. Clone your fork locally:

   ```sh
   git clone git@github.com:your_name_here/segtraq.git
   ```

3. Install your local copy using uv:
```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cd segtraq/
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[test]"
```

4. Download the test data:
```sh
   mkdir -p tests/data
   curl -L --fail --retry 3 \
          -o test_data.tar.gz \
          "https://zenodo.org/records/20474382/files/segtraq_test_data_v2.tar.gz?download=1"
   tar -xzf test_data.tar.gz -C tests/data
```
   Test data is not bundled with the repository. This step is required before running the test suite locally.

5. Create a branch for local development:

   ```sh
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass linting and tests:
```sh
   make qa
   make test
```

7. Commit your changes and push your branch to GitHub:

   ```sh
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
3. The pull request should work for Python 3.12 and 3.13. Tests run in GitHub Actions on every pull request to the main branch, make sure that the tests pass for all supported Python versions.

### Write Documentation

Docs are written as jupytext-paired `.py` files under `docs/notebooks/` — no
`.ipynb` files should be committed. To build the docs locally (requires the
test data from step 4 above):

```sh
uv pip install -e ".[docs]"
make docs
```

This executes the notebooks against real data and builds the HTML site into
`docs/_build/html`. Please double-check the resulting HTML pages for correctness.
Maintainers can publish the built site with:

```sh
make deploy-docs
```

## Tips

To run a subset of tests:

```sh
pytest tests.test_segtraq
```

## Deploying

A reminder for the maintainers on how to deploy. Make sure all your changes are committed (including an entry in HISTORY.md). Then run:

```sh
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

You can set up a [GitHub Actions workflow](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python#publishing-to-pypi) to automatically deploy your package to PyPI when you push a new tag.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
