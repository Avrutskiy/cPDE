# Contributing

The Python client describes work and communicates with a persistent CPU or CUDA server.
Keep server compilation outside normal Python use and keep training loops inside the
server. The current public API and shapes are described in `README.md`.

## Local development

```sh
python -m pip install -e '.[dev,geometry]'
python -B -m unittest discover -s tests -v
```

Without executable environment variables, tests build a serial host probe using
`g++`. It exercises arithmetic and transport but does not validate CUDA or CUB.
The GitHub Actions workflow runs this path with Python 3.9 on Ubuntu 24.04 and
Python 3.13 on Ubuntu 26.04, smoke tests the examples, and builds the Python
distributions and source/samples ZIPs. Python 3.9 is kept on Ubuntu 24.04 because
setup-python does not provide a 3.9 build for Ubuntu 26.04. It follows
GitHub's [Python workflow documentation](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

The CPU-backend workflow also builds the production target on macOS, Linux and
Windows, covering Accelerate, system CBLAS and portable routines. See
[BUILDING.md](BUILDING.md) for local scripts and backend selection.

For CUDA changes, also build with `CPDE_BUILD_REDUCTION_TESTS=ON`, run CTest, and set
both `CPDE_TEST_EXECUTABLE` and `CPDE_TEST_REDUCTION_EXECUTABLE` before running the
Python suite. See the platform commands in the main README. Include the GPU,
toolkit/compiler versions, and checks performed when describing a change.

Do not change optimizer formulas or the retained derivative/bias-gradient paths as
an incidental cleanup. Distinguish implemented behavior from proposed behavior in
`IMPLEMENTATION_PLAN.md`. Add focused numerical tests when changing cost or gradient
semantics, including chunk tails and non-finite handling where relevant.

## Packaging

```sh
python -m build
python scripts/bundle_release.py
```

The wheel contains the Python client only; it does not bundle a server binary.
The source distribution includes CPU/CUDA sources and build scripts, examples, and tests. The portable
source ZIP additionally includes GitHub configuration and a file hash manifest.
Its checksum is written alongside it in `dist/SHA256SUMS`.

The source ZIP has no Git history or remote configured. To publish a repository,
extract it, initialize Git, commit the files, and connect your chosen GitHub remote.
No repository or package registry is created by the build scripts.

## License

This project is distributed under the MIT license in `LICENSE`.
