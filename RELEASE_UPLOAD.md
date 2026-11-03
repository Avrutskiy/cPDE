# Upload cPDE 0.2.0 to GitHub

No repository or GitHub release has been created by this preparation step.
All files needed for the upload are local.

1. Create the desired GitHub repository.
2. Extract `assets/cpde-0.2.0-source.zip`. Upload the **contents** of its `cpde-0.2.0/`
   directory to the repository root, preserving directories and dotfiles. Do not
   upload only the ZIP as the repository's source: that would hide the README,
   code and examples from normal browsing.
3. Create a release with tag **`v0.2.0`**, targeting the uploaded source commit,
   and title **`cPDE 0.2.0`**.
4. Paste `RELEASE_NOTES.md` into the release description.
5. Attach all files from `assets/`: source ZIP, Python wheel, Python sdist,
   samples ZIP and `SHA256SUMS`.
6. Publish when the repository contents and release description look correct.

The source archive is intended for a Git client or GitHub Desktop as well as
manual upload. Git handles nested directories and dotfiles more reliably for a
complete repository. This upload kit itself is not an asset that users need;
its `assets/` directory contains the downloadable release files.

Users build `cpde_cpu_server` or `cpde_server` locally with the included macOS,
Linux and Windows scripts. Only the CUDA backend requires locally installed
CUDA. There is no prebuilt platform-specific executable asset; see `BUILDING.md`
in the source archive.
