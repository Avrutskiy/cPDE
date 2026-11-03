"""Create a portable, GitHub-ready source ZIP without caches or machine data."""
import hashlib
import json
from pathlib import Path
import re
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    version = re.search(r'^version = "([^"]+)"',
                        (root / 'pyproject.toml').read_text(), re.MULTILINE).group(1)
    prefix = 'cpde-' + version
    top_files = ['.gitignore', '.gitattributes', 'LICENSE', 'README.md', 'API_REFERENCE.md',
                 'CONTRIBUTING.md', 'CHANGELOG.md', 'CODE_REVIEW.md', 'IMPLEMENTATION_NOTES.md',
                 'IMPLEMENTATION_PLAN.md', 'KNOWN_ISSUES.md', 'VALIDATION.md',
                 'RELEASE_NOTES.md', 'RELEASE_UPLOAD.md', 'CODE_PROVENANCE.md', 'BUILDING.md', 'CMakeLists.txt', 'pyproject.toml',
                 'MANIFEST.in']
    suffixes = {'.py', '.cu', '.cuh', '.cpp', '.hpp', '.h', '.md', '.yml', '.yaml', '.patch', '.sh', '.ps1'}
    paths = [root / name for name in top_files]
    for directory in ['cpde', 'cuda', 'cpu', 'tests', 'scripts', 'docs', '.github']:
        paths.extend(path for path in (root / directory).rglob('*')
                     if path.is_file() and path.suffix in suffixes
                     and '__pycache__' not in path.parts)
    # Keep the source archive aligned with the documented sample set.
    examples = ['__init__', 'attached_variation', 'fit_function', 'fit_function_explicit', 'fit_function_vanilla', 'poisson_disk', 'poisson_disk_vanilla', 'schrodinger_circle', 'schrodinger_circle_vanilla']
    paths.extend(root/'examples'/(name+'.py') for name in examples)
    paths.append(root/'examples/README.md')
    paths.extend(path for path in (root/'examples/results').iterdir()
                 if path.is_file() and path.suffix in {'.png', '.pdf', '.json'})
    paths.extend(path for path in (root/'docs/benchmarks').rglob('*')
                 if path.is_file() and path.suffix in {'.csv', '.json'})
    contents = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError('Refusing symlink in source bundle: ' + str(path))
        contents[path.relative_to(root).as_posix()] = path.read_bytes()
    manifest = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                for name, data in contents.items()}
    contents['BUNDLE_MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    destination = root / 'dist'
    destination.mkdir(exist_ok=True)
    archive = destination / (prefix + '-source.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(prefix + '/' + name, date_time=(2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, data)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError('Source ZIP integrity check failed')
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (destination / 'SHA256SUMS').write_text(checksum + '  ' + archive.name + '\n')
    print(archive)
    print(f'{len(manifest)} source files; SHA256 {checksum}')


if __name__ == '__main__':
    main()
