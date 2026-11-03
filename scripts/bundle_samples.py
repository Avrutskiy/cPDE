"""Bundle the release samples, guide and rendered figures; no library or checkpoints."""
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    version = re.search(r'^version = "([^"]+)"', (root/'pyproject.toml').read_text(), re.M).group(1)
    prefix = f'cpde-{version}-samples'
    names = ['__init__', 'fit_function_vanilla', 'fit_function', 'fit_function_explicit',
             'poisson_disk_vanilla', 'poisson_disk', 'schrodinger_circle_vanilla',
             'schrodinger_circle', 'attached_variation']
    guide = (root/'examples/README.md').read_text()
    # The guide moves one directory up in the archive; keep its links portable.
    guide = re.sub(r'\]\(([^)# ]+\.py)\)', r'](examples/\1)', guide)
    guide = guide.replace('](results/', '](examples/results/')
    guide = guide.replace('](../API_REFERENCE.md)', '](API_REFERENCE.md)')
    guide = guide.replace('](../BUILDING.md)', '](BUILDING.md)')
    files = {'README.md': guide.encode(),
             'API_REFERENCE.md': (root/'API_REFERENCE.md').read_bytes(),
             'VALIDATION.md': (root/'VALIDATION.md').read_bytes(),
             'BUILDING.md': (root/'BUILDING.md').read_bytes(),
             'KNOWN_ISSUES.md': (root/'KNOWN_ISSUES.md').read_bytes(),
             'CODE_PROVENANCE.md': (root/'CODE_PROVENANCE.md').read_bytes(),
             'LICENSE': (root/'LICENSE').read_bytes()}
    patch_name = 'docs/patches/legacy-repairs-before-R08.APPROVED-NOT-APPLIED.patch'
    files[patch_name] = (root/patch_name).read_bytes()
    for name in names:
        path = root/'examples'/f'{name}.py'
        if path.is_symlink():
            raise ValueError(f'Refusing symlink: {path}')
        data = path.read_bytes()
        ast.parse(data, filename=str(path))
        files[f'examples/{name}.py'] = data
        if name not in ('__init__', 'attached_variation'):
            for extension in ('png', 'pdf'):
                figure = root/'examples/results'/f'{name}.{extension}'
                if figure.is_symlink():
                    raise ValueError(f'Refusing symlink: {figure}')
                files[f'examples/results/{figure.name}'] = figure.read_bytes()
    files['examples/results/provenance.json'] = (root/'examples/results/provenance.json').read_bytes()
    manifest = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                for name, data in sorted(files.items())}
    files['BUNDLE_MANIFEST.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    dest = root/'dist'
    dest.mkdir(exist_ok=True)
    archive = dest/(prefix+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(prefix+'/'+name, date_time=(2026,11,3,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, data)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for name, entry in manifest.items():
            data = bundle.read(prefix+'/'+name)
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            assert len(data) == entry['bytes']
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n')
    print(f'{archive}\n{len(manifest)} files plus manifest; SHA256 {checksum}')


if __name__ == '__main__':
    main()
