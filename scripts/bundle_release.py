"""Prepare local GitHub release assets and an upload kit; never publish anything.

Requires the project's dev extras (build, setuptools and wheel). Servers are built
locally by users; no executable or CUDA runtime is included in these archives.
"""
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    version = re.search(r'^version = "([^"]+)"', (root/'pyproject.toml').read_text(), re.M).group(1)
    dist = root/'dist'
    subprocess.run([sys.executable, '-m', 'build', '--no-isolation'], cwd=root, check=True)
    for script in ['bundle.py', 'bundle_samples.py']:
        subprocess.run([sys.executable, str(root/'scripts'/script)], cwd=root, check=True)
    names = [f'cpde-{version}-source.zip', f'cpde-{version}-samples.zip',
             f'cpde_interface-{version}-py3-none-any.whl', f'cpde_interface-{version}.tar.gz']
    checksums = ''.join(f'{hashlib.sha256((dist/name).read_bytes()).hexdigest()}  {name}\n'
                        for name in sorted(names))
    (dist/'SHA256SUMS').write_text(checksums)
    kit = dist/f'cpde-{version}-github-release'
    assets = kit/'assets'
    assets.mkdir(parents=True, exist_ok=True)
    for name in names+['SHA256SUMS']:
        shutil.copyfile(dist/name, assets/name)
    shutil.copyfile(root/'RELEASE_NOTES.md', kit/'RELEASE_NOTES.md')
    shutil.copyfile(root/'RELEASE_UPLOAD.md', kit/'START_HERE.md')
    # Package only the explicit inventory; stale local files cannot enter the kit.
    inventory = [kit/'START_HERE.md', kit/'RELEASE_NOTES.md']+[assets/name for name in names+['SHA256SUMS']]
    archive = dist/(kit.name+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(inventory):
            info = zipfile.ZipInfo(kit.name+'/'+path.relative_to(kit).as_posix(),
                                   date_time=(2026,11,3,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for line in checksums.splitlines():
            expected,name=line.split('  ',1)
            actual=hashlib.sha256(bundle.read(kit.name+'/assets/'+name)).hexdigest()
            if expected != actual: raise ValueError(f'Checksum mismatch: {name}')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n')
    print(f'Upload kit: {archive}\nAssets: {assets}\nSHA256: {digest}')


if __name__ == '__main__':
    main()
