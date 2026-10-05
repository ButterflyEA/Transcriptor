# Packaging

The service is fully offline, so releases ship the Hebrew model with the binary.

## Windows (zip bundle)

```powershell
cargo build --release -p dictation-service
.\packaging\make-bundle.ps1 -Exe .\target\release\dictation.exe -ModelDir <ivrit-turbo folder>
```

This writes `packaging\dist\dictation-windows\` with `dictation.exe`,
`models\ivrit-turbo\{config.json,model.safetensors,SHA256SUMS}` and
`install.ps1`. Zip it (the model alone is about 3.2 GB, above the 2 GB limit of
NSIS installers, so a zip or a plain folder copy is used instead of an .exe
installer). On the target machine, unzip and run `.\install.ps1` (per-user, no
admin needed; `-Dest` chooses another folder). The script verifies the model
checksums and adds a Start menu shortcut.

## Linux (tarball)

```
dictation-linux/
  bin/dictation
  share/dictation/models/ivrit-turbo/{config.json,model.safetensors,SHA256SUMS}
```

Unpack to `/usr` (or `/usr/local` plus a symlink of the models to
`/usr/share/dictation/models`). The service looks in
`/usr/share/dictation/models/<model>` automatically. Create the tarball with
`tar czf dictation-linux.tgz bin share` after generating `SHA256SUMS` with
`sha256sum config.json model.safetensors > SHA256SUMS` inside the model folder.

## Checking a transfer

`dictation verify-model [folder]` recomputes the checksums in `SHA256SUMS` and
exits non-zero on any missing or changed file.