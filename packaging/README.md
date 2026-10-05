# Packaging

The service is fully offline, so releases ship the Hebrew model with the binary.

## Windows installer (Inno Setup wizard)

Needs [Inno Setup 6](https://jrsoftware.org/isinfo.php) on the build machine
(`winget install JRSoftware.InnoSetup`).

```powershell
cargo build --release -p dictation-service
.\packaging\build-installer.ps1 -Exe .\target\release\dictation.exe -ModelDir <ivrit-turbo folder> -Version 1.0.0
```

Output: `dictation-setup-1.0.0.exe` plus `dictation-setup-1.0.0-1.bin` and
`-2.bin`. **Copy all three files together** (the model is about 3.2 GB, so the
installer is split into two parts of at most 2 GB). Run the .exe on the target
machine; no network and no admin rights are needed.

The wizard installs `dictation.exe` and `models\ivrit-turbo` to
`%LOCALAPPDATA%\Programs\Dictation`, verifies the model checksums, adds Start
menu entries (Dictation, Dictation settings, Uninstall), registers the app in
Windows **Apps & features**, and offers two options: start at sign-in and a
desktop shortcut. The uninstaller removes the program and the autostart entry
but keeps your `config.toml`. Silent install for IT:
`dictation-setup-1.0.0.exe /VERYSILENT /TASKS=autostart /DIR=C:\Apps\Dictation`.

## Windows (plain zip bundle)

`.\packaging\make-bundle.ps1 -Exe ... -ModelDir ...` produces
`dist\dictation-windows\` (exe, model, `SHA256SUMS`, `install.ps1`) for
environments that prefer a folder copy; run `.\install.ps1` from it (per-user,
Start menu shortcut, no registration).
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