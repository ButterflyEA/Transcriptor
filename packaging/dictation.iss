; Inno Setup script for the offline Dictation installer.
; Build: .\build-installer.ps1 (see packaging\README.md). Needs no network to install or run.
#ifndef BundleDir
  #error Pass /DBundleDir=<folder made by make-bundle.ps1>
#endif
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef OutDir
  #define OutDir "dist"
#endif

[Setup]
AppId={{6F0B7C0E-52D1-4B7B-9A0C-0D1C7B1E5A11}
AppName=Dictation
AppVersion={#AppVersion}
AppPublisher=ButterflyEA
DefaultDirName={autopf}\Dictation
DefaultGroupName=Dictation
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutDir}
OutputBaseFilename=dictation-setup-{#AppVersion}
UninstallDisplayIcon={app}\dictation.exe
UninstallDisplayName=Dictation (offline Hebrew dictation)
LicenseFile={#BundleDir}\THIRD_PARTY_NOTICES.md
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
WizardStyle=modern
; The bundled model is about 3.2 GB; a single installer exe cannot hold that.
DiskSpanning=yes
DiskSliceSize=2000000000
Compression=lzma2/fast
SolidCompression=no
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "autostart"; Description: "Start Dictation when I sign in"; GroupDescription: "Options:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Options:"; Flags: unchecked

[Files]
Source: "{#BundleDir}\dictation.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BundleDir}\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Dictation"; Filename: "{app}\dictation.exe"; Parameters: "run"
Name: "{group}\Dictation settings"; Filename: "{app}\dictation.exe"; Parameters: "settings"
Name: "{group}\Uninstall Dictation"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Dictation"; Filename: "{app}\dictation.exe"; Parameters: "run"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Dictation"; ValueData: """{app}\dictation.exe"" run"; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\dictation.exe"; Parameters: "run"; Description: "Start Dictation now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\dictation.exe"; Parameters: "stop"; Flags: runhidden; RunOnceId: "StopRecording"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    // Confirms the copied model matches the checksums shipped with it.
    if (not Exec(ExpandConstant('{app}\dictation.exe'), 'verify-model "' + ExpandConstant('{app}\models\ivrit-turbo') + '"', '', SW_HIDE, ewWaitUntilTerminated, Code)) or (Code <> 0) then
      MsgBox('The model files failed checksum verification. Run the installer again from a fresh copy.', mbError, MB_OK);
  end;
end;