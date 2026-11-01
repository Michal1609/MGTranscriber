; Inno Setup script for a Windows setup .exe (build: see docs/RELEASING.md).
; The .exe only copies the app and runs install.ps1, which does the real work (GPU check, uv, ffmpeg,
; Deno, Python packages, HF token). Uninstall runs install.ps1 -Uninstall and keeps transcripts and data.
;   ISCC.exe /DAppVersion=0.2.0 installer\mgtranscriber.iss   ->  installer\Output\MGTranscriber-setup-0.2.0.exe

#define AppName "MGTranscriber"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{B7D2E4A1-5C93-4F68-A0E7-3D91C2F8B614}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Michal Grznar
AppPublisherURL=https://github.com/Michal1609/MGTranscriber
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename={#AppName}-setup-{#AppVersion}
LicenseFile=..\LICENSE
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}
; On Windows 11 (build 26200) the RedirectionGuard mitigation reaches install.ps1 too, and uv then fails on
; its own junction for the managed Python ("untrusted mount point", os error 448). Setup only writes to the
; user's own %LOCALAPPDATA%, where the guard protects nothing.
RedirectionGuard=no

[Files]
Source: "..\mgtranscriber\*"; DestDir: "{app}\mgtranscriber"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs ignoreversion
Source: "..\static\*"; DestDir: "{app}\static"; Flags: recursesubdirs ignoreversion
Source: "..\docs\*"; DestDir: "{app}\docs"; Flags: recursesubdirs ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\uv.lock"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\install.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\web.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\start.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\server.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\transcribe.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\.env.example"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\web.bat"; WorkingDir: "{app}"; IconFilename: "{sys}\imageres.dll"; IconIndex: 103; Flags: runminimized
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\web.bat"; WorkingDir: "{app}"; IconFilename: "{sys}\imageres.dll"; IconIndex: 103; Flags: runminimized; Tasks: desktopicon

; an upgrade starts from a clean code tree, so modules deleted upstream do not linger (data, output and .env live elsewhere)
[InstallDelete]
Type: filesandordirs; Name: "{app}\mgtranscriber"
Type: filesandordirs; Name: "{app}\static"
Type: filesandordirs; Name: "{app}\docs"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"

[Run]
; a console window: the installer asks for the HuggingFace token there and shows the download progress
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -NoShortcuts -NoAppsEntry"; \
  WorkingDir: "{app}"; StatusMsg: "Installing Python packages and tools (3-5 GB on the first run)..."; Check: not WizardSilent; Flags: waituntilterminated
; silent install (/VERYSILENT): no questions; the HF token comes from the HF_TOKEN environment variable, if set
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -NoShortcuts -NoAppsEntry -NonInteractive -Ollama no"; \
  WorkingDir: "{app}"; StatusMsg: "Installing Python packages and tools (3-5 GB on the first run)..."; Check: WizardSilent; Flags: runhidden waituntilterminated
Filename: "{app}\web.bat"; Description: "Start {#AppName}"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent shellexec runminimized

[UninstallRun]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -Uninstall -NonInteractive"; \
  WorkingDir: "{app}"; RunOnceId: "RemoveVenv"; Flags: runhidden waituntilterminated
