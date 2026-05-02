; RAVE-TFG Inno Setup 6.x script
;
; Prerequisites:
;   - Place uv.exe in this installer\ directory (see BUILD.md)
;
; Build from repo root:
;   & "C:\Program Files (x86)\Inno Setup 6\iscc.exe" installer\setup.iss
;
; Output: dist\installer\RAVE-TFG-Setup-0.4.2.exe  (~15 MB)
; Install time: 5-10 min (downloads PyTorch ~2 GB from pytorch.org)

#define AppName    "RAVE-TFG"
#define AppVersion "0.4.2"
#define AppPublisher "Guillem Garcia"

[Setup]
AppId={{A3F2C1D0-8E4B-4F7A-9C6D-2B5E8A1F3D7C}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist\installer
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\src\*"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements-install.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "launcher.pyw"; DestDir: "{app}"; Flags: ignoreversion
Source: "uv.exe"; DestDir: "{app}\.uv"; Flags: ignoreversion

[Run]
Filename: "{app}\.uv\uv.exe"; Parameters: "venv .venv --python 3.10 --seed"; WorkingDir: "{app}"; StatusMsg: "Creating Python 3.10 virtual environment..."; Flags: runhidden waituntilterminated
Filename: "{app}\.uv\uv.exe"; Parameters: "pip install --python {app}\.venv\Scripts\python.exe -r requirements-install.txt --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple"; WorkingDir: "{app}"; StatusMsg: "Installing dependencies (downloading PyTorch CUDA ~2 GB, please wait)..."; Flags: runhidden waituntilterminated
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\.uv"
