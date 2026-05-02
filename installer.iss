; RAVE-TFG Inno Setup Installer Script
; Requires: Inno Setup 6.x (https://jrsoftware.org/isdl.php)

[Setup]
AppId={{8F3C4A2E-1B5D-4F9A-A7C6-3E2D1F0B9A8C}
AppName=RAVE-TFG
AppVersion=0.4.2
AppPublisher=RAVE-TFG
DefaultDirName={autopf}\RAVE-TFG
DefaultGroupName=RAVE-TFG
OutputDir=.\installer-output
OutputBaseFilename=RAVE-TFG-Setup-0.4.2
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile=app\ui\assets\app_icon.ico
UninstallDisplayIcon={app}\RAVE-TFG.exe
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\RAVE-TFG\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\RAVE-TFG"; Filename: "{app}\RAVE-TFG.exe"
Name: "{group}\{cm:UninstallProgram,RAVE-TFG}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\RAVE-TFG"; Filename: "{app}\RAVE-TFG.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\RAVE-TFG.exe"; Description: "{cm:LaunchProgram,RAVE-TFG}"; Flags: nowait postinstall skipifsilent
