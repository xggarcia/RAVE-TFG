"""Design tokens — single source of truth for colors and fonts.

Ported from `design-mock/styles/tokens.css`. Import from here, do not redefine
these constants in individual widget/page modules.
"""

# Background scale (dark → light)
BG0 = "#1e2320"
BG1 = "#232926"
BG2 = "#28302b"
BG3 = "#2f3832"
BG4 = "#37413a"

# Foreground scale (bright → dim)
FG0 = "#f0f4ee"
FG1 = "#c6cdc3"
FG2 = "#96a092"
FG3 = "#717870"

# Lines / borders
LINE0 = "#333c36"
LINE1 = "#3f4b42"
LINE2 = "#4e5c51"

# Accents
ACID = "#a8e63d"
ACIDDIM = "#6aad1e"
ACIDBG = "#384d28"
AMBER = "#e8c040"
MAG = "#e0406a"
BLUE = "#5090d8"

# Monospace font snippet for inline QSS.
MONO = "font-family:'JetBrains Mono','Consolas',monospace;"
