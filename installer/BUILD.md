# Build del instalador de RAVE-TFG

## Programa utilizado

**Inno Setup 6** — generador de instaladores para Windows.
Descarga: https://jrsoftware.org/isinfo.php

## Qué genera

Un único ejecutable ligero (`RAVE-TFG-Setup-0.4.2.exe`, ~15 MB) que al ejecutarse:

1. Copia los archivos de la aplicación (`app/`, `src/`, `pyproject.toml`).
2. Crea un entorno virtual Python 3.10 con `uv`.
3. Descarga e instala todas las dependencias (PyTorch CUDA ~2 GB, PySide6, etc.).
4. Crea accesos directos en el menú Inicio y, opcionalmente, en el escritorio.

El tiempo de instalación es de 5–10 minutos dependiendo de la conexión.

## Comando para compilar

Desde la raíz del repositorio:

```
"C:\Program Files (x86)\Inno Setup 6\iscc.exe" installer\setup.iss
```

El ejecutable resultante se deposita en `dist\installer\RAVE-TFG-Setup-0.4.2.exe`.

## Requisitos previos

- Inno Setup 6 instalado en la ruta por defecto.
- `installer\uv.exe` presente (ya incluido en el repositorio).

## Archivos relevantes

| Archivo | Descripción |
|---|---|
| `installer/setup.iss` | Script de Inno Setup — define qué se empaqueta y cómo se instala |
| `installer/uv.exe` | Gestor de paquetes Python incluido en el instalador |
| `installer/requirements-install.txt` | Dependencias que se instalan en el equipo del usuario |
| `installer/launcher.pyw` | Punto de entrada de la app (sin ventana de consola) |
